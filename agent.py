"""Agentic RAG built with LangGraph.

Flow:  classify ─(chitchat)─> chitchat ─> END
                └(question)─> retrieve -> grade_documents ─(relevant)─> generate
                                                          └(none)────> web_search -> generate

Two agentic gates:
  1. classify — greetings / small talk answer directly (no retrieval, no web).
  2. grade_documents — if the retrieved PDF chunks don't answer the question,
     fall back to a Tavily web search.
"""
from __future__ import annotations

import functools
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient

import config
from ingest import get_vectorstore


# --- State ----------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    namespace: str
    documents: list[Document]
    generation: str
    source: Literal["documents", "web", "chitchat"]
    web_search: bool
    intent: Literal["chitchat", "question"]


# --- Structured grader output ---------------------------------------------
class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved documents."""

    binary_score: str = Field(
        description="Are the documents relevant to the question? 'yes' or 'no'."
    )


class ClassifyIntent(BaseModel):
    """Whether the user input needs document/web lookup or is just small talk."""

    intent: Literal["chitchat", "question"] = Field(
        description=(
            "'chitchat' for greetings, thanks, or small talk that needs no "
            "information lookup (e.g. 'hi', 'how are you', 'thanks'). "
            "'question' for anything that asks for information."
        )
    )


@functools.lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=0,
        api_key=config.OPENAI_API_KEY,
    )


@functools.lru_cache(maxsize=1)
def _tavily() -> TavilyClient:
    return TavilyClient(api_key=config.TAVILY_API_KEY)


# --- Nodes ----------------------------------------------------------------
def classify(state: GraphState) -> dict:
    """Decide whether the input is small talk or a real information request."""
    classifier = _llm().with_structured_output(ClassifyIntent)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's message. Return 'chitchat' for greetings, "
                "thanks, or small talk that needs no information lookup. Return "
                "'question' for anything asking for information.",
            ),
            ("human", "{question}"),
        ]
    )
    try:
        result = (prompt | classifier).invoke({"question": state["question"]})
        intent = result.intent
    except Exception:
        intent = "question"  # safe default: fall through to RAG
    return {"intent": intent}


def chitchat(state: GraphState) -> dict:
    """Answer greetings / small talk directly, without retrieval or web search."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant for a PDF question-answering app. "
                "Reply briefly to the greeting or small talk, and invite the user "
                "to ask about their uploaded documents.",
            ),
            ("human", "{question}"),
        ]
    )
    answer = (prompt | _llm()).invoke({"question": state["question"]}).content
    return {"generation": answer, "source": "chitchat", "documents": []}


def retrieve(state: GraphState) -> dict:
    try:
        vectorstore = get_vectorstore(state["namespace"])
        docs = vectorstore.similarity_search(state["question"], k=config.TOP_K)
    except Exception:
        docs = []
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    """Keep only relevant chunks; flag a web search if none are relevant."""
    docs = state["documents"]
    if not docs:
        return {"documents": [], "web_search": True}

    grader = _llm().with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document "
                "to a user question. If the document contains keywords or "
                "meaning related to the question, grade it as relevant. Give a "
                "binary score 'yes' or 'no'.",
            ),
            ("human", "Document:\n\n{document}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | grader

    relevant: list[Document] = []
    for doc in docs:
        try:
            result = chain.invoke(
                {"document": doc.page_content, "question": state["question"]}
            )
            if result.binary_score.strip().lower() == "yes":
                relevant.append(doc)
        except Exception:
            continue

    return {"documents": relevant, "web_search": len(relevant) == 0}


def web_search(state: GraphState) -> dict:
    """Fall back to Tavily web search."""
    try:
        response = _tavily().search(
            state["question"], max_results=config.TAVILY_MAX_RESULTS
        )
        results = response.get("results", [])
    except Exception:
        results = []

    docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
        )
        for r in results
    ]
    return {"documents": docs, "source": "web"}


def generate(state: GraphState) -> dict:
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    if not context:
        context = "(no context found)"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the question using ONLY the "
                "context below. If the context does not contain the answer, say "
                "you don't know. Be concise.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | _llm()
    answer = chain.invoke(
        {"context": context, "question": state["question"]}
    ).content
    return {"generation": answer}


# --- Routing --------------------------------------------------------------
def route_intent(state: GraphState) -> Literal["chitchat", "retrieve"]:
    return "chitchat" if state["intent"] == "chitchat" else "retrieve"


def decide_route(state: GraphState) -> Literal["web_search", "generate"]:
    return "web_search" if state["web_search"] else "generate"


@functools.lru_cache(maxsize=1)
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("classify", classify)
    workflow.add_node("chitchat", chitchat)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_intent,
        {"chitchat": "chitchat", "retrieve": "retrieve"},
    )
    workflow.add_edge("chitchat", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_route,
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_agent(question: str, namespace: str) -> dict:
    """Run the agentic RAG graph. Returns {answer, source, documents}."""
    graph = build_graph()
    final = graph.invoke(
        {
            "question": question,
            "namespace": namespace,
            "documents": [],
            "generation": "",
            "source": "documents",
            "web_search": False,
            "intent": "question",
        }
    )
    return {
        "answer": final["generation"],
        "source": final["source"],
        "documents": final["documents"],
    }
