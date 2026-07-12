"""Agentic RAG built with LangGraph.

Flow:  retrieve -> grade_documents -> (relevant?) -> generate
                                     \\-> web_search -> generate

The relevance grader is the agentic gate: if the retrieved PDF chunks don't
answer the question, the graph falls back to a Tavily web search.
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
    source: Literal["documents", "web"]
    web_search: bool


# --- Structured grader output ---------------------------------------------
class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved documents."""

    binary_score: str = Field(
        description="Are the documents relevant to the question? 'yes' or 'no'."
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
def decide_route(state: GraphState) -> Literal["web_search", "generate"]:
    return "web_search" if state["web_search"] else "generate"


@functools.lru_cache(maxsize=1)
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
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
        }
    )
    return {
        "answer": final["generation"],
        "source": final["source"],
        "documents": final["documents"],
    }
