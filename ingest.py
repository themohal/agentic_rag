"""PDF ingestion: read PDFs, chunk, embed, and upsert to Pinecone."""
from __future__ import annotations

import functools

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

import config


@functools.lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


@functools.lru_cache(maxsize=1)
def _get_pinecone() -> Pinecone:
    return Pinecone(api_key=config.PINECONE_API_KEY)


def ensure_index() -> None:
    """Create the serverless Pinecone index if it doesn't already exist."""
    pc = _get_pinecone()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION,
            ),
        )


def get_vectorstore(namespace: str) -> PineconeVectorStore:
    """Return a LangChain vector store bound to the given namespace."""
    ensure_index()
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=_get_embeddings(),
        namespace=namespace,
        pinecone_api_key=config.PINECONE_API_KEY,
    )


def _pdf_to_documents(file) -> list[Document]:
    """Extract text from a Streamlit UploadedFile into per-page Documents."""
    reader = PdfReader(file)
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name, "page": page_num},
                )
            )
    return docs


def process_pdfs(uploaded_files, namespace: str) -> int:
    """Read, chunk, embed, and upsert the given PDFs. Returns chunk count."""
    raw_docs: list[Document] = []
    for file in uploaded_files:
        raw_docs.extend(_pdf_to_documents(file))

    if not raw_docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)

    vectorstore = get_vectorstore(namespace)
    vectorstore.add_documents(chunks)
    return len(chunks)
