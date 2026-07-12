"""Central configuration: loads environment variables and defines constants."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- API keys -------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Pinecone -------------------------------------------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Models ---------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # must match EMBEDDING_MODEL
CHAT_MODEL = "gpt-4o-mini"

# --- Chunking / retrieval -------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
TAVILY_MAX_RESULTS = 3


def missing_keys() -> list[str]:
    """Return the names of any required API keys that are not set."""
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }
    return [name for name, value in required.items() if not value]
