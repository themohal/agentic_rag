# Agentic RAG — Chat with your PDFs

Upload one or more PDFs in the sidebar and chat with them. The app uses an
**agentic RAG** pipeline built with **LangGraph**: it retrieves from your
documents first, grades whether the retrieved chunks actually answer the
question, and — if they don't — falls back to a **Tavily** web search.

## Stack

- **Streamlit** — UI (sidebar uploader + chat)
- **Pinecone** — serverless vector store
- **OpenAI** — `gpt-4o-mini` for generation, `text-embedding-3-small` for embeddings
- **LangGraph** — orchestration (`retrieve → grade → route → web_search / generate`)
- **Tavily** — web search fallback

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell: venv\Scripts\Activate.ps1)
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
copy .env.example .env       # Windows  (cp on macOS/Linux)
# then edit .env and fill in the three keys
```

Required keys in `.env`:

| Key | Where to get it |
| --- | --- |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `PINECONE_API_KEY` | https://app.pinecone.io/ |
| `TAVILY_API_KEY` | https://app.tavily.com/ |

`PINECONE_INDEX_NAME` defaults to `pdf-rag`; the index is created automatically
on first run (serverless, `aws` / `us-east-1`, 1536 dims / cosine).

## Run

```bash
streamlit run app.py
```

Then:

1. Upload one or more PDFs in the sidebar and click **Process documents**.
2. Ask questions in the chat.
   - A **📄** badge means the answer came from your documents.
   - A **🌐** badge means the app fell back to a web search.

## How it works

Each browser session gets its own Pinecone **namespace** (a random UUID), so
uploads stay isolated per session. The relevance grader is the "agentic" part:
it decides, per question, whether the PDFs are enough or the web is needed.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI (sidebar upload + chat) |
| `ingest.py` | PDF → chunk → embed → upsert to Pinecone |
| `agent.py` | LangGraph agentic RAG graph |
| `config.py` | Env vars and constants |
