"""Streamlit UI for the agentic RAG application."""
import uuid

import streamlit as st

import config
from agent import run_agent
from ingest import process_pdfs

st.set_page_config(page_title="Agentic RAG — Chat with your PDFs", page_icon="📚")

# --- Session state --------------------------------------------------------
if "namespace" not in st.session_state:
    st.session_state.namespace = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False

# --- Key check ------------------------------------------------------------
missing = config.missing_keys()
if missing:
    st.error(
        "Missing API keys: "
        + ", ".join(missing)
        + ". Add them to your `.env` file (see `.env.example`)."
    )
    st.stop()

# --- Sidebar: upload & process -------------------------------------------
with st.sidebar:
    st.header("📄 Your documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Reading, chunking, embedding, and indexing…"):
                count = process_pdfs(uploaded_files, st.session_state.namespace)
            if count:
                st.session_state.docs_ready = True
                st.success(f"Indexed {count} chunks from {len(uploaded_files)} file(s).")
            else:
                st.warning("No extractable text found in the uploaded PDF(s).")

    if st.session_state.docs_ready:
        st.caption("✅ Documents are indexed and ready to chat.")
    else:
        st.caption("Upload PDFs and click *Process* to chat with them.")

# --- Main: chat -----------------------------------------------------------
st.title("Agentic RAG 📚")
st.caption(
    "Ask about your uploaded PDFs. If the answer isn't in them, "
    "I'll search the web with Tavily."
)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("badge"):
            st.caption(msg["badge"])

# Handle new input
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = run_agent(prompt, st.session_state.namespace)

        answer = result["answer"]
        source = result["source"]
        badge = (
            "📄 Answered from your documents"
            if source == "documents"
            else "🌐 Answered from web search"
        )

        st.markdown(answer)
        st.caption(badge)

        docs = result["documents"]
        if docs:
            with st.expander("Sources"):
                for i, doc in enumerate(docs, start=1):
                    meta = doc.metadata
                    label = meta.get("source", "unknown")
                    page = meta.get("page")
                    header = f"**{i}. {label}**" + (f" (p. {page})" if page else "")
                    st.markdown(header)
                    snippet = doc.page_content[:500]
                    st.caption(snippet + ("…" if len(doc.page_content) > 500 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "badge": badge}
    )
