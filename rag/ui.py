"""Streamlit web UI for the MoSPI RAG chatbot."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MoSPI Statistical Q&A", page_icon="📊", layout="wide")

st.title("📊 MoSPI Statistical Q&A")
st.caption("Ask questions about Indian statistics — answers are grounded in official MoSPI publications.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    st.divider()
    st.subheader("Rebuild Index")
    if st.button("🔄 Re-ingest corpus"):
        with st.spinner("Rebuilding vector index..."):
            try:
                r = requests.post(f"{API_URL}/ingest", timeout=300)
                r.raise_for_status()
                data = r.json()
                st.success(f"Index built: {data['vectors']} vectors")
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")
    st.divider()
    try:
        h = requests.get(f"{API_URL}/health", timeout=5)
        h.raise_for_status()
        st.success(f"API online · model: {h.json().get('model', 'unknown')}")
    except Exception:
        st.warning("API offline — make sure docker compose is running")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['title']}** ({c.get('date_published', 'n/a')})  \n[{c['url']}]({c['url']})")
                    st.caption(c.get("snippet", ""))

# Input
if question := st.chat_input("Ask a question about Indian statistics..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        citations_placeholder = st.empty()
        placeholder.markdown("_Thinking..._")

        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "k": top_k, "temperature": temperature},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]
            citations = data.get("citations", [])

            placeholder.markdown(answer)

            if citations:
                with citations_placeholder.expander("📎 Sources"):
                    for c in citations:
                        st.markdown(f"**{c['title']}** ({c.get('date_published', 'n/a')})  \n[{c['url']}]({c['url']})")
                        st.caption(c.get("snippet", "")[:200] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
            })

        except requests.exceptions.ConnectionError:
            placeholder.error("Cannot connect to API. Is the server running?")
        except Exception as exc:
            placeholder.error(f"Error: {exc}")