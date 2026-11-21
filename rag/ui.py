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
    # "basic" is the original single-step RAG. It stays in the UI because
    # seeing the two answers side by side on the same question is the clearest
    # demonstration of what the agent layer buys -- and of what it costs in
    # latency.
    mode = st.radio(
        "Mode",
        ["agents", "basic"],
        help="agents = supervisor -> sql/rag -> writer -> critic. "
             "basic = the original single-step RAG (the eval baseline).",
    )
    with_critic = True
    if mode == "agents":
        with_critic = st.checkbox(
            "Run the critic", value=True,
            help="Off = multi-agent without self-reflection, one of the "
                 "configurations the eval compares.",
        )
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
                json={"question": question, "k": top_k,
                      "temperature": temperature, "mode": mode,
                      "with_critic": with_critic},
                # Up to five sequential model calls at several seconds each,
                # so the agent path genuinely needs minutes, not 120s.
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]
            citations = data.get("citations", [])

            placeholder.markdown(answer)

            # How the answer was produced, not just what it was. This is the
            # part that makes the system inspectable rather than a black box:
            # the route, the SQL that ran, any ReAct retries, and whether the
            # critic sent the draft back.
            if data.get("mode") == "agents":
                path = " → ".join(data.get("agent_path") or [])
                route = ", ".join(data.get("route") or []) or "none (refused)"
                verdict = data.get("critic_verdict") or "—"
                with st.expander(f"🧭 How this was answered — {path}", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Route", route)
                    c2.metric("Revisions", data.get("revisions") or 0)
                    c3.metric("Critic", verdict)

                    if data.get("route_reason"):
                        st.caption(f"**Why that route:** {data['route_reason']}")

                    if data.get("sql_query"):
                        st.markdown(
                            f"**SQL** ({data.get('sql_row_count') or 0} rows, "
                            f"{data.get('sql_attempts') or 1} attempt(s))"
                        )
                        st.code(data["sql_query"], language="sql")
                        if (data.get("sql_attempts") or 1) > 1:
                            st.caption(
                                "More than one attempt means the agent hit an "
                                "error and corrected itself (ReAct)."
                            )

                    if data.get("critique"):
                        st.warning(f"**The critic rejected a draft:** {data['critique']}")

                    usage = data.get("usage") or {}
                    cost = usage.get("total_cost_usd")
                    st.caption(
                        f"trace `{data.get('trace_id')}` · "
                        f"{(data.get('duration_ms') or 0) / 1000:.1f}s · "
                        f"in {usage.get('total_input_tokens', 0)} / "
                        f"out {usage.get('total_output_tokens', 0)} / "
                        f"thinking {usage.get('total_thinking_tokens', 0)} tokens · "
                        f"cost {('$%.6f' % cost) if cost is not None else 'unpriced'}"
                    )

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