"""Simple Streamlit client for the FastAPI RAG service.

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 120


def api_url(base_url: str, endpoint: str) -> str:
    """Build a URL without duplicate or missing slashes."""
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def show_api_error(response: requests.Response) -> None:
    """Display a useful API error, including non-JSON responses."""
    try:
        detail: Any = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text or "The API returned an empty error response."
    st.error(f"API error ({response.status_code}): {detail}")


st.set_page_config(page_title="Document Q&A", page_icon=None, layout="centered")
st.title("Document Q&A")
st.caption("Upload documents to the RAG service, then ask questions grounded in those documents.")

with st.sidebar:
    st.header("Service connection")
    base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).strip()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if st.button("New conversation", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.chat_messages = []
        st.rerun()
    if st.session_state.conversation_id:
        st.caption(f"Conversation: {st.session_state.conversation_id}")
    if st.button("Check service", use_container_width=True):
        try:
            response = requests.get(api_url(base_url, "/healthz"), timeout=10)
            response.raise_for_status()
            st.success("Service is reachable.")
        except requests.RequestException as exc:
            st.error(f"Could not reach the service: {exc}")

upload_tab, query_tab = st.tabs(["Upload documents", "Ask a question"])

with upload_tab:
    st.subheader("Add documents to the shared index")
    uploads = st.file_uploader(
        "Choose TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="The API enforces a per-file upload size limit.",
    )
    if st.button("Upload and index", type="primary", disabled=not uploads):
        try:
            multipart_files = [
                (
                    "files",
                    (item.name, item.getvalue(), item.type or "application/octet-stream"),
                )
                for item in uploads
            ]
            response = requests.post(
                api_url(base_url, "/upload"),
                files=multipart_files,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not response.ok:
                show_api_error(response)
            else:
                result = response.json()
                st.success(
                    f"Indexed {result.get('chunk_count', 0)} chunk(s) from "
                    f"{len(result.get('indexed_files', []))} file(s)."
                )
                st.write("Files:", ", ".join(result.get("indexed_files", [])))
        except requests.RequestException as exc:
            st.error(f"Upload request failed: {exc}")
        except ValueError as exc:
            st.error(f"The API returned an invalid response: {exc}")

with query_tab:
    st.subheader("Ask about indexed documents")
    with st.form("query_form"):
        question = st.text_area("Question", placeholder="What do the documents say about ...?")
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                response = requests.post(
                    api_url(base_url, "/query"),
                    json={
                        "query": question.strip(),
                        "conversation_id": st.session_state.conversation_id,
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if not response.ok:
                    show_api_error(response)
                else:
                    result = response.json()
                    st.session_state.conversation_id = result.get("conversation_id")
                    st.session_state.chat_messages.extend(
                        [
                            {"role": "user", "content": question.strip(), "sources": []},
                            {
                                "role": "assistant",
                                "content": result.get("answer", "No answer returned."),
                                "sources": result.get("sources", []),
                            },
                        ]
                    )
            except requests.RequestException as exc:
                st.error(f"Query request failed: {exc}")
            except (ValueError, TypeError) as exc:
                st.error(f"The API returned an invalid response: {exc}")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            for source in message.get("sources", []):
                source_name = source.get("source", "unknown")
                chunk_id = source.get("chunk_id", "unknown")
                score = source.get("score")
                label = f"{source_name} · chunk {chunk_id}"
                if score is not None:
                    label += f" · score {score:.4f}"
                with st.expander(label):
                    st.write(source.get("content", ""))
