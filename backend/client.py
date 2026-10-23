"""Streamlit conversation and spatial PDF citation viewer."""
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer

# Local settings live in .env; Community Cloud supplies the same key via secrets.
load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="Talk to Documents", page_icon="📄", layout="wide")
API_URL = os.getenv("RAG_API_URL", "")
if not API_URL:
    try:
        API_URL = st.secrets["RAG_API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = ""
API_URL = str(API_URL).rstrip("/")
if not API_URL:
    st.error("RAG_API_URL is missing. Set it in .env or Streamlit Cloud secrets.")
    st.stop()
st.markdown("""
<style>
    .block-container {max-width: 1440px; padding-top: 2rem;}
    [data-testid="stSidebar"] {background: #f7f8fa;}
    [data-testid="stChatMessage"] {border: 1px solid #e7e9ed; border-radius: 16px;}
    .subtitle {color: #667085; font-size: 0.95rem; margin-top: -0.7rem;}
</style>
""", unsafe_allow_html=True)

# Keep conversations and active jobs in this browser session.
for key, default in {
    "conversations": {}, "pending_chat": None, "index_job": None,
    "upload_key": 0, "notice": None, "citation": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def api_request(method: str, path: str, **kwargs):
    """Raise a readable error for unsuccessful API responses."""
    response = requests.request(method, API_URL + path, timeout=(5, 90), **kwargs)
    if not response.ok:
        try:
            body = response.json()
            detail = body.get("message", body.get("detail", response.text))
            if body.get("suggested_action"):
                detail += f" {body['suggested_action']}"
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


@st.cache_data(show_spinner=False)
def pdf_bytes(source: str) -> bytes:
    """Cache the original PDF whose page coordinates are stored in Qdrant."""
    response = requests.get(f"{API_URL}/documents/{source}/file", timeout=(5, 90))
    response.raise_for_status()
    return response.content


def show_answer(message: dict, message_index: int):
    """Render the answer and numbered citation controls."""
    st.markdown(message["content"])
    sources = message.get("sources", [])
    if not sources:
        return
    st.caption("Sources · select a citation to view its location")
    columns = st.columns(min(len(sources), 4))
    for index, source in enumerate(sources):
        number = source.get("citation_id", index + 1)
        page = source.get("page_number")
        label = f"[{number}] {source.get('filename', 'Document')}"
        if page:
            label += f" · p. {page}"
        with columns[index % len(columns)]:
            if st.button(label, key=f"cite_{message_index}_{index}",
                         use_container_width=True):
                st.session_state.citation = source
                st.rerun()


def show_citation(source: dict):
    """Scroll to a source page and highlight its measured bounding box."""
    st.subheader("Source viewer")
    st.caption(source.get("filename", "Document"))
    locations = source.get("locations") or [source]
    page_locations = [item for item in locations if item.get("page_number")]
    if page_locations:
        location = st.selectbox(
            "Cited page", page_locations,
            format_func=lambda item: f"{item.get('filename', source.get('filename', 'Document'))} "
            f"· page {item['page_number']} · paragraph "
            f"{', '.join(item.get('paragraph_ids', [])) or 'source'}",
            key=f"location_{source.get('source')}_{source.get('chunk_id')}")
        page = int(location["page_number"])
        location_source = location.get("source", source["source"])
        box = location.get("bbox")
        annotations = []
        if box:
            annotations = [{"page": page, "x": box[0], "y": box[1],
                            "width": box[2] - box[0], "height": box[3] - box[1],
                            "color": "#F9B84A"}]
        try:
            pdf_viewer(
                pdf_bytes(location_source), height=680, width="100%",
                key=f"pdf_{location_source}_{page}_{box}",
                annotations=annotations, pages_to_render=[page],
                scroll_to_annotation=1 if annotations else None,
                scroll_to_page=None if annotations else page)
        except requests.RequestException as exc:
            st.error(f"Could not load the original PDF: {exc}")
    else:
        st.info("This citation comes from a text document.")
    with st.expander("Read cited passage"):
        st.write(source.get("text", ""))


# Refresh document readiness on every full rerun.
try:
    documents = api_request("GET", "/documents")
except (requests.RequestException, RuntimeError) as exc:
    st.error(f"Cannot connect to the document service: {exc}")
    st.stop()

document_by_id = {item["id"]: item for item in documents}
status_labels = {"pending": "Queued", "processing": "Indexing",
                 "completed": "Ready", "failed": "Failed"}

with st.sidebar:
    st.title("📄 Documents")
    st.caption("Upload, select, and manage your knowledge base.")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file", type=["pdf", "txt"],
        key=f"upload_{st.session_state.upload_key}")
    if st.button("Add document", disabled=uploaded_file is None,
                 use_container_width=True, type="primary"):
        try:
            with st.spinner("Uploading document…"):
                uploaded_file.seek(0)
                content_type = ("application/pdf" if uploaded_file.name.lower().endswith(".pdf")
                                else "text/plain")
                result = api_request("POST", "/upload", files={
                    "file": (uploaded_file.name, uploaded_file, content_type)})
            st.session_state.index_job = result["job_id"]
            st.session_state.upload_key += 1
            st.session_state.notice = ("success", "Upload complete. Indexing has started.")
            st.rerun()
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Upload failed: {exc}")

    if st.session_state.notice:
        kind, message = st.session_state.notice
        (st.success if kind == "success" else st.error)(message)
        st.session_state.notice = None

    st.divider()
    choices = [None, *document_by_id]
    selected = st.selectbox(
        "Chat with", choices,
        format_func=lambda source: "Select a document" if source is None else
        f"{document_by_id[source]['filename']} · {source[:8]} · "
        f"{status_labels.get(document_by_id[source]['status'], 'Unknown')}")
    selected_doc = document_by_id.get(selected)
    if selected_doc:
        st.caption(f"{selected_doc.get('chunk_count') or 0} passages · "
                   f"{selected_doc.get('page_count') or 0} pages")
        if selected_doc["status"] in ("pending", "failed", "completed"):
            label = ("Retry indexing" if selected_doc["status"] == "failed" else
                     "Reindex document" if selected_doc["status"] == "completed" else
                     "Start indexing")
            if st.button(label, use_container_width=True):
                try:
                    result = api_request("POST", f"/documents/{selected}/index")
                    st.session_state.index_job = result["job_id"]
                    st.rerun()
                except (requests.RequestException, RuntimeError) as exc:
                    st.error(str(exc))
        if selected_doc.get("error"):
            st.error(selected_doc["error"])

    with st.expander("Advanced settings"):
        top_k = st.slider("Maximum sources", 1, 10, 3)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.55, 0.05)
        st.caption("Lower the threshold if relevant passages are missing.")

    if selected and st.button("Clear conversation", use_container_width=True):
        st.session_state.conversations[selected] = []
        st.session_state.citation = None
        st.rerun()

st.title("Talk to Documents")
st.markdown('<p class="subtitle">Ask questions grounded in your documents. '
            'Select a source to inspect the original page.</p>', unsafe_allow_html=True)

if not selected_doc:
    st.info("Choose a document in the sidebar to start a conversation.")
elif selected_doc["status"] != "completed":
    st.info("Chat becomes available when indexing finishes.")

chat_column, viewer_column = st.columns([1.25, 1], gap="large")
with chat_column:
    messages = st.session_state.conversations.setdefault(selected, [])
    for message_index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                show_answer(message, message_index)
            else:
                st.markdown(message["content"])
    pending = st.session_state.pending_chat
    if pending and pending["source"] == selected:
        with st.chat_message("assistant"):
            st.status("Searching and composing an answer…", state="running")

    # The backend checks readiness again inside its job-queuing transaction.
    can_chat = bool(selected_doc and selected_doc["status"] == "completed"
                    and pending is None)
    question = st.chat_input("Ask a question about this document…", disabled=not can_chat)
    if question:
        history = [{"role": item["role"], "content": item["content"]}
                   for item in messages[-6:]]
        try:
            result = api_request("POST", "/chat", json={
                "question": question, "source": selected, "top_k": top_k,
                "threshold": threshold, "history": history})
            messages.append({"role": "user", "content": question})
            st.session_state.pending_chat = {"source": selected, "job_id": result["job_id"]}
            st.rerun()
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Could not submit your question: {exc}")

with viewer_column:
    if st.session_state.citation:
        show_citation(st.session_state.citation)
    else:
        st.subheader("Source viewer")
        st.caption("Select a citation below an answer to inspect its PDF page.")


@st.fragment(run_every="2s")
def monitor_jobs():
    """Poll persisted jobs and rerun when a result becomes available."""
    index_job = st.session_state.index_job
    if index_job:
        try:
            job = api_request("GET", f"/jobs/{index_job}")
            if job["status"] in ("completed", "failed"):
                st.session_state.index_job = None
                st.session_state.notice = (
                    ("success", "Document is ready for questions.")
                    if job["status"] == "completed" else
                    ("error", f"Indexing failed: {job['error']}"))
                st.rerun()
            st.progress(job["progress"], text=f"Indexing · {job['stage']}")
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Could not read indexing status: {exc}")

    pending = st.session_state.pending_chat
    if pending:
        try:
            job = api_request("GET", f"/jobs/{pending['job_id']}")
            if job["status"] in ("completed", "failed"):
                if job["status"] == "completed":
                    result = job["result"]
                    reply = {"role": "assistant", "content": result["answer"],
                             "sources": result.get("sources", [])}
                else:
                    reply = {"role": "assistant",
                             "content": f"I couldn't answer this question: {job['error']}"}
                st.session_state.conversations[pending["source"]].append(reply)
                st.session_state.pending_chat = None
                st.rerun()
            st.progress(job["progress"], text=f"Answering · {job['stage']}")
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Could not read answer status: {exc}")


monitor_jobs()
