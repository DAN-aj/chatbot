"""Talk to Documents: a complete session-scoped RAG app for Streamlit Cloud."""
import os
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer

from cloud_rag import answer_question, index_document, new_store, remove_document
from providers import ProviderOperationError, create_provider

# Cloud uses Streamlit secrets; a local .env is convenient for development.
load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="Talk to Documents", page_icon="📄", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1440px; padding-top: 2rem;}
    [data-testid="stSidebar"] {background: #f7f8fa;}
    [data-testid="stChatMessage"] {border: 1px solid #e7e9ed; border-radius: 16px;}
    .subtitle {color: #667085; font-size: 0.95rem; margin-top: -0.7rem;}
</style>
""", unsafe_allow_html=True)

# All documents, vectors, and chat messages belong to the current browser session.
for key, default in {
    "documents": {}, "conversations": {}, "citation": None,
    "upload_key": 0, "indexing": False, "answering": False, "notice": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if "vectors" not in st.session_state:
    st.session_state.vectors = new_store()


def configured_api_key(provider: str) -> str:
    """Read only the chosen provider's server-owned key."""
    name = "GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY"
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets[name])
    except (FileNotFoundError, KeyError):
        return ""


def friendly_error(exc: Exception, provider: str) -> str:
    """Turn service errors into actionable text without leaking request data."""
    root = exc.cause if isinstance(exc, ProviderOperationError) else exc
    detail = str(root).lower()
    stage = ("tạo câu trả lời" if getattr(exc, "stage", "") == "answer"
             else "tạo embedding")
    code = getattr(root, "code", getattr(root, "status_code", None))
    if provider == "OpenAI" and any(term in detail for term in (
            "credit_balance_exhausted", "insufficient_quota", "no credits remaining")):
        return ("OpenAI API đã hết tín dụng. Chọn Gemini trong thanh bên và "
                "nhập Gemini API key, hoặc nạp tín dụng OpenAI để tiếp tục.")
    if code == 429 or "429" in detail or "resource_exhausted" in detail:
        return (f"{provider} đang giới hạn lượt sử dụng khi {stage}. Hãy đợi rồi thử lại "
                "hoặc kiểm tra quota của API key.")
    if (code in (401, 403) or "401" in detail or "403" in detail
            or "api key not valid" in detail or "api_key_invalid" in detail):
        return f"Không thể xác thực {provider} API key. Hãy kiểm tra lại key."
    if "returned an empty answer" in detail:
        return (f"{provider} không trả về nội dung ở bước {stage}. "
                "Hãy thử câu hỏi ngắn hơn hoặc kiểm tra Manage app → Logs.")
    if code == 404 or "not found" in detail:
        return (f"{provider} không cho API key này truy cập mô hình "
                f"{getattr(exc, 'model', '')} ở bước {stage} (HTTP 404).")
    if isinstance(exc, ProviderOperationError):
        status = f"HTTP {code}" if code is not None else type(root).__name__
        return (f"{provider} gặp lỗi khi {stage} bằng {exc.model} ({status}). "
                "Mở Manage app → Logs để xem chi tiết; không gửi API key.")
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Không thể xử lý yêu cầu với {provider}. Kiểm tra API key và thử lại."


def show_answer(message: dict, message_index: int) -> None:
    """Render grounded text and compact numbered citation controls."""
    st.markdown(message["content"])
    sources = message.get("sources", [])
    if not sources:
        return
    st.caption("Sources · select a citation to inspect the original page")
    columns = st.columns(min(len(sources), 4))
    for index, source in enumerate(sources):
        number = source.get("citation_id", index + 1)
        page = source.get("page_number")
        label = f"[{number}] {source['filename']}"
        if page:
            label += f" · p. {page}"
        with columns[index % len(columns)]:
            if st.button(label, key=f"cite_{message_index}_{index}",
                         use_container_width=True):
                st.session_state.citation = source
                st.rerun()


def show_citation(source: dict) -> None:
    """Render the PDF beside the chat at its exact highlighted coordinates."""
    st.subheader("Source viewer")
    st.caption(source["filename"])
    locations = source.get("locations") or [source]
    page_locations = [item for item in locations if item.get("page_number")]
    if page_locations:
        location = st.selectbox(
            "Cited page", page_locations,
            format_func=lambda item: f"{item.get('filename', source['filename'])} · "
            f"page {item['page_number']} · "
            f"{', '.join(item.get('paragraph_ids', [])) or 'passage'}",
            key=f"location_{source['source']}_{source['chunk_id']}")
        page = int(location["page_number"])
        location_source = location.get("source", source["source"])
        box = location.get("bbox")
        annotations = []
        if box:
            annotations = [{
                "page": page, "x": box[0], "y": box[1],
                "width": box[2] - box[0], "height": box[3] - box[1],
                "color": "#F9B84A",
            }]
        document = st.session_state.documents.get(location_source)
        if document:
            pdf_viewer(
                document["data"], height=680, width="100%",
                key=f"pdf_{location_source}_{page}_{box}",
                annotations=annotations, pages_to_render=[page],
                scroll_to_annotation=1 if annotations else None,
                scroll_to_page=None if annotations else page)
        else:
            st.warning("The original PDF is no longer in this session.")
    else:
        st.info("This citation comes from a text document.")
    with st.expander("Read cited passage"):
        st.write(source.get("text", ""))


with st.sidebar:
    st.title("📄 Documents")
    st.caption("Files and vectors are held temporarily in this app session.")

    if "provider_choice" not in st.session_state:
        st.session_state.provider_choice = (
            "OpenAI" if st.session_state.documents or
            (st.session_state.get("visitor_openai_key") and
             not configured_api_key("Gemini")) else "Gemini")
    provider = st.radio(
        "AI provider", ["Gemini", "OpenAI"], horizontal=True,
        key="provider_choice", disabled=bool(st.session_state.documents),
        help="Remove all documents before switching provider; their vectors "
             "were created by the selected model.")
    server_key = configured_api_key(provider)
    if server_key:
        api_key = server_key
        st.caption("Language model access is configured for this app.")
    else:
        with st.expander(f"{provider} API key", expanded=True):
            api_key = st.text_input(
                "API key", type="password",
                key=f"visitor_{provider.lower()}_key",
                help="Used for multilingual embeddings and answers in this session.")
            st.caption("Your key is held in this session and is not saved to the repository.")
            if provider == "Gemini":
                st.caption("Create a key in [Google AI Studio](https://aistudio.google.com/api-keys). "
                           "Free-tier usage has rate limits and may be used to improve Google's products.")

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file", type=["pdf", "txt"],
        key=f"upload_{st.session_state.upload_key}")
    st.caption("Up to 20 MiB. PDF text must be selectable; scans need OCR.")
    st.caption(f"Uploaded text is sent to {provider} for embeddings and answers.")
    can_upload = (uploaded_file is not None and bool(api_key)
                  and not st.session_state.indexing
                  and len(st.session_state.documents) < 5)
    if len(st.session_state.documents) >= 5:
        st.info("This session holds up to five documents. Remove one to add another.")
    if st.button("Add document", type="primary", use_container_width=True,
                 disabled=not can_upload):
        status = st.status("Indexing document…", expanded=True)
        bar = st.progress(0)
        st.session_state.indexing = True
        try:
            data = uploaded_file.getvalue()
            client = create_provider(provider, api_key)

            def progress(percent: int, stage: str) -> None:
                bar.progress(percent, text=stage)

            document = index_document(
                st.session_state.vectors, client, data, uploaded_file.name, progress)
            document["data"] = data
            st.session_state.documents[document["id"]] = document
            st.session_state.upload_key += 1
            st.session_state.notice = ("success", f"{document['filename']} is ready.")
            status.update(label="Document ready", state="complete")
            st.rerun()
        except Exception as exc:
            logging.exception("Document indexing failed")
            status.update(label="Indexing failed", state="error")
            st.error(friendly_error(exc, provider))
        finally:
            st.session_state.indexing = False

    if st.session_state.notice:
        kind, message = st.session_state.notice
        (st.success if kind == "success" else st.error)(message)
        st.session_state.notice = None

    st.divider()
    documents = st.session_state.documents
    choices = [None, *documents]
    selected = st.selectbox(
        "Chat with", choices,
        format_func=lambda source: "Select a document" if source is None else
        f"{documents[source]['filename']} · {source[:8]} · Ready")
    if selected:
        current = documents[selected]
        st.caption(f"{current['chunk_count']} passages · "
                   f"{current['page_count'] or 0} pages")

    with st.expander("Advanced settings"):
        top_k = st.slider("Maximum sources", 1, 10, 3)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.55, 0.05)
        st.caption("Lower the threshold if relevant passages are missing.")
    if selected and st.button("Clear conversation", use_container_width=True):
        st.session_state.conversations[selected] = []
        st.session_state.citation = None
        st.rerun()
    if selected and st.button("Remove document", use_container_width=True):
        remove_document(st.session_state.vectors, selected)
        del st.session_state.documents[selected]
        st.session_state.conversations.pop(selected, None)
        st.session_state.citation = None
        st.rerun()

st.title("Talk to Documents")
st.markdown(
    '<p class="subtitle">Ask questions grounded in your documents. '
    'Select a source to inspect its original page.</p>',
    unsafe_allow_html=True)
if not api_key:
    st.info(f"Add a {provider} API key in the sidebar to begin.")
elif not selected:
    st.info("Upload and select a document to start a conversation.")

# The viewer stays beside the conversation for direct source inspection.
chat_column, viewer_column = st.columns([1.25, 1], gap="large")
with chat_column:
    messages = st.session_state.conversations.setdefault(selected, [])
    for message_index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                show_answer(message, message_index)
            else:
                st.markdown(message["content"])

    ready = bool(selected and api_key and not st.session_state.indexing
                 and not st.session_state.answering)
    question = st.chat_input(
        "Ask a question about this document…", disabled=not ready)
    if question:
        messages.append({"role": "user", "content": question})
        st.session_state.answering = True
        with st.chat_message("assistant"):
            with st.status("Searching and composing an answer…", expanded=True):
                bar = st.progress(0)

                def progress(percent: int, stage: str) -> None:
                    bar.progress(percent, text=stage)

                try:
                    client = create_provider(provider, api_key)
                    history = [
                        {"role": item["role"], "content": item["content"]}
                        for item in messages[:-1] if not item.get("error")][-6:]
                    result = answer_question(
                        st.session_state.vectors, client, selected, question,
                        top_k, threshold, history, progress)
                    messages.append({
                        "role": "assistant", "content": result["answer"],
                        "sources": result.get("sources", [])})
                except Exception as exc:
                    logging.exception("Document question failed")
                    messages.append({
                        "role": "assistant",
                        "content": friendly_error(exc, provider), "error": True})
                finally:
                    st.session_state.answering = False
        st.rerun()

with viewer_column:
    citation = st.session_state.citation
    if citation and citation.get("source") == selected:
        show_citation(citation)
    else:
        st.subheader("Source viewer")
        st.caption("Select a numbered citation below an answer to inspect its PDF page.")
