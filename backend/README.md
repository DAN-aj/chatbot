# Talk to Documents

Local PDF/TXT question answering with FastAPI, Streamlit, Qdrant, layout-aware
PDF parsing, multilingual embeddings, and configurable generation models.
This backend is independent of the [Streamlit Cloud app](../README.md) at the
repository root. Run the commands below from the `backend/` directory.

## Start

Use Python 3.11 or newer. Qdrant must be running. The first indexing or answer
request downloads the model weights from Hugging Face.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-backend.txt
cp .env.example .env
docker run -d --name qdrant-genai -p 127.0.0.1:6333:6333 -v qdrant_genai_data:/qdrant/storage qdrant/qdrant
```

If the container already exists, use `docker start qdrant-genai`. Set
`RAG_API_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `RAG_DATA_DIR`, and
`MAX_UPLOAD_BYTES` in `.env` for your environment. The local Streamlit client
reads its API URL from `.env`.

Set `GEMINI_API_KEY` to analyze charts, figures, and scanned pages with Gemini
Vision. Without it, text and tables still index; the job result records a
`vision_unconfigured` warning for visual regions. Select `EMBEDDING_PROVIDER`
as `local`, `openai`, or `fastembed` (`pip install fastembed` for the last
option). OpenAI embeddings are sent in batches of 64 by default. Select
`LLM_PROVIDER` as `local`, `gemini`, or `openai`; optionally set a different
`LLM_FALLBACK_PROVIDER`. External calls retry transient 429/5xx failures with
exponential backoff and jitter. The LLM circuit opens after repeated transient
failures and uses the fallback for the configured cooldown period.

Run these commands in separate terminals:

```bash
source .venv/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
source .venv/bin/activate
python -m streamlit run client.py
```

Open http://127.0.0.1:8501 for the chatbot and http://127.0.0.1:8000/docs for
the API. Run one FastAPI process: it owns separate durable indexing and query
workers. Multiple Uvicorn processes would compete for the same local models.

## Deployment scope

The [Streamlit Cloud app](../README.md) runs directly from the repository root
and does not call this FastAPI server. It accepts a Gemini or OpenAI API key in
the sidebar, or the app owner can set a key in Streamlit Cloud Secrets. Its
documents and vectors are scoped to one app session. This backend stores files,
jobs, and vectors on the server where it runs. To expose it outside localhost,
configure authentication and durable storage for that deployment.

## Workflow

1. Upload a PDF or UTF-8 TXT file in the sidebar. Files are limited to 20 MiB
   by default. Visual regions need `GEMINI_API_KEY` for analysis.
2. The API immediately returns a job ID. Streamlit shows indexing progress and
   disables chat until that document is marked `completed`.
3. Ask a question. One action performs retrieval and generation. Broad summary
   requests use hierarchical map-reduce over every indexed chunk.
4. Select a numbered citation to open the original PDF page and highlight the
   measured text location. TXT citations show the cited passage.

The chat history is held in the current Streamlit session, separately for each
document. In Advanced settings, Top-K defaults to 3 and the cosine threshold
defaults to 0.55. Similarity is a vector score, not answer confidence.

## Processing and storage

- `upload.py` streams uploads to `uploads/` and checks type, size, PDF header,
  and UTF-8 validity.
- `rag/extractor.py` combines pdfplumber tables/word coordinates with PyMuPDF
  image and vector-region detection. Tables become Markdown; Vision turns
  figures into quantitative summaries and data tables. It logs structured
  per-page warnings and continues after recoverable page failures.
- `rag/transform.py` keeps each table, code block, and figure as one parent.
  Sentence-based child chunks and per-row table vectors point to that parent.
- `rag/embedding.py` supplies local MiniLM, optional FastEmbed, and batched
  OpenAI adapters. External calls use bounded retry with jitter.
- `rag/repository.py` stores stable unique point IDs in the cosine Qdrant
  collection. Each child vector retains its full parent context and spatial
  location; retrieval deduplicates parents and sends those to the generator.
- `rag/state.py` keeps documents and jobs in `var/jobs.sqlite3` (or the
  configured data directory). Jobs have queued, processing, completed, and
  failed states, progress, stage, result, and error. Interrupted jobs are
  requeued on restart.
- `rag/pipeline.py` retrieves unique passages once per normal question.
  Summary questions traverse all content and reduce section summaries.
- `rag/llm.py` bounds prompts by complete parent blocks and routes between
  local Qwen, Gemini, and OpenAI with an optional circuit-breaker fallback.

The collection name is derived from `QDRANT_COLLECTION` with a
`_parent_v1_{provider}` suffix. Earlier collections are retained. Stored
documents from an older index format are queued for reindexing on startup.
The app has no authentication; keep it on localhost or add authentication before
exposing it to a network. PDF positioning is based on selectable text; complex
layouts may produce a broad highlight.

## API

- `POST /upload`: save file, queue indexing, return HTTP 202 and job ID.
- `GET /documents`, `GET /documents/{source}`: document status and counts.
- `GET /documents/{source}/file`: original file for spatial citations.
- `GET /documents/{source}/figures/{figure_name}`: extracted figure image.
- `POST /documents/{source}/index`: queue reindexing.
- `POST /chat`: queue retrieval and answer generation.
- `POST /search`: queue retrieval only.
- `GET /jobs`, `GET /jobs/{job_id}`: durable progress and results.

An index job for a selected document prevents its chat job from being queued;
the API returns HTTP 409. The UI also disables its input until indexing
completes.

API errors use `{code, message, suggested_action}`. A completed index job can
include `warnings` for unreadable pages or figures; inspect these when source
coverage matters. Missing Vision credentials, provider quota, and model
availability are deployment concerns and do not invalidate healthy pages.
