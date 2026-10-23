# Talk to Documents

A PDF/TXT RAG chatbot designed to run inside one Streamlit Community Cloud app.
It replaces the former GPT-3.5 chatbot template.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Add `OPENAI_API_KEY` to an uncommitted `.env` file, or enter your own key in the
app sidebar. On Streamlit Community Cloud, set `OPENAI_API_KEY` in the app's
Settings → Secrets to let visitors use the chatbot without supplying a key.
Do not commit an API key to GitHub.

## How it works

1. Upload a selectable-text PDF or UTF-8 TXT file, up to 20 MiB.
2. `pdfplumber` extracts text with page numbers, paragraph IDs and measured PDF
   bounding boxes. The app packs passages by page and merges duplicate text.
3. Gemini's multilingual `gemini-embedding-2` or OpenAI's multilingual
   `text-embedding-3-small` embeds unique passages.
   Qdrant local mode stores vectors inside the current Streamlit session.
4. A question performs one cosine search with a default score threshold of
   0.55, then `gemini-3.5-flash-lite` or `gpt-6-luna` answers from the retrieved passages. Broad summary
   requests read the full document, using section reduction for long files.
5. Numbered citations open the original PDF page beside the chat and highlight
   the measured source location.

Uploaded text is sent to the selected AI provider for embeddings and answers. Each browser
session has its own documents, vectors, and chat history. Streamlit Community
Cloud does not guarantee persistence for generated files, so documents are
lost when the session or app restarts. Upload them again to continue. If
persistent documents across sessions are required, add external storage and
authentication before using the app for private data.

The app accepts either `GEMINI_API_KEY` or `OPENAI_API_KEY` in Streamlit Secrets,
the local `.env`, or a visitor's sidebar input. Gemini is the default when no
provider is configured and its free tier has rate limits; OpenAI requires
available API credit. Documents must be removed before changing providers,
because their vectors come from a different embedding model. If a document is
a scan without a text layer, run OCR before uploading it.
