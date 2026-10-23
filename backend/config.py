"""Application settings loaded once from the project .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

_data_path = Path(os.getenv("RAG_DATA_DIR", "var")).expanduser()
DATA_DIR = (_data_path if _data_path.is_absolute() else ROOT / _data_path).resolve()
DATABASE_PATH = DATA_DIR / "jobs.sqlite3"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
_collection_base = os.getenv("QDRANT_COLLECTION", "documents_multilingual_minilm_v2_spatial")
COLLECTION = f"{_collection_base}_parent_v1_{EMBEDDING_PROVIDER}"
EMBEDDING_BATCH_SIZE = max(1, min(100, int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))))
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "gemini").lower()
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.5-flash-lite")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").lower()
LLM_CIRCUIT_FAILURES = max(1, int(os.getenv("LLM_CIRCUIT_FAILURES", "3")))
LLM_CIRCUIT_COOLDOWN_SECONDS = max(1, int(os.getenv("LLM_CIRCUIT_COOLDOWN_SECONDS", "60")))
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
