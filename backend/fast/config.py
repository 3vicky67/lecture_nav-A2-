import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env in CWD and module directory
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Ensure ffmpeg on PATH (kept for compatibility)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin"

# Performance-related constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
try:
    import torch  # type: ignore
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
WHISPER_MODEL = "tiny"

# Logging
logger = logging.getLogger("video_rag_fast")
logger.setLevel(logging.INFO)
stream_h = logging.StreamHandler()
stream_h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(stream_h)

def log_json(event: str, payload: dict, trace_id: str = None, level: str = "info") -> None:
    entry = {"event": event, "trace_id": trace_id or "-", **payload}
    if level == "info":
        logger.info(json.dumps(entry, default=str))
    else:
        logger.error(json.dumps(entry, default=str))


