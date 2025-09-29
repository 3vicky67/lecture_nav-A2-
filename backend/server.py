from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import uuid
import time
import logging
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Hugging Face / Cohere / Torch / FAISS imports
import whisper
import cohere
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
from typing import List, Dict

# ------------------ Setup ------------------ #
# Load environment variables from .env file
load_dotenv()

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin"

# ------------------ Secrets from environment variables ------------------ #
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is required")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

co = cohere.Client(COHERE_API_KEY)

EMBEDDING_MODEL = "google/gemma-2-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Logging ------------------ #
logger = logging.getLogger("video_rag")
logger.setLevel(logging.INFO)
stream_h = logging.StreamHandler()
stream_h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(stream_h)

def log_json(event: str, payload: dict, trace_id: str = None, level="info"):
    entry = {"event": event, "trace_id": trace_id or "-", **payload}
    if level == "info":
        logger.info(json.dumps(entry, default=str))
    else:
        logger.error(json.dumps(entry, default=str))

# ------------------ In-memory DB ------------------ #
VIDEO_DB: Dict[str, dict] = {}

# ------------------ Embedding model ------------------ #
tokenizer = None
model = None

def load_embedding_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, token=HF_TOKEN, use_fast=True)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL, token=HF_TOKEN).to(DEVICE)
        model.eval()
        print(f"Loaded embedding model '{EMBEDDING_MODEL}'")
    except Exception as e:
        print(f"Failed to load '{EMBEDDING_MODEL}', falling back to 'sentence-transformers/all-MiniLM-L6-v2'")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
        model.eval()

@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = 8) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts
        arr = mean_pooled.cpu().numpy().astype("float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        all_vecs.append(arr)
    return np.vstack(all_vecs)

@torch.no_grad()
def embed_query(text: str) -> np.ndarray:
    return embed_texts([text], batch_size=1)[0]

# ------------------ Cohere Q&A ------------------ #
def ask_cohere(question: str) -> str:
    response = co.chat(model="command-xlarge-nightly", message=question)
    return response.text

# ------------------ Flask app ------------------ #
app = Flask(__name__)
CORS(app)

# Load embedding model at startup
load_embedding_model()

# ------------------ Default Home Route ------------------ #
@app.route("/", methods=["GET"])
def home():
    return "🎬 Video RAG API is running! Use /api/ingest_video and /api/search_timestamps endpoints."

# ------------------ (rest of your API endpoints remain unchanged) ------------------ #
