from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import subprocess
import uuid
import whisper
import cohere
import json
import time
import logging
from typing import List, Dict
from werkzeug.utils import secure_filename

# Embedding/torch
import torch
from transformers import AutoTokenizer, AutoModel

# FAISS
import numpy as np
import faiss

# ------------------ Setup ------------------ #
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin"

COHERE_API_KEY = "myg3oaQlv4v6vYEghw13ZVTHOKjXcqwsnMWlMdG1"
co = cohere.Client(COHERE_API_KEY)

# ------------------ Hugging Face ------------------ #
HF_TOKEN = "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"  # Replace with your HF token
EMBEDDING_MODEL = "google/gemma-2-2b"  # gated model
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

# Global variables for embedding model
tokenizer = None
model = None

# ------------------ Embedding utils ------------------ #
def load_embedding_model():
    """
    Load tokenizer + model with Hugging Face token.
    Fallback to open model if gated access fails.
    """
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, use_auth_token=HF_TOKEN, use_fast=True)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL, use_auth_token=HF_TOKEN).to(DEVICE)
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

# ------------------ Transcription ------------------ #
def transcribe_video(video_path: str, model_name="small"):
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, verbose=False)
    return result['segments']

# ------------------ Video Downloader ------------------ #
def download_video(url: str) -> str:
    """
    Download video from YouTube, Vimeo, or other supported platforms using yt-dlp
    """
    import yt_dlp
    
    temp_dir = tempfile.mkdtemp()
    
    # Configure yt-dlp options for multiple platforms
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        'quiet': True,  # Suppress output
        'extract_flat': False,  # Ensure we download the actual video
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Download the video
            ydl.download([url])
            
            # Find the downloaded file
            for file in os.listdir(temp_dir):
                if file.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
                    return os.path.join(temp_dir, file)
            
            raise FileNotFoundError(f"No video file found in {temp_dir}")
        except Exception as e:
            raise Exception(f"Failed to download video from {url}: {str(e)}")

# Keep the old function name for backward compatibility
def download_youtube_video(url: str) -> str:
    return download_video(url)

# ------------------ Format Time ------------------ #
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}.{int(sec):02d}m"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load embedding model on startup
load_embedding_model()

# ------------------ Default Home Route ------------------ #
@app.route("/", methods=["GET"])
def home():
    return "🎬 Video RAG API is running! Use /api/ingest_video and /api/search_timestamps endpoints."

# ------------------ API Endpoints ------------------ #
@app.route("/api/ingest_video", methods=["POST"])
def ingest_video():
    trace_id = uuid.uuid4().hex[:8]
    start_ts = time.time()
    video_id = str(uuid.uuid4())
    
    try:
        # Check if it's a file upload
        if 'video_file' in request.files:
            file = request.files['video_file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Save uploaded file to temp directory
            temp_dir = tempfile.mkdtemp()
            filename = secure_filename(file.filename)
            video_path = os.path.join(temp_dir, filename)
            file.save(video_path)
            source_type = "file"
            video_source = filename
            
            log_json("ingest.started", {"video_file": filename, "video_id": video_id}, trace_id)
            
        # Check if it's JSON data with URL or file path
        elif request.is_json:
            data = request.json
            video_url = data.get("video_url")
            video_file = data.get("video_file")
            
            if video_url:
                log_json("ingest.started", {"video_url": video_url, "video_id": video_id}, trace_id)
                video_path = download_youtube_video(video_url)
                source_type = "url"
                video_source = video_url
            elif video_file:
                log_json("ingest.started", {"video_file": video_file, "video_id": video_id}, trace_id)
                if not os.path.exists(video_file):
                    return jsonify({"error": f"File {video_file} not found"}), 400
                video_path = video_file
                source_type = "file"
                video_source = video_file
            else:
                return jsonify({"error": "Must provide video_url or video_file"}), 400
        else:
            return jsonify({"error": "Must provide video file or JSON data"}), 400

        # Transcribe video
        segments = transcribe_video(video_path, model_name="small")
        
        # Prepare texts and metadata for embedding
        texts, metadatas = [], []
        for i, seg in enumerate(segments):
            t = seg.get("text", "").strip() or "<empty_segment>"
            texts.append(t)
            metadatas.append({"start": round(seg["start"], 2), "end": round(seg["end"], 2), "idx": i})

        # Create embeddings and FAISS index
        embeddings = embed_texts(texts, batch_size=8)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Store in database
        VIDEO_DB[video_id] = {
            "segments": segments,
            "source": source_type,
            "path": video_source,
            "faiss_index": index,
            "embeddings": embeddings,
            "metadatas": metadatas
        }

        elapsed = time.time() - start_ts
        log_json("ingest.completed", {"video_id": video_id, "segments": len(segments), "elapsed_s": elapsed}, trace_id)
        return jsonify({"video_id": video_id})

    except Exception as e:
        log_json("ingest.exception", {"error": str(e)}, trace_id, level="error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/search_timestamps", methods=["POST"])
def search_timestamps():
    data = request.json
    video_id = data.get("video_id")
    query = data.get("query")
    k = int(data.get("k", 3))
    
    trace_id = uuid.uuid4().hex[:8]
    
    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404

    try:
        db_entry = VIDEO_DB[video_id]
        index = db_entry["faiss_index"]
        metadatas = db_entry["metadatas"]
        segments = db_entry["segments"]
        video_url = db_entry["path"]

        # Perform semantic search using embeddings
        q_vec = embed_query(query).astype("float32").reshape(1, -1)
        D, I = index.search(q_vec, k)
        D = D[0]; I = I[0]

        formatted_results = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0 or idx >= len(metadatas):
                continue
            md = metadatas[idx]
            seg = segments[md["idx"]]
            formatted_results.append({
                "video_id": video_id,
                "video_file": os.path.basename(video_url),  # Changed from video_url to video_file
                "start_time": format_time(md.get("start", 0)),
                "end_time": format_time(md.get("end", 0)),
                "snippets": seg.get("text", ""),  # Changed from snippet to snippets
                "score": float(score)
            })

        # Generate AI answer using retrieved context
        retrieved_texts = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0 or idx >= len(metadatas):
                continue
            md = metadatas[idx]
            seg = segments[md["idx"]]
            retrieved_texts.append(seg.get("text", ""))
        
        # Create RAG prompt with retrieved context for AI answer
        context_block = "\n\n---\n\n".join([f"[{i}] {t}" for i, t in enumerate(retrieved_texts, start=1)])
        rag_prompt = f"Use the following passages to answer the question in one concise sentence:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            ai_answer = ask_cohere(rag_prompt)
        except Exception as e:
            print(f"AI answer generation failed: {e}")
            ai_answer = None

        # Save to output.json (as in original new.py)
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, indent=2)

        log_json("search.completed", {"video_id": video_id, "query": query, "results_count": len(formatted_results)}, trace_id)
        return jsonify({"query": query, "results": formatted_results, "answer": ai_answer})

    except Exception as e:
        log_json("search.exception", {"error": str(e)}, trace_id, level="error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    data = request.json
    video_id = data.get("video_id")
    question = data.get("question")
    k = int(data.get("k", 3))
    
    trace_id = uuid.uuid4().hex[:8]
    
    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404

    try:
        db_entry = VIDEO_DB[video_id]
        index = db_entry["faiss_index"]
        metadatas = db_entry["metadatas"]
        segments = db_entry["segments"]

        # Perform semantic search for relevant context
        q_vec = embed_query(question).astype("float32").reshape(1, -1)
        D, I = index.search(q_vec, k)
        D = D[0]; I = I[0]

        retrieved_texts, retrieval_meta = [], []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0 or idx >= len(metadatas):
                continue
            md = metadatas[idx]
            seg = segments[md["idx"]]
            retrieved_texts.append(seg.get("text", ""))
            retrieval_meta.append({"score": float(score), **md})

        # Create RAG prompt with retrieved context
        context_block = "\n\n---\n\n".join([f"[{i}] {t}" for i, t in enumerate(retrieved_texts, start=1)])
        rag_prompt = f"Use the following passages to answer the question:\n{context_block}\n\nQuestion: {question}\n\nAnswer concisely."
        answer = ask_cohere(rag_prompt)

        log_json("ai.completed", {"video_id": video_id, "question": question, "retrieved_count": len(retrieved_texts)}, trace_id)
        return jsonify({"answer": answer, "retrieved": retrieval_meta})

    except Exception as e:
        log_json("ai.exception", {"error": str(e)}, trace_id, level="error")
        return jsonify({"error": str(e)}), 500

# ------------------ Run Flask ------------------ #
if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')