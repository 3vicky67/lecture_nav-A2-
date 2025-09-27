from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import uuid
import whisper
import cohere
import json

# ------------------ Setup ------------------ #
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin"

COHERE_API_KEY = "myg3oaQlv4v6vYEghw13ZVTHOKjXcqwsnMWlMdG1"
co = cohere.Client(COHERE_API_KEY)

VIDEO_DB = {}  # {video_id: {"segments": [...], "source": "file|url", "path": "..."}}

app = Flask(__name__)

# ------------------ Cohere Q&A ------------------ #
def ask_cohere(question: str) -> str:
    response = co.chat(
        model="command-xlarge-nightly",
        message=question
    )
    return response.text

# ------------------ Transcription ------------------ #
def transcribe_video(video_path: str, model_name="small"):
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, verbose=False)
    return result['segments']

# ------------------ Search ------------------ #
def search_query(segments, query, context_window=1):
    query_lower = query.lower()
    matches = []
    for i, seg in enumerate(segments):
        if query_lower in seg['text'].lower():
            start_idx = max(0, i - context_window)
            end_idx = min(len(segments), i + context_window + 1)
            snippet = " ".join([s['text'] for s in segments[start_idx:end_idx]])
            matches.append({
                "t_start": round(seg['start'], 2),
                "t_end": round(seg['end'], 2),
                "title": f"Segment {i}",
                "snippet": snippet,
                "score": 1.0
            })
    return matches

# ------------------ Video Downloader ------------------ #
def download_youtube_video(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%(id)s.%(ext)s")
    cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4", "-o", output_path, url]
    subprocess.run(cmd, check=True)
    for file in os.listdir(temp_dir):
        if file.endswith((".mp4", ".mkv", ".webm")):
            return os.path.join(temp_dir, file)
    raise FileNotFoundError(f"No video file found in {temp_dir}.")

# ------------------ Format Time ------------------ #
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        sec = seconds % 60
        return f"{minutes}.{int(sec):02d}m"

# ------------------ Default Home Route ------------------ #
@app.route("/", methods=["GET"])
def home():
    return "🎬 Video Transcript API is running! Use /api/ingest_video and /api/search_timestamps endpoints."

# ------------------ API Endpoints ------------------ #
@app.route("/api/ingest_video", methods=["POST"])
def ingest_video():
    data = request.json
    video_url = data.get("video_url")
    video_file = data.get("video_file")

    video_id = str(uuid.uuid4())
    try:
        if video_url:
            video_path = download_youtube_video(video_url)
            source_type = "url"
        elif video_file:
            if not os.path.exists(video_file):
                return jsonify({"error": f"File {video_file} not found"}), 400
            video_path = video_file
            source_type = "file"
        else:
            return jsonify({"error": "Must provide video_url or video_file"}), 400

        segments = transcribe_video(video_path, model_name="small")
        VIDEO_DB[video_id] = {
            "segments": segments,
            "source": source_type,
            "path": video_url if source_type=="url" else video_file
        }
        return jsonify({"video_id": video_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search_timestamps", methods=["POST"])
def search_timestamps():
    data = request.json
    video_id = data.get("video_id")
    query = data.get("query")
    k = int(data.get("k", 3))

    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404

    segments = VIDEO_DB[video_id]["segments"]
    video_url = VIDEO_DB[video_id]["path"]

    matches = search_query(segments, query)
    top_matches = matches[:k]

    formatted_results = []
    for r in top_matches:
        formatted_results.append({
            "video_id": video_id,
            "video_url": video_url,
            "start_time": format_time(r["t_start"]),
            "end_time": format_time(r["t_end"]),
            "title": r["title"],
            "snippet": r["snippet"],
            "score": r["score"]
        })

    return jsonify({"query": query, "results": formatted_results})

# ------------------ Run Flask ------------------ #
if __name__ == "__main__":
    app.run(debug=True, port=5000)
