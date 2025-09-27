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

VIDEO_DB = {}  # {video_id: {"segments": [...], "source": "file|url", "path": "..." }}

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

    # Run yt-dlp and capture output for debugging
    subprocess.run(cmd, check=True)

    # Find downloaded video file in temp_dir
    for file in os.listdir(temp_dir):
        if file.endswith((".mp4", ".mkv", ".webm")):  # accept multiple formats
            return os.path.join(temp_dir, file)

    raise FileNotFoundError(f"No video file found in {temp_dir}. Check yt-dlp output.")

# ------------------ API: /ingest_video ------------------ #
def api_ingest_video(video_url=None, video_file=None):
    video_id = str(uuid.uuid4())
    try:
        if video_url:
            print(f"Downloading YouTube video: {video_url}")
            video_path = download_youtube_video(video_url)
            source_type = "url"
        elif video_file:
            if not os.path.exists(video_file):
                return {"error": f"File {video_file} not found"}
            video_path = video_file
            source_type = "file"
        else:
            return {"error": "Must provide video_url or file"}

        segments = transcribe_video(video_path, model_name="small")
        VIDEO_DB[video_id] = {
            "segments": segments,
            "source": source_type,
            "path": video_url if source_type=="url" else video_file
        }
        return {"video_id": video_id}
    except Exception as e:
        return {"error": str(e)}

# ------------------ API: /search_timestamps ------------------ #
def format_time(seconds):
    """Format seconds into min.sec string (e.g., 65.04 -> '1.05m')."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        sec = seconds % 60
        return f"{minutes}.{int(sec):02d}m"

def api_search_timestamps(query, video_id, k=3):
    if video_id not in VIDEO_DB:
        return {"error": f"Video ID {video_id} not found"}

    segments = VIDEO_DB[video_id]["segments"]
    video_url = VIDEO_DB[video_id]["path"]  # store url or file path

    video_filename = os.path.basename(video_url) 

    matches = search_query(segments, query)
    top_matches = matches[:k]

    # Prepare custom JSON output
    formatted_results = []
    for r in top_matches:
        formatted_results.append({
            "video_id": video_id,
            "video_url":video_url,
            "start_time": format_time(r["t_start"]),
            "end_time": format_time(r["t_end"]),
            "title": r["title"],
            "snippet": r["snippet"],
            "score": r["score"]
        })

    # Save to output.json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=2)

    return {
        "query": query,
        "results": formatted_results
    }

# ------------------ Separate AI Q&A ------------------ #
def api_ask_ai(video_id, question):
    if video_id not in VIDEO_DB:
        return {"error": f"Video ID {video_id} not found"}

    segments = VIDEO_DB[video_id]["segments"]
    transcript_text = " ".join([seg['text'] for seg in segments])
    combined_question = f"Video transcript: {transcript_text}\n\nQuestion: {question}"
    ai_answer = ask_cohere(combined_question)

    return {"answer": ai_answer}

# ------------------ Console Simulation ------------------ #
if __name__ == "__main__":
    print("=== Simulated API Console ===")
    while True:
        print("\nOptions:\n1. Ingest video\n2. Search query\n3. Ask AI question\n4. Exit")
        choice = input("Choose an option: ").strip()

        if choice == '1':
            method = input("Enter '1' for local file or '2' for YouTube link: ").strip()
            if method == '1':
                path = input("Enter local video file path: ").strip()
                response = api_ingest_video(video_file=path)
            elif method == '2':
                url = input("Enter YouTube URL: ").strip()
                response = api_ingest_video(video_url=url)
            else:
                print("Invalid choice.")
                continue
            print(json.dumps(response, indent=2))

        elif choice == '2':
            vid = input("Enter video_id: ").strip()
            query = input("Enter search query: ").strip()
            k = input("Enter top_k (default 3): ").strip()
            k = int(k) if k.isdigit() else 3
            response = api_search_timestamps(query=query, video_id=vid, k=k)
            print(json.dumps(response, indent=2))

        elif choice == '3':
            vid = input("Enter video_id: ").strip()
            question = input("Enter your AI question: ").strip()
            response = api_ask_ai(video_id=vid, question=question)
            print(json.dumps(response, indent=2))

        elif choice == '4':
            break
        else:
            print("Invalid choice. Try again.")
