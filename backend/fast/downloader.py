import os
import tempfile

def download_video(url: str) -> str:
    import yt_dlp
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        for file in os.listdir(temp_dir):
            if file.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
                return os.path.join(temp_dir, file)
    raise FileNotFoundError(f"No video file found in {temp_dir}")

def download_youtube_video(url: str) -> str:
    return download_video(url)


