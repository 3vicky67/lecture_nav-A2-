import whisper

def transcribe_video_fast(video_path: str, model_name: str = "tiny"):
    print(f"🚀 Loading fast Whisper model '{model_name}'...")
    whisper_model = whisper.load_model(model_name)
    print("✅ Whisper model loaded, starting transcription...")
    result = whisper_model.transcribe(video_path, verbose=False)
    return result['segments']


