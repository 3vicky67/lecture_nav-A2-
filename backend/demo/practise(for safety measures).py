import os
import json
import whisper
import cohere

# ------------------ FFmpeg Setup ------------------ #
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin"

# ------------------ Cohere AI Agent ------------------ #
COHERE_API_KEY = "myg3oaQlv4v6vYEghw13ZVTHOKjXcqwsnMWlMdG1"  # replace with your key
co = cohere.Client(COHERE_API_KEY)

def ask_cohere(question):
    response = co.chat(
        model="command-xlarge-nightly",
        message=question
    )
    return response.text

# ------------------ Transcript + Retrieval ------------------ #
def transcribe_video(video_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, verbose=False)
    return result['segments']

def search_query(segments, query, context_window=1):
    query_lower = query.lower()
    matches = []

    for i, seg in enumerate(segments):
        if query_lower in seg['text'].lower():
            start_idx = max(0, i - context_window)
            end_idx = min(len(segments), i + context_window + 1)
            snippet = " ".join([s['text'] for s in segments[start_idx:end_idx]])

            # Jump-to timestamp in mm:ss
            minutes = int(seg['start'] // 60)
            seconds = int(seg['start'] % 60)
            jump_to = f"{minutes:02d}:{seconds:02d}"

            matches.append({
                "t_start": round(seg['start'], 2),
                "t_end": round(seg['end'], 2),
                "snippet": snippet,
                "jump_to": jump_to
            })
    return matches

# ------------------ Generate Video Snippet Summary ------------------ #
def summarize_video_snippets(matches):
    if not matches:
        return "No relevant video snippets found for your query."
    # Combine top 2-3 snippets for short summary
    combined_snippets = " ".join([m['snippet'] for m in matches[:3]])
    # Trim for initial RAG answer
    summary = combined_snippets[:300] + ("..." if len(combined_snippets) > 300 else "")
    return f"Video snippet summary: {summary}"

# ------------------ Interactive RAG + AI ------------------ #
def interactive_video_agent(video_file):
    print("Transcribing video (this may take a few minutes)...")
    segments = transcribe_video(video_file, model_name="small")
    print("Video transcription complete.")

    # --- Runtime user query ---
    user_query = input("Enter your query about the video: ").strip()
    results = search_query(segments, user_query)

    # --- Initial RAG answer ---
    video_summary = summarize_video_snippets(results)
    rag_answer = f"{video_summary} Refer to the video from the given timestamps for details."

    output = {
        "query": user_query,
        "matches": results if results else [],
        "answer": rag_answer,
        "followup": "You can ask additional questions about the video or general knowledge, and the AI agent will answer.",
        "followup_questions": [],
        "final_prompt": "I hope you have clarified all questions. If you have more, AI can answer; else have a good day!"
    }

    print("\nInitial output:")
    print(json.dumps(output, indent=2))

    # --- Follow-up Q&A loop ---
    while True:
        user_input = input("\nDo you have any further questions? (yes/no): ").strip().lower()
        if user_input == "no":
            output["user_input"] = "no"
            break
        elif user_input == "yes":
            followup_question = input("Enter your follow-up question: ").strip()
            
            # Combine video context + user question
            context_snippets = " ".join([m['snippet'] for m in results[:3]]) if results else ""
            combined_question = f"Video context: {context_snippets}\n\nQuestion: {followup_question}"
            
            # Ask Cohere AI
            followup_answer = ask_cohere(combined_question)
            output["followup_questions"].append({
                "question": followup_question,
                "answer": followup_answer
            })
            print(f"\nAI Answer: {followup_answer}")
        else:
            print("Please answer with 'yes' or 'no'.")

    # --- Final JSON output ---
    print("\nFinal JSON output:")
    print(json.dumps(output, indent=2))

# ------------------ Run ------------------ #
if __name__ == "__main__":
    video_file = input("Enter path to your video file: ").strip()
    interactive_video_agent(video_file)
