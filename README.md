# Video RAG (Retrieval-Augmented Generation) Application

A sophisticated web application that uses advanced AI embeddings and semantic search to find relevant content in videos with timestamp-based results.

## Features

- 🎥 Video upload and transcription using OpenAI Whisper
- 🧠 **Semantic Search** using Hugging Face embeddings and FAISS vector database
- 🔍 **RAG-powered search** that understands context, not just keywords
- 📊 JSON-formatted search results with relevance scores
- 🎯 Jump to specific timestamps in videos
- 🤖 **AI-powered Q&A** using Cohere with video context retrieval
- 📈 Advanced logging and performance monitoring

## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   python run_server.py
   ```
   
   Or directly:
   ```bash
   python server_fast.py
   ```

   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## How to Use

1. **Start both servers** (backend on port 5000, frontend on port 5173)

2. **Upload a video:**
   - Click "Choose Video File" and select a video file
   - Click "Transcribe Video" to process the video

3. **Search the video:**
   - Enter your search query in the search bar
   - Click "Search" to find relevant timestamps
   - Results will be displayed in JSON format with:
     - `video_id`: Unique identifier for the video
     - `video_file`: Name of the video file
     - `start_time`: Start timestamp (formatted)
     - `end_time`: End timestamp (formatted)
     - `snippets`: Relevant text content
     - `score`: Relevance score

4. **Jump to timestamps:**
   - Click "Jump to Timestamp" on any result to navigate to that point in the video

## API Endpoints

### POST /api/ingest_video
Upload and transcribe a video file.

**Request:**
- Form data with `video_file` field
- Or JSON with `video_url` or `video_file` path

**Response:**
```json
{
  "video_id": "unique-video-id"
}
```

### POST /api/search_timestamps
Search through transcribed video content.

**Request:**
```json
{
  "video_id": "unique-video-id",
  "query": "search term",
  "k": 3
}
```

**Response:**
```json
{
  "query": "search term",
  "results": [
    {
      "video_id": "unique-video-id",
      "video_file": "filename.mp4",
      "start_time": "1.05m",
      "end_time": "2.30m",
      "snippets": "relevant text content...",
      "score": 1.0
    }
  ]
}
```

## Technical Details

- **Backend:** Flask with OpenAI Whisper for transcription
- **Frontend:** React with TypeScript and Vite
- **Search:** **Semantic search** using Hugging Face embeddings and FAISS vector database
- **AI Integration:** Cohere with RAG (Retrieval-Augmented Generation) for context-aware responses
- **Embeddings:** Google Gemma-2-2b model (with fallback to sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database:** FAISS for efficient similarity search

## How RAG Works

1. **Video Ingestion:** Video is transcribed using Whisper
2. **Embedding Creation:** Each transcript segment is converted to high-dimensional vectors using Hugging Face models
3. **Index Building:** FAISS creates an efficient search index of all embeddings
4. **Semantic Search:** User queries are embedded and matched against the video content using cosine similarity
5. **Context Retrieval:** Most relevant segments are retrieved and used as context for AI responses

## Troubleshooting

- Ensure FFmpeg is installed and in your PATH
- Make sure both servers are running on their respective ports
- Check browser console for any CORS or connection errors
- Verify that video files are in supported formats (MP4, MKV, WebM)
- **GPU Support:** The system will use CUDA if available, otherwise falls back to CPU
- **Model Loading:** First run may take time to download Hugging Face models

## Dependencies

### Backend
- Flask & Flask-CORS
- OpenAI Whisper
- Cohere
- yt-dlp
- **PyTorch** (for embeddings)
- **Transformers** (Hugging Face models)
- **FAISS** (vector database)
- **NumPy** (numerical operations)
- **Hugging Face Hub** (model access)

### Frontend
- React
- TypeScript
- Vite
