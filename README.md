# 🎥 Video RAG (Retrieval-Augmented Generation) Application

A powerful web application that combines **AI transcription, semantic search, and RAG (Retrieval-Augmented Generation)** to help you **find and understand relevant content inside videos** with timestamp-based navigation.

---

## 🚀 Key Features
* **google-gemma 2bmodel->for transcription and embedding model
* **Video Upload & Transcription** → Accurate speech-to-text using **OpenAI Whisper**
* **Semantic Search** → Contextual retrieval with Hugging Face embeddings + **FAISS**
* **RAG-Powered Q&A** → AI answers your questions using video context with **Cohere**
* **Timestamp Navigation** → Jump directly to exact points in the video
* **JSON Results** → Structured search output with relevance scores
* **Performance Monitoring** → Advanced logging for smooth workflows

---

## 📸 Application Preview

<p align="center">  
  <img src="https://github.com/3vicky67/lecture_nav-A2-/blob/main/outputs_sample/Screenshot%20(45).png" width="600"/>  
</p>  

<p align="center">  
  <img src="https://github.com/3vicky67/lecture_nav-A2-/blob/main/outputs_sample/Screenshot%20(54).png" width="600"/>  
</p>  

<p align="center">  
  <img src="https://github.com/3vicky67/lecture_nav-A2-/blob/main/outputs_sample/Screenshot%20(58).png" width="600"/>  
</p>  

<p align="center">  
  <img src="https://github.com/3vicky67/lecture_nav-A2-/blob/main/outputs_sample/Screenshot%20(59).png" width="600"/>  
</p>  

<p align="center">  
  <img src="https://github.com/3vicky67/lecture_nav-A2-/blob/main/outputs_sample/Screenshot%20(60).png" width="600"/>  
</p>  

<p align="center">  
  <img src="https://github.com/3vicky67/lecture_nav-A2-/blob/main/outputs_sample/Screenshot%20(62).png" width="600"/>  
</p>  

---

## ⚡ Quick Setup

### 🔹 Backend (Flask + Whisper + FAISS)

```bash
cd backend
pip install -r requirements.txt
python run_server.py   # or python server_fast.py
```

Runs at **[http://localhost:5000](http://localhost:5000)**

### 🔹 Frontend (React + Vite + TypeScript)

```bash
cd frontend
npm install
npm run dev
```

Runs at **[http://localhost:5173](http://localhost:5173)**

---

## 🎯 How to Use

1. **Upload a Video**

   * Choose a file or paste a YouTube URL
   * Click **Transcribe Video** → Whisper generates transcript

2. **Search the Video**

   * Enter your query in the search bar
   * Get **JSON-formatted results** including:

     * `video_id`
     * `video_file`
     * `start_time`, `end_time`
     * `snippets` (relevant transcript text)
     * `score` (semantic relevance)

3. **Jump to Timestamp**

   * Click **Jump to Timestamp** → Player seeks to exact point

4. **Ask Questions with AI**

   * Cohere + RAG retrieves relevant transcript snippets
   * AI generates context-aware answers

---

## 🔌 API Endpoints

### **1. Ingest Video**

`POST /api/ingest_video`

```json
{
  "video_id": "unique-video-id"
}
```

### **2. Search Timestamps**

`POST /api/search_timestamps`

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

---

## 🧠 How RAG Works

1. **Transcription** → Whisper converts speech → text
2. **Embeddings** → Transcript split into chunks + encoded with Hugging Face models
3. **Vector Indexing** → FAISS stores & indexes high-dimensional vectors
4. **Semantic Search** → Queries embedded → cosine similarity search
5. **Context Retrieval** → Most relevant transcript snippets selected
6. **AI Answering** → Cohere generates context-aware responses

---

## 🛠️ Tech Stack

* **Backend**: Flask, Whisper, Cohere, PyTorch, Transformers, FAISS
* **Frontend**: React, TypeScript, Vite
* **Vector DB**: FAISS for similarity search
* **Embeddings**: Google Gemma-2-2b (fallback → `all-MiniLM-L6-v2`)

---

## ⚙️ Troubleshooting

* ✅ Install **FFmpeg** & add to PATH
* ✅ Ensure both servers (backend: `5000`, frontend: `5173`) are running
* ✅ Supported video formats: `MP4`, `MKV`, `WebM`
* ✅ First run may take time (models downloading)
* ✅ GPU automatically used if CUDA available

---

## 📊 Dependencies

**Backend**

* Flask, Flask-CORS
* OpenAI Whisper
* Cohere
* yt-dlp
* PyTorch + Transformers
* FAISS + NumPy

**Frontend**

* React
* TypeScript
* Vite

---

## 🌟 Why This Project?

This system makes **long-form video content accessible** through:

* Fast **contextual search**
* Accurate **speech transcription**
* AI-powered **Q&A**
* Seamless **timestamp navigation**

Perfect for **lecture videos, tutorials, corporate training, or any long-form content.**

