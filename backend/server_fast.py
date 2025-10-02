from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import uuid
import json
import time
from typing import List, Dict
from werkzeug.utils import secure_filename

from fast.config import EMBEDDING_MODEL, DEVICE, WHISPER_MODEL, log_json
from fast.db import VIDEO_DB, MODEL_CACHE
from fast.embeddings import load_embedding_model, embed_texts_fast, embed_query_fast
from fast.ai import ask_cohere
from fast.transcribe import transcribe_video_fast as _transcribe_video_fast
try:
    from fast.transcribe import transcribe_video_with_progress as _transcribe_video_with_progress
except Exception:
    _transcribe_video_with_progress = None
from fast.windows import build_overlapping_windows, choose_optimal_window_size
from fast.downloader import download_video
from fast.srt import parse_srt_file
import faiss

"""
Fast Video RAG server (modularized).
Routes and behavior preserved; implementation moved into fast/* modules.
"""

from collections import defaultdict

METRICS = {
    "search_latencies": [],
    "ingest_latencies": [],
    "search_count": 0,
    "ingest_count": 0,
    "error_count": 0
}

from fast.evaluation import (
    load_gold_dataset,
    save_gold_dataset,
    evaluate_query_performance,
)
from fast.hybrid_retrieval import HybridRetriever, ReRanker
from fast.citations import CitationManager
from fast.safety import safety_filter
from fast.monitoring import monitor
from fast.fallback import fallback_manager
GOLD_DATASET_FILE = "gold_dataset.json"
EVALUATION_RESULTS_FILE = "evaluation_results.json"

def calculate_mrr_at_10(query_results, ground_truth):
    """
    Calculate Mean Reciprocal Rank at 10 for a set of queries
    
    Args:
        query_results: Dict of {query: [results]} where results are sorted by relevance
        ground_truth: Dict of {query: [correct_timestamps]}
    
    Returns:
        float: MRR@10 score (0.0 to 1.0)
    """
    if not query_results or not ground_truth:
        return 0.0
    
    mrr_scores = []
    
    for query, results in query_results.items():
        if query not in ground_truth:
            continue
            
        gt_timestamps = set(ground_truth[query])
        if not gt_timestamps:
            continue
            
        # Find the rank of the first relevant result
        rank = 0
        for i, result in enumerate(results[:10]):  # Only check top 10
            result_start = result.get('t_start', 0)
            result_end = result.get('t_end', 0)
            
            # Check if any ground truth timestamp falls within this result's time range
            for gt_time in gt_timestamps:
                if result_start <= gt_time <= result_end:
                    rank = i + 1
                    break
            
            if rank > 0:
                break
        
        # Calculate reciprocal rank
        if rank > 0:
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)
    
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

def calculate_precision_at_k(results, ground_truth, k=3):
    """Calculate Precision@K for a single query"""
    if not results or not ground_truth:
        return 0.0
    
    gt_timestamps = set(ground_truth)
    relevant_count = 0
    
    for result in results[:k]:
        result_start = result.get('t_start', 0)
        result_end = result.get('t_end', 0)
        
        # Check if any ground truth timestamp falls within this result's time range
        for gt_time in gt_timestamps:
            if result_start <= gt_time <= result_end:
                relevant_count += 1
                break
    
    return relevant_count / min(k, len(results))

def evaluate_query_performance(video_id, query, ground_truth_timestamps, k=3):
    """
    Evaluate performance for a single query against ground truth
    
    Returns:
        dict: Evaluation metrics for this query
    """
    try:
        # Perform search
        start_time = time.time()
        
        db_entry = VIDEO_DB[video_id]
        index = db_entry["faiss_index"]
        metadatas = db_entry["metadatas"]
        win_segments = db_entry.get("win_segments", [])
        
        # Semantic search
        q_vec = embed_query_fast(query).astype("float32").reshape(1, -1)
        D, I = index.search(q_vec, k)
        D = D[0]; I = I[0]
        
        # Format results
        results = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0 or idx >= len(metadatas):
                continue
            md = metadatas[idx]
            seg = win_segments[md["idx"]]
            
            results.append({
                "t_start": md.get("start", 0),
                "t_end": md.get("end", 0),
                "title": f"Segment {md.get('idx', 0) + 1}",
                "snippet": str(seg.get("segment_text", "")),
                "score": float(score)
            })
        
        search_latency = time.time() - start_time
        
        # Calculate metrics
        mrr = calculate_mrr_at_10({query: results}, {query: ground_truth_timestamps})
        precision = calculate_precision_at_k(results, ground_truth_timestamps, k)
        
        return {
            "query": query,
            "mrr_at_10": mrr,
            "precision_at_k": precision,
            "latency": search_latency,
            "results_count": len(results),
            "ground_truth_count": len(ground_truth_timestamps)
        }
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "mrr_at_10": 0.0,
            "precision_at_k": 0.0,
            "latency": 0.0
        }

def run_ablation_study(video_id, queries, window_sizes=[30, 45, 60], overlap_ratios=[0.25, 0.33, 0.5]):
    """
    Run ablation study comparing different window sizes and overlap ratios
    
    Args:
        video_id: Video to test
        queries: List of test queries
        window_sizes: List of window sizes to test
        overlap_ratios: List of overlap ratios to test
    
    Returns:
        dict: Ablation study results
    """
    if video_id not in VIDEO_DB:
        return {"error": f"Video ID {video_id} not found"}
    
    results = {}
    
    for window_size in window_sizes:
        for overlap_ratio in overlap_ratios:
            overlap_seconds = int(window_size * overlap_ratio)
            config_key = f"w{window_size}_o{overlap_ratio}"
            
            print(f"🧪 Testing config: {window_size}s windows, {overlap_seconds}s overlap")
            
            try:
                # Re-segment video with new parameters
                segments = VIDEO_DB[video_id]["segments"]
                win_segments = build_overlapping_windows(segments, window_size, overlap_seconds)
                
                # Create new embeddings and index
                texts = [str(w["segment_text"]) for w in win_segments]
                metadatas = [{
                    "start": float(w["t_start"]),
                    "end": float(w["t_end"]),
                    "idx": int(w["idx"]),
                } for w in win_segments]
                
                embeddings = embed_texts_fast(texts, batch_size=16)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                
                # Test each query
                query_results = []
                total_latency = 0
                
                for query in queries:
                    start_time = time.time()
                    
                    # Search
                    q_vec = embed_query_fast(query).astype("float32").reshape(1, -1)
                    D, I = index.search(q_vec, 3)
                    D = D[0]; I = I[0]
                    
                    # Format results
                    results_list = []
                    for score, idx in zip(D.tolist(), I.tolist()):
                        if idx < 0 or idx >= len(metadatas):
                            continue
                        md = metadatas[idx]
                        seg = win_segments[md["idx"]]
                        
                        results_list.append({
                            "t_start": md.get("start", 0),
                            "t_end": md.get("end", 0),
                            "title": f"Segment {md.get('idx', 0) + 1}",
                            "snippet": str(seg.get("segment_text", "")),
                            "score": float(score)
                        })
                    
                    latency = time.time() - start_time
                    total_latency += latency
                    
                    query_results.append({
                        "query": query,
                        "results": results_list,
                        "latency": latency
                    })
                
                # Calculate average metrics
                avg_latency = total_latency / len(queries) if queries else 0
                
                results[config_key] = {
                    "window_size": window_size,
                    "overlap_seconds": overlap_seconds,
                    "overlap_ratio": overlap_ratio,
                    "segments_count": len(win_segments),
                    "avg_latency": avg_latency,
                    "query_results": query_results
                }
                
            except Exception as e:
                results[config_key] = {
                    "error": str(e),
                    "window_size": window_size,
                    "overlap_seconds": overlap_seconds,
                    "overlap_ratio": overlap_ratio
                }
    
    return results

def log_metrics(operation: str, latency: float, success: bool = True):
    """Log operation metrics"""
    if operation == "search":
        METRICS["search_latencies"].append(latency)
        METRICS["search_count"] += 1
    elif operation == "ingest":
        METRICS["ingest_latencies"].append(latency)
        METRICS["ingest_count"] += 1
    
    if not success:
        METRICS["error_count"] += 1

def get_p95_latency(latencies: List[float]) -> float:
    """Calculate 95th percentile latency"""
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    index = int(0.95 * len(sorted_latencies))
    return sorted_latencies[min(index, len(sorted_latencies) - 1)]

# removed in favor of modular imports

# ------------------ Format Time ------------------ #
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}.{int(sec):02d}m"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load embedding model on startup
load_embedding_model()

# ------------------ Default Home Route ------------------ #
@app.route("/", methods=["GET"])
def home():
    return "🚀 Fast Video RAG API is running! Use /api/ingest_video and /api/search_timestamps endpoints."

# ------------------ Fast API Endpoints ------------------ #
@app.route("/api/transcript", methods=["GET"])
def get_transcript():
    """Return raw transcript segments for a given video_id.

    Query params:
      - video_id: required

    Response JSON:
      {
        "video_id": str,
        "filename": str | None,
        "segments": [{"start": float, "end": float, "text": str}, ...]
      }
    """
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404

    entry = VIDEO_DB[video_id]
    segments = entry.get("segments", [])
    # Try to derive a human-friendly filename from the source path/url
    src = entry.get("path")
    filename = None
    try:
        if isinstance(src, str):
            filename = os.path.basename(src) or src
    except Exception:
        filename = None

    return jsonify({
        "video_id": video_id,
        "filename": filename,
        "segments": segments,
    })
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
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            source_type = "file"
            video_source = filename
            
            log_json("ingest.started", {"video_file": filename, "video_id": video_id}, trace_id)
            
            # Check if it's an SRT file
            if filename.lower().endswith('.srt'):
                segments = parse_srt_file(file_path)
                video_path = None  # No video file for SRT
            else:
                video_path = file_path
            
        # Check if it's JSON data with URL or file path
        elif request.is_json:
            data = request.json
            video_url = data.get("video_url")
            video_file = data.get("video_file")
            
            if video_url:
                log_json("ingest.started", {"video_url": video_url, "video_id": video_id}, trace_id)
                video_path = download_video(video_url)  # Now supports YouTube, Vimeo, etc.
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

        # Handle transcription based on file type
        if video_path:
            # Fast transcription with tiny model
            print("🎬 Starting fast transcription...")
            segments = _transcribe_video_fast(video_path, model_name=WHISPER_MODEL)
        else:
            # SRT file already parsed, use those segments
            print("📄 Using SRT file segments...")
            # segments already set from SRT parsing
        
        # Calculate video duration and choose optimal window size
        video_duration = max(seg.get("end", 0) for seg in segments) if segments else 0
        window_seconds = choose_optimal_window_size(video_duration)
        overlap_seconds = max(15, window_seconds // 3)  # 15s minimum, or 1/3 of window size
        
        print(f"📊 Video duration: {video_duration:.1f}s, using {window_seconds}s windows with {overlap_seconds}s overlap")
        
        # Build windows with overlap, then embed those windows
        win_segments = build_overlapping_windows(segments, window_seconds, overlap_seconds)

        texts, metadatas = [], []
        for w in win_segments:
            texts.append(str(w["segment_text"]))
            metadatas.append({
                "start": float(w["t_start"]),
                "end": float(w["t_end"]),
                "idx": int(w["idx"]),
            })

        # Fast embedding creation
        print("🧠 Creating embeddings...")
        embeddings = embed_texts_fast(texts, batch_size=16)
        
        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Store in database
        VIDEO_DB[video_id] = {
            "segments": segments,          # raw whisper segments
            "win_segments": win_segments,  # windowed segments used for retrieval
            "source": source_type,
            "path": video_source,
            "faiss_index": index,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }

        elapsed = time.time() - start_ts
        log_json("ingest.completed", {"video_id": video_id, "segments": len(segments), "elapsed_s": elapsed}, trace_id)
        print(f"✅ Video processed in {elapsed:.2f} seconds!")
        return jsonify({"video_id": video_id, "processing_time": elapsed})

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
    start_time = time.time()
    
    # Safety validation
    safety_result = safety_filter.validate_search_query(query)
    if not safety_result['valid']:
        return jsonify({"error": safety_result['error']}), 400
    
    query = safety_result.get('sanitized_query', query)
    
    if video_id not in VIDEO_DB:
        log_metrics("search", time.time() - start_time, success=False)
        return jsonify({"error": f"Video ID {video_id} not found"}), 404

    def primary_search():
        """Primary search function with hybrid retrieval"""
        db_entry = VIDEO_DB[video_id]
        
        # Initialize hybrid retriever if not exists
        if 'hybrid_retriever' not in db_entry:
            win_segments = db_entry.get("win_segments", [])
            texts = []
            for seg in win_segments:
                if isinstance(seg, dict):
                    texts.append(str(seg.get("segment_text", "")))
                else:
                    texts.append(str(seg))
            
            metadatas = db_entry["metadatas"]
            embeddings = db_entry["embeddings"]
            
            hybrid_retriever = HybridRetriever()
            hybrid_retriever.build_index(texts, metadatas, embeddings)
            db_entry['hybrid_retriever'] = hybrid_retriever
        
        # Hybrid search
        hybrid_retriever = db_entry['hybrid_retriever']
        results = hybrid_retriever.hybrid_search(query, k=k*2, alpha=0.7)
        
        # Re-ranking
        reranker = ReRanker()
        query_embedding = embed_query_fast(query).astype('float32')
        reranked_results = reranker.rerank(query, results, query_embedding)
        
        return reranked_results[:k]
    
    def fallback_search():
        """Fallback to simple vector search"""
        db_entry = VIDEO_DB[video_id]
        index = db_entry["faiss_index"]
        metadatas = db_entry["metadatas"]
        win_segments = db_entry.get("win_segments", [])
        
        q_vec = embed_query_fast(query).astype("float32").reshape(1, -1)
        D, I = index.search(q_vec, k)
        D = D[0]; I = I[0]
        
        results = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0 or idx >= len(metadatas):
                continue
            md = metadatas[idx]
            
            # Ensure md is a dictionary
            if isinstance(md, str):
                print(f"Warning: metadata at index {idx} is a string: {md}")
                continue
                
            if not isinstance(md, dict):
                print(f"Warning: metadata at index {idx} is not a dict: {type(md)}")
                continue
            
            # Get segment index safely
            seg_idx = md.get("idx", 0)
            if seg_idx >= len(win_segments):
                print(f"Warning: segment index {seg_idx} out of range")
                continue
                
            seg = win_segments[seg_idx]
            
            # Ensure seg is a dictionary
            if isinstance(seg, str):
                print(f"Warning: segment at index {seg_idx} is a string: {seg}")
                continue
                
            if not isinstance(seg, dict):
                print(f"Warning: segment at index {seg_idx} is not a dict: {type(seg)}")
                continue
            
            results.append({
                'index': idx,
                'score': float(score),
                'metadata': md,
                'text': str(seg.get("segment_text", ""))
            })
        
        return results
    
    try:
        # Execute with fallback
        search_results = fallback_manager.execute_with_fallback(
            'search', 
            primary_search, 
            fallback_search,
            cache_key=f"search_{video_id}_{hash(query)}"
        )
        
        # Add citations
        citation_manager = CitationManager()
        cited_results = citation_manager.add_citations_to_results(search_results, video_id)
        formatted_results = citation_manager.format_citations_for_display(cited_results)
        
        # Format for API response
        final_results = []
        video_url = VIDEO_DB[video_id]["path"]
        
        for result in formatted_results:
            md = result['metadata']
            start_seconds = md.get("start", 0)
            end_seconds = md.get("end", 0)
            start_formatted = f"{int(start_seconds//60):02d}:{int(start_seconds%60):02d}"
            end_formatted = f"{int(end_seconds//60):02d}:{int(end_seconds%60):02d}"
            
            final_results.append({
                "video_id": video_id,
                "video_file": os.path.basename(video_url) if isinstance(video_url, str) else str(video_url),
                "t_start": start_seconds,
                "t_end": end_seconds,
                "start_time": start_formatted,
                "end_time": end_formatted,
                "title": f"Segment {md.get('idx', 0) + 1} ({start_formatted}-{end_formatted})",
                "snippet": result['text'],
                "score": result['score'],
                "citation": result.get('citation', {}),
                "source_id": result.get('source_id', ''),
                "spans": result.get('spans', {})
            })

        # Build concise one-sentence answer
        concise_answer = None
        try:
            if final_results:
                snippets = [r['snippet'] for r in final_results[:3]]
                joined = "\n".join(snippets)
                
                # Try to use Cohere AI first
                try:
                    prompt = (
                        "Based on the following video transcript snippets, provide a single, concise sentence that directly answers the user's question.\n\n"
                        f"Question: {query}\n"
                        f"Relevant transcript snippets:\n{joined}\n\n"
                        "IMPORTANT: Respond with exactly ONE sentence. Be direct and factual. If the answer isn't clear from the snippets, say 'The answer is not clearly mentioned in the video.'"
                    )
                    concise_answer = ask_cohere(prompt)
                    
                    # Ensure it's actually one sentence
                    if concise_answer and len(concise_answer.split('.')) > 2:
                        # If multiple sentences, take the first one
                        first_sentence = concise_answer.split('.')[0] + '.'
                        concise_answer = first_sentence
                        
                except Exception as cohere_error:
                    print(f"Cohere AI unavailable: {cohere_error}")
                    # Fallback: Generate a simple answer from the snippets
                    if snippets:
                        # Find the most relevant snippet that contains keywords from the query
                        query_words = query.lower().split()
                        best_snippet = snippets[0]  # Default to first snippet
                        
                        for snippet in snippets:
                            snippet_lower = snippet.lower()
                            matches = sum(1 for word in query_words if word in snippet_lower)
                            if matches > 0:
                                best_snippet = snippet
                                break
                        
                        # Create a simple one-sentence answer
                        if len(best_snippet) > 100:
                            concise_answer = best_snippet[:97] + "..."
                        else:
                            concise_answer = best_snippet
                    else:
                        concise_answer = "Based on the video content, relevant information was found but cannot be summarized at this time."
                        
        except Exception as e:
            print(f"Error generating AI answer: {e}")
            concise_answer = "Relevant content found in the video, but AI summarization is currently unavailable."

        # Save to output.json
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

        # Track metrics
        search_latency = time.time() - start_time
        log_metrics("search", search_latency, success=True)
        monitor.track_latency("search", search_latency * 1000, success=True)
        
        # Generate citation summary
        citation_summary = citation_manager.get_citation_summary(cited_results)
        
        log_json("search.completed", {
            "video_id": video_id, 
            "query": query, 
            "results_count": len(final_results), 
            "latency": search_latency,
            "citations": citation_summary
        }, trace_id)
        
        return jsonify({
            "query": query, 
            "answer": concise_answer, 
            "results": final_results,
            "citations": citation_summary,
            "performance": {
                "latency_ms": search_latency * 1000,
                "p95_target_met": search_latency <= 2.0
            }
        })

    except Exception as e:
        search_latency = time.time() - start_time
        log_metrics("search", search_latency, success=False)
        monitor.track_latency("search", search_latency * 1000, success=False)
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

        # Fast semantic search for relevant context
        q_vec = embed_query_fast(question).astype("float32").reshape(1, -1)
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

        # Create RAG prompt with retrieved context for one-sentence response
        context_block = "\n\n---\n\n".join([f"[{i}] {t}" for i, t in enumerate(retrieved_texts, start=1)])
        
        try:
            rag_prompt = (
                f"Based on the following video transcript passages, provide a single, concise sentence that directly answers the question.\n\n"
                f"Passages:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "IMPORTANT: Respond with exactly ONE sentence. Be direct and factual. If the answer isn't clear from the passages, say 'The answer is not clearly mentioned in the video.'"
            )
            answer = ask_cohere(rag_prompt)
            
            # Ensure it's actually one sentence
            if answer and len(answer.split('.')) > 2:
                # If multiple sentences, take the first one
                first_sentence = answer.split('.')[0] + '.'
                answer = first_sentence
                
        except Exception as cohere_error:
            print(f"Cohere AI unavailable for ask_ai: {cohere_error}")
            # Fallback: Generate a simple answer from the retrieved texts
            if retrieved_texts:
                # Find the most relevant passage that contains keywords from the question
                question_words = question.lower().split()
                best_passage = retrieved_texts[0]  # Default to first passage
                
                for passage in retrieved_texts:
                    passage_lower = passage.lower()
                    matches = sum(1 for word in question_words if word in passage_lower)
                    if matches > 0:
                        best_passage = passage
                        break
                
                # Create a simple one-sentence answer
                if len(best_passage) > 100:
                    answer = best_passage[:97] + "..."
                else:
                    answer = best_passage
            else:
                answer = "Based on the video content, relevant information was found but cannot be summarized at this time."

        log_json("ai.completed", {"video_id": video_id, "question": question, "retrieved_count": len(retrieved_texts)}, trace_id)
        return jsonify({"answer": answer, "retrieved": retrieval_meta})

    except Exception as e:
        log_json("ai.exception", {"error": str(e)}, trace_id, level="error")
        return jsonify({"error": str(e)}), 500

# ------------------ Performance Monitoring ------------------ #
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "device": DEVICE,
        "embedding_model": EMBEDDING_MODEL,
        "whisper_model": WHISPER_MODEL,
        "videos_processed": len(VIDEO_DB),
        "models_cached": len(MODEL_CACHE)
    })

@app.route("/api/ingest_video_async", methods=["POST"])
def ingest_video_async():
    """
    Asynchronous video ingestion with progress updates.
    Returns immediately with a job ID, progress can be checked via /api/progress/<job_id>
    """
    trace_id = uuid.uuid4().hex[:8]
    job_id = str(uuid.uuid4())
    
    try:
        data = request.json
        video_url = data.get("video_url")
        
        if not video_url:
            return jsonify({"error": "video_url is required"}), 400
        
        # Store job status
        METRICS["async_jobs"] = METRICS.get("async_jobs", {})
        METRICS["async_jobs"][job_id] = {
            "status": "started",
            "progress": 0,
            "message": "Starting video download...",
            "video_url": video_url,
            "trace_id": trace_id,
            "start_time": time.time()
        }
        
        def progress_callback(progress, message):
            if job_id in METRICS["async_jobs"]:
                METRICS["async_jobs"][job_id]["progress"] = progress
                METRICS["async_jobs"][job_id]["message"] = message
        
        def transcription_callback(result, error):
            if error:
                METRICS["async_jobs"][job_id]["status"] = "error"
                METRICS["async_jobs"][job_id]["error"] = str(error)
            else:
                METRICS["async_jobs"][job_id]["status"] = "completed"
                METRICS["async_jobs"][job_id]["result"] = result
                METRICS["async_jobs"][job_id]["progress"] = 100
        
        # Start async transcription
        def async_worker():
            try:
                progress_callback(10, "Downloading video...")
                video_path = download_video(video_url)
                
                progress_callback(30, "Starting transcription...")
                if _transcribe_video_with_progress:
                    segments = _transcribe_video_with_progress(video_path, WHISPER_MODEL, progress_callback)
                else:
                    progress_callback(40, "Transcribing (no progress updates available)...")
                    segments = _transcribe_video_fast(video_path, WHISPER_MODEL)
                
                # Process embeddings
                progress_callback(80, "Generating embeddings...")
                
                # Calculate video duration and choose optimal window size
                video_duration = max(seg.get("end", 0) for seg in segments) if segments else 0
                window_seconds = choose_optimal_window_size(video_duration)
                overlap_seconds = max(15, window_seconds // 3)  # 15s minimum, or 1/3 of window size
                
                # Build windows with overlap, then embed those windows
                win_segments = build_overlapping_windows(segments, window_seconds, overlap_seconds)

                texts, metadatas = [], []
                for w in win_segments:
                    texts.append(str(w["segment_text"]))
                    metadatas.append({
                        "start": float(w["t_start"]),
                        "end": float(w["t_end"]),
                        "idx": int(w["idx"]),
                    })
                
                embeddings = embed_texts_fast(texts, batch_size=16)
                
                # Create FAISS index
                progress_callback(90, "Creating search index...")
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                
                # Store in database
                video_id = str(uuid.uuid4())
                VIDEO_DB[video_id] = {
                    "segments": segments,          # raw whisper segments
                    "win_segments": win_segments,  # windowed segments used for retrieval
                    "source": "url",
                    "path": video_url,
                    "faiss_index": index,
                    "embeddings": embeddings,
                    "metadatas": metadatas,
                    "created_at": time.time()
                }
                
                METRICS["async_jobs"][job_id]["status"] = "completed"
                METRICS["async_jobs"][job_id]["video_id"] = video_id
                METRICS["async_jobs"][job_id]["progress"] = 100
                METRICS["async_jobs"][job_id]["message"] = f"Video processed successfully! {len(segments)} segments created."
                
            except Exception as e:
                METRICS["async_jobs"][job_id]["status"] = "error"
                METRICS["async_jobs"][job_id]["error"] = str(e)
                METRICS["async_jobs"][job_id]["message"] = f"Error: {str(e)}"
        
        # Start the async worker
        import threading
        thread = threading.Thread(target=async_worker)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "Video processing started. Use /api/progress/<job_id> to check status."
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/progress/<job_id>", methods=["GET"])
def get_progress(job_id):
    """Get the progress of an async transcription job."""
    if "async_jobs" not in METRICS or job_id not in METRICS["async_jobs"]:
        return jsonify({"error": "Job not found"}), 404
    
    job = METRICS["async_jobs"][job_id]
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "video_url": job.get("video_url"),
        "video_id": job.get("video_id"),
        "error": job.get("error")
    })

@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Get performance metrics"""
    search_p95 = get_p95_latency(METRICS["search_latencies"])
    ingest_p95 = get_p95_latency(METRICS["ingest_latencies"])
    
    return jsonify({
        "search_metrics": {
            "count": METRICS["search_count"],
            "p95_latency": search_p95,
            "avg_latency": sum(METRICS["search_latencies"]) / len(METRICS["search_latencies"]) if METRICS["search_latencies"] else 0,
            "meets_p95_target": search_p95 <= 2.0
        },
        "ingest_metrics": {
            "count": METRICS["ingest_count"],
            "p95_latency": ingest_p95,
            "avg_latency": sum(METRICS["ingest_latencies"]) / len(METRICS["ingest_latencies"]) if METRICS["ingest_latencies"] else 0
        },
        "error_count": METRICS["error_count"],
        "videos_processed": len(VIDEO_DB),
        "fallback_status": fallback_manager.get_system_status()
    })

@app.route("/api/monitoring/cost-report", methods=["GET"])
def cost_report():
    """Get cost and latency report"""
    hours = int(request.args.get('hours', 24))
    report = monitor.get_comprehensive_report(hours)
    return jsonify(report)

@app.route("/api/monitoring/fallback-status", methods=["GET"])
def fallback_status():
    """Get fallback system status"""
    return jsonify(fallback_manager.get_system_status())

@app.route("/api/monitoring/reset-circuit-breaker", methods=["POST"])
def reset_circuit_breaker():
    """Reset circuit breaker for an operation"""
    data = request.json
    operation = data.get('operation')
    if operation:
        fallback_manager.reset_circuit_breaker(operation)
        return jsonify({"message": f"Circuit breaker reset for {operation}"})
    return jsonify({"error": "Operation required"}), 400

# ------------------ Evaluation API Endpoints ------------------ #
@app.route("/api/evaluation/gold_dataset", methods=["GET"])
def get_gold_dataset():
    """Get current gold standard dataset"""
    dataset = load_gold_dataset()
    return jsonify({
        "dataset": dataset,
        "query_count": len(dataset),
        "total_annotations": sum(len(timestamps) for timestamps in dataset.values())
    })

@app.route("/api/evaluation/gold_dataset", methods=["POST"])
def add_gold_annotation():
    """Add a new query→timestamp annotation to gold dataset"""
    data = request.json
    query = data.get("query")
    timestamps = data.get("timestamps", [])
    video_id = data.get("video_id")
    
    if not query or not timestamps:
        return jsonify({"error": "Query and timestamps required"}), 400
    
    dataset = load_gold_dataset()
    if video_id not in dataset:
        dataset[video_id] = {}
    
    dataset[video_id][query] = timestamps
    save_gold_dataset(dataset)
    
    return jsonify({
        "message": "Annotation added successfully",
        "query": query,
        "timestamps": timestamps,
        "total_queries": len(dataset.get(video_id, {}))
    })

@app.route("/api/evaluation/evaluate", methods=["POST"])
def evaluate_performance():
    """Evaluate performance against gold dataset"""
    data = request.json
    video_id = data.get("video_id")
    queries = data.get("queries", [])
    
    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404
    
    dataset = load_gold_dataset()
    video_annotations = dataset.get(video_id, {})
    
    # Require at least 30 query->timestamp pairs
    total_pairs = sum(len(ts) for ts in video_annotations.values())
    if total_pairs < 30:
        return jsonify({"error": "Insufficient gold annotations: need at least 30 annotated timestamps"}), 400
    
    # Use provided queries or all available queries
    test_queries = queries if queries else list(video_annotations.keys())
    
    results = []
    total_mrr = 0
    total_precision = 0
    total_latency = 0
    
    for query in test_queries:
        if query not in video_annotations:
            continue
            
        ground_truth = video_annotations[query]
        eval_result = evaluate_query_performance(video_id, query, ground_truth)
        results.append(eval_result)
        
        total_mrr += eval_result.get("mrr_at_10", 0)
        total_precision += eval_result.get("precision_at_k", 0)
        total_latency += eval_result.get("latency", 0)
    
    # Calculate averages
    query_count = len(results)
    avg_mrr = total_mrr / query_count if query_count > 0 else 0
    avg_precision = total_precision / query_count if query_count > 0 else 0
    avg_latency = total_latency / query_count if query_count > 0 else 0
    
    # Check if meets requirements
    meets_mrr_target = avg_mrr >= 0.5  # Reasonable MRR target
    meets_latency_target = avg_latency <= 2.0
    
    evaluation_summary = {
        "video_id": video_id,
        "query_count": query_count,
        "avg_mrr_at_10": avg_mrr,
        "avg_precision_at_k": avg_precision,
        "avg_latency": avg_latency,
        "meets_mrr_target": meets_mrr_target,
        "meets_latency_target": meets_latency_target,
        "overall_pass": meets_mrr_target and meets_latency_target,
        "detailed_results": results
    }
    
    # Save evaluation results
    with open(EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    return jsonify(evaluation_summary)

@app.route("/api/evaluation/ablation", methods=["POST"])
def run_ablation_study_endpoint():
    """Run ablation study comparing different configurations"""
    data = request.json
    video_id = data.get("video_id")
    queries = data.get("queries", [])
    window_sizes = data.get("window_sizes", [30, 60])
    overlap_ratios = data.get("overlap_ratios", [0.25, 0.33, 0.5])
    
    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404
    
    # Use default test queries if none provided
    if not queries:
        queries = [
            "explain machine learning",
            "what is gradient descent", 
            "neural networks basics",
            "deep learning applications",
            "data preprocessing techniques"
        ]
    
    results = run_ablation_study(video_id, queries, window_sizes, overlap_ratios)
    
    return jsonify({
        "video_id": video_id,
        "test_queries": queries,
        "configurations_tested": len(results),
        "results": results
    })

@app.route("/api/evaluation/test_suite", methods=["POST"])
def run_full_test_suite():
    """Run complete test suite: MRR@10, P95 latency, and ablation study"""
    data = request.json
    video_id = data.get("video_id")
    
    if video_id not in VIDEO_DB:
        return jsonify({"error": f"Video ID {video_id} not found"}), 404
    
    # Load gold dataset
    dataset = load_gold_dataset()
    video_annotations = dataset.get(video_id, {})
    
    test_results = {
        "video_id": video_id,
        "timestamp": time.time(),
        "tests": {}
    }
    
    # Test 1: MRR@10 Evaluation
    if video_annotations:
        print("🧪 Running MRR@10 evaluation...")
        eval_data = {"video_id": video_id, "queries": list(video_annotations.keys())}
        mrr_response = evaluate_performance()
        test_results["tests"]["mrr_evaluation"] = mrr_response.get_json()
    else:
        test_results["tests"]["mrr_evaluation"] = {"error": "No gold annotations available"}
    
    # Test 2: P95 Latency Test
    print("🧪 Running P95 latency test...")
    current_metrics = METRICS.copy()
    search_p95 = get_p95_latency(current_metrics["search_latencies"])
    test_results["tests"]["latency_test"] = {
        "p95_latency": search_p95,
        "meets_target": search_p95 <= 2.0,
        "sample_count": len(current_metrics["search_latencies"])
    }
    
    # Test 3: Ablation Study
    print("🧪 Running ablation study...")
    ablation_response = run_ablation_study_endpoint()
    test_results["tests"]["ablation_study"] = ablation_response.get_json()
    
    # Overall assessment
    mrr_pass = test_results["tests"]["mrr_evaluation"].get("overall_pass", False)
    latency_pass = test_results["tests"]["latency_test"].get("meets_target", False)
    
    test_results["overall_assessment"] = {
        "mrr_test_passed": mrr_pass,
        "latency_test_passed": latency_pass,
        "all_tests_passed": mrr_pass and latency_pass,
        "test_timestamp": time.time()
    }
    
    return jsonify(test_results)

# ------------------ Run Flask ------------------ #
if __name__ == "__main__":
    print("🚀 Starting Fast Video RAG Server...")
    print(f"📱 Device: {DEVICE}")
    print(f"🧠 Embedding Model: {EMBEDDING_MODEL}")
    print(f"🎬 Whisper Model: {WHISPER_MODEL}")
    app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)
