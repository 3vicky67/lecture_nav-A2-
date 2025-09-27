import json
import os
from typing import List

from .embeddings import embed_query_fast
from .db import VIDEO_DB

GOLD_DATASET_FILE = "gold_dataset.json"
EVALUATION_RESULTS_FILE = "evaluation_results.json"

def load_gold_dataset():
    if os.path.exists(GOLD_DATASET_FILE):
        with open(GOLD_DATASET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_gold_dataset(dataset):
    with open(GOLD_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

def calculate_mrr_at_10(query_results, ground_truth):
    if not query_results or not ground_truth:
        return 0.0
    mrr_scores = []
    for query, results in query_results.items():
        if query not in ground_truth:
            continue
        gt_timestamps = set(ground_truth[query])
        if not gt_timestamps:
            continue
        rank = 0
        for i, result in enumerate(results[:10]):
            result_start = result.get('t_start', 0)
            result_end = result.get('t_end', 0)
            for gt_time in gt_timestamps:
                if result_start <= gt_time <= result_end:
                    rank = i + 1
                    break
            if rank > 0:
                break
        if rank > 0:
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

def calculate_precision_at_k(results, ground_truth, k=3):
    if not results or not ground_truth:
        return 0.0
    gt_timestamps = set(ground_truth)
    relevant_count = 0
    for result in results[:k]:
        result_start = result.get('t_start', 0)
        result_end = result.get('t_end', 0)
        for gt_time in gt_timestamps:
            if result_start <= gt_time <= result_end:
                relevant_count += 1
                break
    return relevant_count / min(k, len(results))

def evaluate_query_performance(video_id, query, ground_truth_timestamps, k=3):
    try:
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
            seg = win_segments[md["idx"]]
            results.append({
                "t_start": md.get("start", 0),
                "t_end": md.get("end", 0),
                "title": f"Segment {md.get('idx', 0) + 1}",
                "snippet": str(seg.get("segment_text", "")),
                "score": float(score)
            })
        mrr = calculate_mrr_at_10({query: results}, {query: ground_truth_timestamps})
        precision = calculate_precision_at_k(results, ground_truth_timestamps, k)
        return {
            "query": query,
            "mrr_at_10": mrr,
            "precision_at_k": precision,
            "latency": 0.0,
            "results_count": len(results),
            "ground_truth_count": len(ground_truth_timestamps)
        }
    except Exception as e:
        return {"query": query, "error": str(e), "mrr_at_10": 0.0, "precision_at_k": 0.0, "latency": 0.0}


