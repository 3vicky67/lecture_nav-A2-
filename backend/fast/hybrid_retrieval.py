"""
Hybrid Retrieval System: BM25 + Vector Search + Re-ranker
Combines keyword-based (BM25) and semantic (vector) search for better results
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from .embeddings import embed_query_fast

class HybridRetriever:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.faiss_index = None
        self.texts = []
        self.metadatas = []
        self.tfidf_matrix = None
        
    def build_index(self, texts: List[str], metadatas: List[Dict], embeddings: np.ndarray):
        """Build both BM25 and vector indices"""
        self.texts = texts
        self.metadatas = metadatas
        
        # Build TF-IDF (BM25-like) index
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Build FAISS vector index
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype('float32'))
        
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining BM25 and vector search
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (1-alpha for BM25)
        """
        # Vector search
        query_embedding = embed_query_fast(query).astype('float32').reshape(1, -1)
        vector_scores, vector_indices = self.faiss_index.search(query_embedding, k*2)
        vector_scores = vector_scores[0]
        vector_indices = vector_indices[0]
        
        # BM25 search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        bm25_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Combine scores
        combined_scores = {}
        for i, (vec_score, vec_idx) in enumerate(zip(vector_scores, vector_indices)):
            if vec_idx >= 0 and vec_idx < len(bm25_scores):
                bm25_score = bm25_scores[vec_idx]
                combined_score = alpha * vec_score + (1 - alpha) * bm25_score
                combined_scores[vec_idx] = combined_score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for idx, score in sorted_indices[:k]:
            if idx < len(self.metadatas):
                results.append({
                    'index': idx,
                    'score': float(score),
                    'metadata': self.metadatas[idx],
                    'text': self.texts[idx]
                })
        
        return results

class ReRanker:
    def __init__(self):
        self.weights = {
            'semantic': 0.4,
            'keyword': 0.3,
            'position': 0.2,
            'length': 0.1
        }
    
    def rerank(self, query: str, results: List[Dict], query_embedding: np.ndarray) -> List[Dict]:
        """
        Re-rank results using multiple signals
        
        Args:
            query: Original query
            results: Initial search results
            query_embedding: Query embedding for semantic similarity
        """
        reranked = []
        
        for result in results:
            text = result['text']
            metadata = result['metadata']
            
            # Calculate re-ranking features
            features = self._calculate_features(query, text, metadata, query_embedding)
            
            # Weighted combination
            final_score = sum(self.weights[k] * v for k, v in features.items())
            
            result['rerank_score'] = final_score
            result['features'] = features
            reranked.append(result)
        
        # Sort by re-rank score
        return sorted(reranked, key=lambda x: x['rerank_score'], reverse=True)
    
    def _calculate_features(self, query: str, text: str, metadata: Dict, query_embedding: np.ndarray) -> Dict:
        """Calculate re-ranking features"""
        # Semantic similarity (already computed)
        semantic_score = metadata.get('score', 0)
        
        # Keyword overlap
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        keyword_score = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
        
        # Position bias (earlier segments get slight boost)
        position_score = 1.0 / (metadata.get('idx', 1) + 1)
        
        # Length normalization (prefer medium-length segments)
        text_length = len(text.split())
        length_score = 1.0 - abs(text_length - 50) / 100  # Optimal around 50 words
        length_score = max(0, min(1, length_score))
        
        return {
            'semantic': semantic_score,
            'keyword': keyword_score,
            'position': position_score,
            'length': length_score
        }
