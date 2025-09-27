"""
Citation System: Source IDs & Spans
Provides proper attribution and source tracking for retrieved content
"""

import uuid
from typing import List, Dict, Any
from datetime import datetime

class CitationManager:
    def __init__(self):
        self.citation_cache = {}
        
    def generate_citation_id(self, video_id: str, segment_idx: int, start_time: float, end_time: float) -> str:
        """Generate unique citation ID"""
        return f"cite_{video_id}_{segment_idx}_{start_time:.1f}_{end_time:.1f}"
    
    def create_citation(self, video_id: str, segment_idx: int, start_time: float, end_time: float, 
                       text: str, score: float, span_start: int = 0, span_end: int = None) -> Dict:
        """Create a citation with source information"""
        citation_id = self.generate_citation_id(video_id, segment_idx, start_time, end_time)
        
        if span_end is None:
            span_end = len(text)
            
        citation = {
            "citation_id": citation_id,
            "source": {
                "video_id": video_id,
                "segment_idx": segment_idx,
                "start_time": start_time,
                "end_time": end_time,
                "timestamp": f"{start_time:.1f}s-{end_time:.1f}s"
            },
            "content": {
                "text": text,
                "span_start": span_start,
                "span_end": span_end,
                "span_text": text[span_start:span_end] if span_end <= len(text) else text[span_start:]
            },
            "metadata": {
                "relevance_score": score,
                "created_at": datetime.now().isoformat(),
                "confidence": min(1.0, max(0.0, score))
            }
        }
        
        return citation
    
    def add_citations_to_results(self, results: List[Dict], video_id: str) -> List[Dict]:
        """Add citations to search results"""
        cited_results = []
        
        for i, result in enumerate(results):
            # Extract metadata
            metadata = result.get('metadata', {})
            segment_idx = metadata.get('idx', i)
            start_time = metadata.get('start', 0)
            end_time = metadata.get('end', 0)
            score = result.get('score', 0)
            text = result.get('text', '')
            
            # Create citation
            citation = self.create_citation(
                video_id=video_id,
                segment_idx=segment_idx,
                start_time=start_time,
                end_time=end_time,
                text=text,
                score=score
            )
            
            # Add citation to result
            result['citation'] = citation
            result['source_id'] = citation['citation_id']
            result['spans'] = {
                'start': citation['content']['span_start'],
                'end': citation['content']['span_end'],
                'text': citation['content']['span_text']
            }
            
            cited_results.append(result)
        
        return cited_results
    
    def format_citations_for_display(self, results: List[Dict]) -> List[Dict]:
        """Format citations for frontend display"""
        formatted = []
        
        for result in results:
            citation = result.get('citation', {})
            source = citation.get('source', {})
            
            formatted_result = {
                **result,
                'display_citation': {
                    'id': citation.get('citation_id', ''),
                    'source': f"Video {source.get('video_id', '')[:8]}...",
                    'timestamp': source.get('timestamp', ''),
                    'confidence': f"{citation.get('metadata', {}).get('confidence', 0):.2f}",
                    'span_preview': citation.get('content', {}).get('span_text', '')[:100] + '...'
                }
            }
            
            formatted.append(formatted_result)
        
        return formatted
    
    def get_citation_summary(self, results: List[Dict]) -> Dict:
        """Generate citation summary for the response"""
        if not results:
            return {"total_citations": 0, "sources": [], "confidence_avg": 0}
        
        sources = []
        confidences = []
        
        for result in results:
            citation = result.get('citation', {})
            source = citation.get('source', {})
            
            sources.append({
                'video_id': source.get('video_id', ''),
                'timestamp': source.get('timestamp', ''),
                'segment': source.get('segment_idx', 0)
            })
            
            confidences.append(citation.get('metadata', {}).get('confidence', 0))
        
        return {
            "total_citations": len(results),
            "sources": sources,
            "confidence_avg": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_range": {
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0
            }
        }
