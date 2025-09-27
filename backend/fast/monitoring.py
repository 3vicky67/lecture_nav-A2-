"""
Cost & Latency Monitoring System
Tracks API costs, latency metrics, and generates comprehensive reports
"""

import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    """Cost tracking for API calls"""
    operation: str
    model: str
    tokens_used: int
    cost_per_token: float
    total_cost: float
    timestamp: str

@dataclass
class LatencyMetrics:
    """Latency tracking for operations"""
    operation: str
    duration_ms: float
    p95_latency: float
    success: bool
    timestamp: str

class CostLatencyMonitor:
    def __init__(self):
        self.cost_metrics: List[CostMetrics] = []
        self.latency_metrics: List[LatencyMetrics] = []
        self.operation_costs = {
            'embedding': 0.0001,  # per token
            'whisper': 0.006,     # per minute
            'cohere_chat': 0.0002, # per token
            'search': 0.00001    # per search
        }
        
    def track_cost(self, operation: str, model: str, tokens_used: int, cost_per_token: Optional[float] = None):
        """Track API costs"""
        if cost_per_token is None:
            cost_per_token = self.operation_costs.get(operation, 0.0001)
        
        total_cost = tokens_used * cost_per_token
        
        cost_metric = CostMetrics(
            operation=operation,
            model=model,
            tokens_used=tokens_used,
            cost_per_token=cost_per_token,
            total_cost=total_cost,
            timestamp=datetime.now().isoformat()
        )
        
        self.cost_metrics.append(cost_metric)
        logger.info(f"Cost tracked: {operation} - ${total_cost:.4f}")
    
    def track_latency(self, operation: str, duration_ms: float, success: bool = True):
        """Track operation latency"""
        # Calculate P95 latency for this operation type
        operation_latencies = [m.duration_ms for m in self.latency_metrics if m.operation == operation]
        operation_latencies.append(duration_ms)
        operation_latencies.sort()
        
        p95_index = int(0.95 * len(operation_latencies))
        p95_latency = operation_latencies[min(p95_index, len(operation_latencies) - 1)]
        
        latency_metric = LatencyMetrics(
            operation=operation,
            duration_ms=duration_ms,
            p95_latency=p95_latency,
            success=success,
            timestamp=datetime.now().isoformat()
        )
        
        self.latency_metrics.append(latency_metric)
        logger.info(f"Latency tracked: {operation} - {duration_ms:.2f}ms (P95: {p95_latency:.2f}ms)")
    
    def get_cost_report(self, hours: int = 24) -> Dict:
        """Generate cost report for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        recent_costs = [c for c in self.cost_metrics if c.timestamp >= cutoff_str]
        
        if not recent_costs:
            return {
                'period_hours': hours,
                'total_cost': 0.0,
                'operation_breakdown': {},
                'model_breakdown': {},
                'cost_trend': []
            }
        
        # Calculate totals
        total_cost = sum(c.total_cost for c in recent_costs)
        
        # Operation breakdown
        operation_breakdown = {}
        for cost in recent_costs:
            if cost.operation not in operation_breakdown:
                operation_breakdown[cost.operation] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            operation_breakdown[cost.operation]['cost'] += cost.total_cost
            operation_breakdown[cost.operation]['tokens'] += cost.tokens_used
            operation_breakdown[cost.operation]['calls'] += 1
        
        # Model breakdown
        model_breakdown = {}
        for cost in recent_costs:
            if cost.model not in model_breakdown:
                model_breakdown[cost.model] = {'cost': 0.0, 'tokens': 0, 'calls': 0}
            model_breakdown[cost.model]['cost'] += cost.total_cost
            model_breakdown[cost.model]['tokens'] += cost.tokens_used
            model_breakdown[cost.model]['calls'] += 1
        
        # Cost trend (hourly)
        cost_trend = []
        for i in range(hours):
            hour_start = datetime.now() - timedelta(hours=i+1)
            hour_end = datetime.now() - timedelta(hours=i)
            hour_costs = [c for c in recent_costs 
                         if hour_start.isoformat() <= c.timestamp < hour_end.isoformat()]
            hour_total = sum(c.total_cost for c in hour_costs)
            cost_trend.append({
                'hour': hour_start.strftime('%H:00'),
                'cost': hour_total
            })
        
        return {
            'period_hours': hours,
            'total_cost': total_cost,
            'operation_breakdown': operation_breakdown,
            'model_breakdown': model_breakdown,
            'cost_trend': list(reversed(cost_trend))
        }
    
    def get_latency_report(self, hours: int = 24) -> Dict:
        """Generate latency report for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        recent_latencies = [l for l in self.latency_metrics if l.timestamp >= cutoff_str]
        
        if not recent_latencies:
            return {
                'period_hours': hours,
                'operations': {},
                'overall_p95': 0.0,
                'success_rate': 0.0
            }
        
        # Operation breakdown
        operations = {}
        for latency in recent_latencies:
            if latency.operation not in operations:
                operations[latency.operation] = {
                    'avg_latency': 0.0,
                    'p95_latency': 0.0,
                    'success_rate': 0.0,
                    'total_calls': 0,
                    'successful_calls': 0
                }
            
            operations[latency.operation]['total_calls'] += 1
            if latency.success:
                operations[latency.operation]['successful_calls'] += 1
        
        # Calculate metrics for each operation
        for op, metrics in operations.items():
            op_latencies = [l.duration_ms for l in recent_latencies if l.operation == op]
            op_successes = [l.success for l in recent_latencies if l.operation == op]
            
            metrics['avg_latency'] = sum(op_latencies) / len(op_latencies)
            metrics['p95_latency'] = sorted(op_latencies)[int(0.95 * len(op_latencies))]
            metrics['success_rate'] = sum(op_successes) / len(op_successes)
        
        # Overall metrics
        all_latencies = [l.duration_ms for l in recent_latencies]
        all_successes = [l.success for l in recent_latencies]
        
        overall_p95 = sorted(all_latencies)[int(0.95 * len(all_latencies))] if all_latencies else 0
        success_rate = sum(all_successes) / len(all_successes) if all_successes else 0
        
        return {
            'period_hours': hours,
            'operations': operations,
            'overall_p95': overall_p95,
            'success_rate': success_rate
        }
    
    def get_comprehensive_report(self, hours: int = 24) -> Dict:
        """Generate comprehensive cost and latency report"""
        cost_report = self.get_cost_report(hours)
        latency_report = self.get_latency_report(hours)
        
        # Generate narrative
        narrative = self._generate_narrative(cost_report, latency_report)
        
        return {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'period_hours': hours,
            'cost_analysis': cost_report,
            'latency_analysis': latency_report,
            'narrative': narrative,
            'recommendations': self._generate_recommendations(cost_report, latency_report)
        }
    
    def _generate_narrative(self, cost_report: Dict, latency_report: Dict) -> str:
        """Generate human-readable narrative"""
        total_cost = cost_report['total_cost']
        overall_p95 = latency_report['overall_p95']
        success_rate = latency_report['success_rate']
        
        narrative = f"""
## Performance Report Summary

**Cost Analysis:**
- Total cost over {cost_report['period_hours']} hours: ${total_cost:.4f}
- Most expensive operation: {max(cost_report['operation_breakdown'].items(), key=lambda x: x[1]['cost'])[0] if cost_report['operation_breakdown'] else 'N/A'}
- Cost per hour: ${total_cost / cost_report['period_hours']:.4f}

**Latency Analysis:**
- Overall P95 latency: {overall_p95:.2f}ms
- Success rate: {success_rate:.1%}
- Performance target (≤2.5s): {'✅ PASSING' if overall_p95 <= 2500 else '❌ FAILING'}

**Key Insights:**
- System is {'performing well' if overall_p95 <= 2500 and success_rate >= 0.95 else 'experiencing issues'}
- Cost efficiency: {'Good' if total_cost < 1.0 else 'High - consider optimization'}
- Reliability: {'Excellent' if success_rate >= 0.99 else 'Good' if success_rate >= 0.95 else 'Needs improvement'}
        """
        
        return narrative.strip()
    
    def _generate_recommendations(self, cost_report: Dict, latency_report: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        total_cost = cost_report['total_cost']
        overall_p95 = latency_report['overall_p95']
        success_rate = latency_report['success_rate']
        
        if overall_p95 > 2500:
            recommendations.append("Consider optimizing search algorithms or increasing server resources")
        
        if success_rate < 0.95:
            recommendations.append("Investigate error patterns and improve error handling")
        
        if total_cost > 1.0:
            recommendations.append("Review API usage patterns and consider caching strategies")
        
        # Find most expensive operation
        if cost_report['operation_breakdown']:
            most_expensive = max(cost_report['operation_breakdown'].items(), key=lambda x: x[1]['cost'])
            recommendations.append(f"Optimize {most_expensive[0]} operation (${most_expensive[1]['cost']:.4f})")
        
        if not recommendations:
            recommendations.append("System performance is optimal - no immediate optimizations needed")
        
        return recommendations

# Global monitor instance
monitor = CostLatencyMonitor()
