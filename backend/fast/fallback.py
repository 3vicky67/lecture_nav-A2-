"""
Rollback/Fallback System
Provides graceful degradation and recovery mechanisms
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class FallbackLevel(Enum):
    """Fallback severity levels"""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    CRITICAL = "critical"

@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    fallback_enabled: bool = True
    cache_fallback: bool = True
    degraded_mode: bool = True

class FallbackManager:
    def __init__(self):
        self.fallback_configs = {}
        self.circuit_breakers = {}
        self.cache_fallback = {}
        self.degraded_mode = False
        
    def register_fallback(self, operation: str, config: FallbackConfig):
        """Register fallback configuration for an operation"""
        self.fallback_configs[operation] = config
        
    def execute_with_fallback(self, operation: str, primary_func: Callable, 
                            fallback_func: Optional[Callable] = None, 
                            cache_key: Optional[str] = None) -> Any:
        """
        Execute operation with fallback mechanisms
        
        Args:
            operation: Operation name
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            cache_key: Cache key for fallback data
        """
        config = self.fallback_configs.get(operation, FallbackConfig())
        
        if not config.fallback_enabled:
            return primary_func()
        
        # Check circuit breaker
        if self._is_circuit_open(operation):
            logger.warning(f"Circuit breaker open for {operation}, using fallback")
            return self._execute_fallback(operation, fallback_func, cache_key)
        
        # Try primary function with retries
        for attempt in range(config.max_retries):
            try:
                start_time = time.time()
                result = primary_func()
                
                # Record success
                self._record_success(operation, time.time() - start_time)
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {operation}: {e}")
                
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    # All retries failed, use fallback
                    logger.error(f"All retries failed for {operation}, using fallback")
                    self._record_failure(operation)
                    return self._execute_fallback(operation, fallback_func, cache_key)
    
    def _execute_fallback(self, operation: str, fallback_func: Optional[Callable], 
                         cache_key: Optional[str]) -> Any:
        """Execute fallback function or return cached result"""
        
        # Try fallback function first
        if fallback_func:
            try:
                return fallback_func()
            except Exception as e:
                logger.error(f"Fallback function failed for {operation}: {e}")
        
        # Try cache fallback
        if cache_key and cache_key in self.cache_fallback:
            logger.info(f"Using cached fallback for {operation}")
            return self.cache_fallback[cache_key]
        
        # Return degraded response
        return self._get_degraded_response(operation)
    
    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for operation"""
        if operation not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation]
        if breaker['failures'] >= 5:  # Open after 5 failures
            if time.time() - breaker['last_failure'] < 60:  # Stay open for 60 seconds
                return True
            else:
                # Reset circuit breaker
                breaker['failures'] = 0
                breaker['last_failure'] = 0
        
        return False
    
    def _record_success(self, operation: str, duration: float):
        """Record successful operation"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'last_failure': 0}
        
        # Reset failure count on success
        self.circuit_breakers[operation]['failures'] = 0
    
    def _record_failure(self, operation: str):
        """Record failed operation"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'last_failure': 0}
        
        breaker = self.circuit_breakers[operation]
        breaker['failures'] += 1
        breaker['last_failure'] = time.time()
    
    def _get_degraded_response(self, operation: str) -> Dict:
        """Get degraded response when all fallbacks fail"""
        self.degraded_mode = True
        
        degraded_responses = {
            'search': {
                'results': [],
                'message': 'Search temporarily unavailable',
                'degraded': True
            },
            'ai_chat': {
                'answer': 'AI service temporarily unavailable. Please try again later.',
                'degraded': True
            },
            'embedding': {
                'embedding': None,
                'message': 'Embedding service temporarily unavailable',
                'degraded': True
            }
        }
        
        return degraded_responses.get(operation, {
            'error': 'Service temporarily unavailable',
            'degraded': True
        })
    
    def cache_result(self, key: str, result: Any, ttl: int = 3600):
        """Cache result for fallback use"""
        self.cache_fallback[key] = {
            'result': result,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if still valid"""
        if key not in self.cache_fallback:
            return None
        
        cached = self.cache_fallback[key]
        if time.time() - cached['timestamp'] > cached['ttl']:
            del self.cache_fallback[key]
            return None
        
        return cached['result']
    
    def get_system_status(self) -> Dict:
        """Get current system status including fallback states"""
        status = {
            'degraded_mode': self.degraded_mode,
            'circuit_breakers': {},
            'cache_size': len(self.cache_fallback),
            'fallback_configs': len(self.fallback_configs)
        }
        
        for operation, breaker in self.circuit_breakers.items():
            status['circuit_breakers'][operation] = {
                'failures': breaker['failures'],
                'is_open': self._is_circuit_open(operation)
            }
        
        return status
    
    def reset_circuit_breaker(self, operation: str):
        """Manually reset circuit breaker for operation"""
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'last_failure': 0}
            logger.info(f"Circuit breaker reset for {operation}")
    
    def enable_degraded_mode(self):
        """Enable degraded mode for all operations"""
        self.degraded_mode = True
        logger.warning("Degraded mode enabled")
    
    def disable_degraded_mode(self):
        """Disable degraded mode"""
        self.degraded_mode = False
        logger.info("Degraded mode disabled")

# Global fallback manager
fallback_manager = FallbackManager()

# Register default fallback configurations
fallback_manager.register_fallback('search', FallbackConfig(max_retries=2, timeout=10.0))
fallback_manager.register_fallback('ai_chat', FallbackConfig(max_retries=1, timeout=15.0))
fallback_manager.register_fallback('embedding', FallbackConfig(max_retries=3, timeout=5.0))
