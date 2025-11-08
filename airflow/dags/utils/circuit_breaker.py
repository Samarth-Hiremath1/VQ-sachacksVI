"""
Circuit Breaker Pattern Implementation

Implements circuit breaker for ML service failures with graceful degradation.
Requirements: 2.7, 3.5, 6.4
"""

import logging
import time
from typing import Callable, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "CircuitBreaker"
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
        logger.info(f"Initialized {name} circuit breaker (threshold={failure_threshold}, timeout={recovery_timeout}s)")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception if function fails
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"{self.name}: Attempting recovery (HALF_OPEN)")
                    self.state = CircuitState.HALF_OPEN
                else:
                    logger.warning(f"{self.name}: Circuit is OPEN, rejecting request")
                    raise CircuitBreakerOpenError(
                        f"{self.name} circuit breaker is OPEN. "
                        f"Service unavailable. Retry after {self.recovery_timeout}s"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"{self.name}: Recovery successful, closing circuit")
                self.state = CircuitState.CLOSED
            
            self.failure_count = 0
            self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            logger.warning(
                f"{self.name}: Failure {self.failure_count}/{self.failure_threshold}"
            )
            
            if self.failure_count >= self.failure_threshold:
                logger.error(f"{self.name}: Opening circuit breaker")
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"{self.name}: Recovery failed, reopening circuit")
                self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            logger.info(f"{self.name}: Manual reset")
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitState.CLOSED
    
    def get_state(self) -> str:
        """Get current circuit state"""
        return self.state.value
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'recovery_timeout': self.recovery_timeout
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class MLServiceCircuitBreakers:
    """Manages circuit breakers for ML services"""
    
    def __init__(self):
        self.pytorch_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            name="PyTorch-BodyLanguage"
        )
        
        self.tensorflow_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            name="TensorFlow-Speech"
        )
        
        logger.info("Initialized ML service circuit breakers")
    
    def call_pytorch_service(self, func: Callable, *args, **kwargs) -> Any:
        """Call PyTorch service with circuit breaker protection"""
        return self.pytorch_breaker.call(func, *args, **kwargs)
    
    def call_tensorflow_service(self, func: Callable, *args, **kwargs) -> Any:
        """Call TensorFlow service with circuit breaker protection"""
        return self.tensorflow_breaker.call(func, *args, **kwargs)
    
    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers"""
        return {
            'pytorch': self.pytorch_breaker.get_stats(),
            'tensorflow': self.tensorflow_breaker.get_stats()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        self.pytorch_breaker.reset()
        self.tensorflow_breaker.reset()
        logger.info("Reset all ML service circuit breakers")
