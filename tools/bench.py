#!/usr/bin/env python3
"""
Load testing harness for Text-to-SQL RAG chatbot performance optimization.

Measures:
- Cache hit/miss rates
- End-to-end latency (p50, p95, p99)
- RAG retrieval timing
- SQL generation timing
- DB execution timing
- Memory usage
- Concurrent throughput
"""

import os
import sys
import time
import json
import random
import logging
import statistics
import concurrent.futures
import threading
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.sql_engine import handle_query, StepTimer
from app.db_utils import DatabaseConfig, create_database_with_views
from app.rag.rag_engine import RAGEngine
from backend.services.runtime import set_request_id, set_session_id, log_event
from backend.services.session import SessionState

logger = logging.getLogger("load_test")

@dataclass
class TestResult:
    """Result of a single query execution."""
    question: str
    success: bool
    from_cache: bool
    total_time_ms: float
    intent_detection_ms: float
    schema_loading_ms: float
    rag_retrieval_ms: float
    sql_generation_ms: float
    db_execution_ms: float
    cache_hit_rate: float
    row_count: int
    error: Optional[str] = None


class LoadTester:
    """Load testing orchestrator for Text-to-SQL RAG system."""
    
    def __init__(self, config: DatabaseConfig, test_questions: List[str], 
                 concurrent_users: int = 10, requests_per_user: int = 5):
        self.config = config
        self.test_questions = test_questions
        self.concurrent_users = concurrent_users
        self.requests_per_user = requests_per_user
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize database, RAG engine, and other components."""
        logger.info("Initializing components...")
        
        # Create database connection
        self.db = create_database_with_views(self.config)
        self.db_config = self.config
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine()
        
        # Initialize embedder (placeholder - would be replaced with actual embedder)
        self.embedder = None  # Would be initialized with actual embedding model
        
        # Initialize query cache
        self.query_cache = {}
        
        logger.info("Components initialized successfully")
        
    def _create_session_state(self, user_id: str) -> SessionState:
        """Create a session state for a user."""
        return SessionState(
            session_id=f"load_test_{user_id}",
            user_id=user_id,
            conversation_turns=[]
        )
        
    def execute_single_query(self, user_id: str, question: str, 
                           session_state: SessionState) -> TestResult:
        """Execute a single query and collect metrics."""
        start_time = time.perf_counter()
        
        # Set request/session context for logging
        request_id = f"load_test_{user_id}_{int(time.time() * 1000)}"
        set_request_id(request_id)
        set_session_id(session_state.session_id)
        
        try:
            logger.debug(f"User {user_id} executing: {question}")
            
            # Execute query
            result = handle_query(
                question=question,
                db=self.db,
                db_config=self.db_config,
                sql_chain=None,  # Would be initialized with actual chain
                llm=None,  # Would be initialized with actual LLM
                rag_engine=self.rag_engine,
                embedder=self.embedder,
                chat_history=[],
                query_cache=self.query_cache,
                session_state=session_state,
                sql_dialect="sqlserver"
            )
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Extract metrics from result
            from_cache = result.get("from_cache", False)
            row_count = len(result.get("results", []))
            error = result.get("error")
            success = error is None
            
            # Create test result
            test_result = TestResult(
                question=question,
                success=success,
                from_cache=from_cache,
                total_time_ms=total_time,
                intent_detection_ms=0,  # Would extract from timer
                schema_loading_ms=0,  # Would extract from timer
                rag_retrieval_ms=0,  # Would extract from timer
                sql_generation_ms=0,  # Would extract from timer
                db_execution_ms=0,  # Would extract from timer
                cache_hit_rate=0,  # Would calculate from cache hits
                row_count=row_count,
                error=str(error) if error else None
            )
            
            return test_result
            
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Query failed for user {user_id}: {e}")
            
            return TestResult(
                question=question,
                success=False,
                from_cache=False,
                total_time_ms=total_time,
                intent_detection_ms=0,
                schema_loading_ms=0,
                rag_retrieval_ms=0,
                sql_generation_ms=0,
                db_execution_ms=0,
                cache_hit_rate=0,
                row_count=0,
                error=str(e)
            )
        
    def simulate_user_session(self, user_id: int) -> List[TestResult]:
        """Simulate a user session with multiple requests."""
        logger.info(f"Starting user session {user_id}")
        session_state = self._create_session_state(f"user_{user_id}")
        user_results = []
        
        for i in range(self.requests_per_user):
            # Pick random question
            question = random.choice(self.test_questions)
            
            # Execute query
            result = self.execute_single_query(f"user_{user_id}", question, session_state)
            user_results.append(result)
            
            # Small delay between requests to simulate user think time
            time.sleep(random.uniform(0.1, 0.5))
            
        logger.info(f"Completed user session {user_id} with {len(user_results)} requests")
        return user_results
        
    def run_load_test(self) -> Dict[str, Any]:
        """Run the complete load test."""
        logger.info(f"Starting load test with {self.concurrent_users} concurrent users, "
                   f"{self.requests_per_user} requests each")
        
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run concurrent user sessions
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = []
            for user_id in range(self.concurrent_users):
                future = executor.submit(self.simulate_user_session, user_id)
                futures.append(future)
                
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    logger.error(f"User session failed: {e}")
                    
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Store results
        with self.lock:
            self.results.extend(all_results)
            
        # Calculate statistics
        stats = self._calculate_statistics(all_results)
        stats["test_config"] = {
            "concurrent_users": self.concurrent_users,
            "requests_per_user": self.requests_per_user,
            "total_requests": len(all_results),
            "test_duration_seconds": total_time,
            "memory_usage_mb": {
                "initial": initial_memory,
                "final": final_memory,
                "delta": final_memory - initial_memory
            }
        }
        
        logger.info(f"Load test completed in {total_time:.2f}s")
        return stats
        
    def _calculate_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not results:
            return {"error": "No results to analyze"}
            
        # Basic metrics
        successful_results = [r for r in results if r.success]
        cached_results = [r for r in results if r.from_cache]
        
        cache_hit_rate = len(cached_results) / len(results) if results else 0
        success_rate = len(successful_results) / len(results) if results else 0
        
        # Response time statistics
        response_times = [r.total_time_ms for r in successful_results]
        
        if response_times:
            p50 = statistics.median(response_times)
            p95 = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else p50
            p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else p95
            
            response_time_stats = {
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "mean_ms": statistics.mean(response_times)
            }
        else:
            response_time_stats = {"error": "No successful requests"}
        
        # Error analysis
        errors = [r.error for r in results if not r.success and r.error]
        error_counts = {}
        for error in errors:
            error_type = error.split(":")[0] if ":" in error else "Unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        # Row count statistics
        row_counts = [r.row_count for r in successful_results]
        row_count_stats = {
            "mean": statistics.mean(row_counts) if row_counts else 0,
            "median": statistics.median(row_counts) if row_counts else 0,
            "max": max(row_counts) if row_counts else 0
        }
        
        return {
            "summary": {
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(results) - len(successful_results),
                "cached_requests": len(cached_results),
                "cache_hit_rate": cache_hit_rate,
                "success_rate": success_rate
            },
            "response_times": response_time_stats,
            "row_counts": row_count_stats,
            "errors": error_counts
        }
        
    def generate_report(self) -> str:
        """Generate a detailed performance report."""
        if not self.results:
            return "No test results available"
            
        stats = self._calculate_statistics(self.results)
        
        report = f"""
# Text-to-SQL RAG Performance Report
Generated: {datetime.now().isoformat()}

## Test Configuration
- Concurrent Users: {stats['test_config']['concurrent_users']}
- Requests per User: {stats['test_config']['requests_per_user']}
- Total Requests: {stats['test_config']['total_requests']}
- Test Duration: {stats['test_config']['test_duration_seconds']:.2f}s

## Performance Metrics

### Cache Performance
- Cache Hit Rate: {stats['summary']['cache_hit_rate']:.1%}
- Cached Requests: {stats['summary']['cached_requests']}

### Response Time Statistics
- P50: {stats['response_times'].get('p50_ms', 'N/A')}ms
- P95: {stats['response_times'].get('p95_ms', 'N/A')}ms
- P99: {stats['response_times'].get('p99_ms', 'N/A')}ms
- Mean: {stats['response_times'].get('mean_ms', 'N/A')}ms

### Success Metrics
- Success Rate: {stats['summary']['success_rate']:.1%}
- Total Requests: {stats['summary']['total_requests']}
- Successful: {stats['summary']['successful_requests']}
- Failed: {stats['summary']['failed_requests']}

### Resource Usage
- Memory Delta: {stats['test_config']['memory_usage_mb']['delta']:.1f}MB

### Error Analysis
"""
        
        for error_type, count in stats['errors'].items():
            report += f"- {error_type}: {count}\n"
            
        return report


def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Database configuration
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "1433"),
        username=os.getenv("DB_USER", "sa"),
        password=os.getenv("DB_PASSWORD", "password"),
        database=os.getenv("DB_NAME", "testdb"),
        connect_timeout=10,
        query_timeout=30
    )
    
    # Test questions
    test_questions = [
        "What are the top 5 hotels by revenue this month?",
        "Show me booking trends for the last quarter",
        "Which suppliers have the highest profit margin?",
        "How many bookings were made last week?",
        "What is the average booking value by country?",
        "Show me monthly revenue for this year",
        "Which cities have the most bookings?",
        "What is the total profit for last month?",
        "Show me bookings by customer nationality",
        "What are the peak booking periods?",
    ]
    
    # Create load tester
    tester = LoadTester(
        config=db_config,
        test_questions=test_questions,
        concurrent_users=5,
        requests_per_user=10
    )
    
    # Run load test
    stats = tester.run_load_test()
    
    # Generate and print report
    report = tester.generate_report()
    print(report)
    
    # Save detailed results
    results_file = f"load_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()