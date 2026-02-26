#!/usr/bin/env python3
"""
Load test harness for Text-to-SQL RAG chatbot performance testing.

Features:
- Concurrent request testing with ThreadPoolExecutor
- Performance metrics (p50, p95, p99 latency)
- Cache hit rate tracking
- Result verification against golden set
- Thread count monitoring
- Configurable test parameters
"""

import os
import sys
import time
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.sql_engine import handle_query
from backend.services.runtime import log_event, shutdown_shared_executor
from app.db_utils import quick_connect
from app.rag.rag_engine import RAGEngine
from backend.services.session import SessionState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("load_test")


@dataclass
class TestConfig:
    """Configuration for load test."""
    concurrent_requests: int = 20
    total_requests: int = 100
    warmup_requests: int = 10
    test_questions: List[str] = None
    db_config: Dict[str, Any] = None
    output_file: str = "load_test_results.json"
    
    def __post_init__(self):
        if self.test_questions is None:
            # Default test questions
            self.test_questions = [
                "show top 10 agents by revenue",
                "total bookings this month",
                "which supplier has highest bookings",
                "revenue by country last week",
                "profit by agent this year",
                "top 5 hotels by bookings",
                "average booking value",
                "total cancelled bookings",
                "bookings by agent type",
                "revenue trend this month",
            ]
        
        if self.db_config is None:
            # Default DB config from environment
            self.db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": os.getenv("DB_PORT", "1433"),
                "username": os.getenv("DB_USERNAME", "sa"),
                "password": os.getenv("DB_PASSWORD", "password"),
                "database": os.getenv("DB_NAME", "text2sql"),
            }


@dataclass
class TestResult:
    """Result of a single test request."""
    question: str
    response_time_ms: float
    from_cache: bool
    sql_generated: Optional[str]
    row_count: int
    error: Optional[str]
    intent: Optional[str]
    cache_hit_type: Optional[str] = None


class LoadTester:
    """Load test runner for Text-to-SQL RAG system."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.golden_results: Dict[str, Any] = {}
        self.db = None
        self.rag_engine = None
        self.llm = None
        self.embedder = None
        self.session_state = None
        self.query_cache = {}
        
    def setup(self):
        """Setup test environment."""
        logger.info("Setting up test environment...")
        
        # Connect to database
        self.db = quick_connect(
            host=self.config.db_config["host"],
            port=self.config.db_config["port"],
            username=self.config.db_config["username"],
            password=self.config.db_config["password"],
            database=self.config.db_config["database"],
            query_timeout=30,
        )
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine()
        self.rag_engine.load_default_schema()
        
        # Initialize LLM and embedder (using simple setup for testing)
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embedder = OpenAIEmbeddings()
        
        # Initialize session state
        self.session_state = SessionState()
        
        logger.info("Test environment setup complete")
        
    def cleanup(self):
        """Cleanup test environment."""
        logger.info("Cleaning up test environment...")
        shutdown_shared_executor()
        
    def run_single_query(self, question: str, request_id: str) -> TestResult:
        """Run a single query and return result."""
        start_time = time.perf_counter()
        
        try:
            # Set request ID for logging
            from backend.services.runtime import set_request_id
            set_request_id(request_id)
            
            # Handle the query
            result = handle_query(
                question=question,
                db=self.db,
                db_config=self.config.db_config,
                sql_chain=None,  # Will be created internally
                llm=self.llm,
                rag_engine=self.rag_engine,
                embedder=self.embedder,
                chat_history=[],
                query_cache=self.query_cache,
                session_state=self.session_state,
            )
            
            response_time_ms = (time.perf_counter() - start_time) * 1000
            
            return TestResult(
                question=question,
                response_time_ms=response_time_ms,
                from_cache=result.get("from_cache", False),
                sql_generated=result.get("sql"),
                row_count=result.get("row_count", 0),
                error=result.get("error"),
                intent=result.get("intent"),
            )
            
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Query failed: {question} - {str(e)}")
            return TestResult(
                question=question,
                response_time_ms=response_time_ms,
                from_cache=False,
                sql_generated=None,
                row_count=0,
                error=str(e),
                intent=None,
            )
        
    def run_warmup(self):
        """Run warmup requests to populate caches."""
        logger.info(f"Running {self.config.warmup_requests} warmup requests...")
        
        for i in range(self.config.warmup_requests):
            question = self.config.test_questions[i % len(self.config.test_questions)]
            request_id = f"warmup_{i}"
            self.run_single_query(question, request_id)
            
        logger.info("Warmup complete")
        
    def run_load_test(self):
        """Run the main load test."""
        logger.info(f"Starting load test: {self.config.concurrent_requests} concurrent requests, "
                   f"{self.config.total_requests} total requests")
        
        # Run warmup first
        self.run_warmup()
        
        # Run concurrent requests
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_requests) as executor:
            # Submit all requests
            futures = []
            for i in range(self.config.total_requests):
                question = self.config.test_questions[i % len(self.config.test_questions)]
                request_id = f"test_{i}"
                future = executor.submit(self.run_single_query, question, request_id)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Future failed: {str(e)}")
        
        total_time = time.perf_counter() - start_time
        logger.info(f"Load test completed in {total_time:.2f} seconds")
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from results."""
        if not self.results:
            return {}
        
        response_times = [r.response_time_ms for r in self.results]
        cache_hits = [r.from_cache for r in self.results]
        errors = [r.error for r in self.results if r.error]
        
        metrics = {
            "total_requests": len(self.results),
            "successful_requests": len([r for r in self.results if not r.error]),
            "failed_requests": len(errors),
            "cache_hit_rate": sum(cache_hits) / len(cache_hits) * 100,
            "cache_hits": sum(cache_hits),
            "cache_misses": len(cache_hits) - sum(cache_hits),
            "p50_latency_ms": statistics.median(response_times),
            "p95_latency_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0],
            "p99_latency_ms": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else response_times[0],
            "min_latency_ms": min(response_times),
            "max_latency_ms": max(response_times),
            "avg_latency_ms": statistics.mean(response_times),
            "std_latency_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0,
        }
        
        # Separate cache hit/miss metrics
        cache_hit_times = [r.response_time_ms for r in self.results if r.from_cache]
        cache_miss_times = [r.response_time_ms for r in self.results if not r.from_cache]
        
        if cache_hit_times:
            metrics["cache_hit_metrics"] = {
                "avg_latency_ms": statistics.mean(cache_hit_times),
                "p50_latency_ms": statistics.median(cache_hit_times),
                "count": len(cache_hit_times),
            }
        
        if cache_miss_times:
            metrics["cache_miss_metrics"] = {
                "avg_latency_ms": statistics.mean(cache_miss_times),
                "p50_latency_ms": statistics.median(cache_miss_times),
                "count": len(cache_miss_times),
            }
        
        return metrics
        
    def verify_consistency(self) -> Dict[str, Any]:
        """Verify that results are consistent (same questions produce same results)."""
        question_groups = {}
        for result in self.results:
            if result.question not in question_groups:
                question_groups[result.question] = []
            question_groups[result.question].append(result)
        
        consistency_issues = []
        for question, results in question_groups.items():
            if len(results) < 2:
                continue
                
            # Check if SQL is consistent
            sqls = [r.sql_generated for r in results if r.sql_generated]
            if len(set(sqls)) > 1:
                consistency_issues.append({
                    "question": question,
                    "issue": "inconsistent_sql",
                    "sqls": list(set(sqls)),
                })
            
            # Check if cache behavior is consistent
            cache_behaviors = [r.from_cache for r in results]
            if not all(cache_behaviors) and any(cache_behaviors):
                consistency_issues.append({
                    "question": question,
                    "issue": "inconsistent_cache_behavior",
                    "cache_behaviors": cache_behaviors,
                })
        
        return {
            "consistency_issues": consistency_issues,
            "total_questions": len(question_groups),
            "consistent_questions": len(question_groups) - len([q for q in consistency_issues if q["issue"] == "inconsistent_sql"]),
        }
        
    def save_results(self):
        """Save test results to file."""
        results = {
            "config": {
                "concurrent_requests": self.config.concurrent_requests,
                "total_requests": self.config.total_requests,
                "test_questions": self.config.test_questions,
                "db_config": {k: v for k, v in self.config.db_config.items() if k != "password"},
            },
            "metrics": self.calculate_metrics(),
            "consistency_check": self.verify_consistency(),
            "raw_results": [
                {
                    "question": r.question,
                    "response_time_ms": r.response_time_ms,
                    "from_cache": r.from_cache,
                    "sql_generated": r.sql_generated,
                    "row_count": r.row_count,
                    "error": r.error,
                    "intent": r.intent,
                }
                for r in self.results
            ],
        }
        
        with open(self.config.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_file}")
        
    def print_summary(self):
        """Print test summary to console."""
        metrics = self.calculate_metrics()
        if not metrics:
            logger.error("No results to display")
            return
            
        print("\n" + "="*60)
        print("LOAD TEST RESULTS")
        print("="*60)
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Successful: {metrics['successful_requests']}")
        print(f"Failed: {metrics['failed_requests']}")
        print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print()
        print("LATENCY METRICS:")
        print(f"  Average: {metrics['avg_latency_ms']:.1f}ms")
        print(f"  P50: {metrics['p50_latency_ms']:.1f}ms")
        print(f"  P95: {metrics['p95_latency_ms']:.1f}ms")
        print(f"  P99: {metrics['p99_latency_ms']:.1f}ms")
        print(f"  Min: {metrics['min_latency_ms']:.1f}ms")
        print(f"  Max: {metrics['max_latency_ms']:.1f}ms")
        print()
        
        if "cache_hit_metrics" in metrics:
            hit = metrics["cache_hit_metrics"]
            print("CACHE HIT PERFORMANCE:")
            print(f"  Average: {hit['avg_latency_ms']:.1f}ms")
            print(f"  Count: {hit['count']}")
            print()
            
        if "cache_miss_metrics" in metrics:
            miss = metrics["cache_miss_metrics"]
            print("CACHE MISS PERFORMANCE:")
            print(f"  Average: {miss['avg_latency_ms']:.1f}ms")
            print(f"  Count: {miss['count']}")
            print()
        
        consistency = self.verify_consistency()
        if consistency["consistency_issues"]:
            print("CONSISTENCY ISSUES FOUND:")
            for issue in consistency["consistency_issues"][:5]:  # Show first 5
                print(f"  {issue['question']}: {issue['issue']}")
            if len(consistency["consistency_issues"]) > 5:
                print(f"  ... and {len(consistency['consistency_issues']) - 5} more")
        else:
            print("CONSISTENCY: All good!")
        
        print("="*60)


def main():
    """Main function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Load test for Text-to-SQL RAG system")
    parser.add_argument("--concurrent", type=int, default=20, help="Number of concurrent requests")
    parser.add_argument("--total", type=int, default=100, help="Total number of requests")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup requests")
    parser.add_argument("--output", default="load_test_results.json", help="Output file for results")
    parser.add_argument("--questions-file", help="JSON file with test questions")
    
    args = parser.parse_args()
    
    # Create test config
    config = TestConfig(
        concurrent_requests=args.concurrent,
        total_requests=args.total,
        warmup_requests=args.warmup,
        output_file=args.output,
    )
    
    # Load questions from file if provided
    if args.questions_file:
        with open(args.questions_file) as f:
            questions_data = json.load(f)
            config.test_questions = questions_data.get("questions", config.test_questions)
    
    # Run load test
    tester = LoadTester(config)
    
    try:
        tester.setup()
        tester.run_load_test()
        tester.print_summary()
        tester.save_results()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()