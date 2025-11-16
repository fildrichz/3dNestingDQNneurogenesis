"""
Simple test script to verify multiprocessing works on your system.

Run this before using the GA evolution with parallel=True to ensure
multiprocessing is functioning correctly on Windows.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def simple_worker(task_id, sleep_time):
    """Simple worker function that just sleeps and returns."""
    import time
    print(f"Worker {task_id} started")
    time.sleep(sleep_time)
    print(f"Worker {task_id} completed after {sleep_time}s")
    return task_id, sleep_time


def test_multiprocessing(num_workers=4, tasks=8):
    """Test basic multiprocessing functionality."""
    print("="*60)
    print("MULTIPROCESSING TEST")
    print("="*60)
    print(f"Testing with {num_workers} workers and {tasks} tasks")
    print()

    start_time = time.time()
    results = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks
            futures = {}
            for i in range(tasks):
                sleep_time = 1  # Each task sleeps for 1 second
                future = executor.submit(simple_worker, i, sleep_time)
                futures[future] = i

            # Collect results with timeout
            timeout = 30  # 30 seconds total timeout
            print(f"Waiting for results (timeout: {timeout}s)...")
            print()

            for future in as_completed(futures, timeout=timeout):
                task_id = futures[future]
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                    print(f"✓ Task {result[0]} completed successfully")
                except Exception as e:
                    print(f"✗ Task {task_id} failed: {e}")

    except Exception as e:
        print(f"\n✗ MULTIPROCESSING TEST FAILED: {e}")
        print("\nThis means multiprocessing is not working correctly on your system.")
        print("Recommendation: Use parallel=False in ga_config")
        return False

    elapsed = time.time() - start_time

    print()
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total tasks: {tasks}")
    print(f"Completed: {len(results)}")
    print(f"Failed: {tasks - len(results)}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print()

    if len(results) == tasks:
        speedup = (tasks * 1.0) / elapsed
        print(f"✓ ALL TESTS PASSED")
        print(f"  Expected sequential time: {tasks}s")
        print(f"  Actual parallel time: {elapsed:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print()
        print("Multiprocessing is working correctly on your system!")
        return True
    else:
        print(f"✗ SOME TESTS FAILED")
        print(f"  Only {len(results)}/{tasks} tasks completed")
        print()
        print("Multiprocessing may have issues on your system.")
        print("Recommendation: Use fewer workers or parallel=False")
        return False


if __name__ == "__main__":
    # Test with 4 workers
    success = test_multiprocessing(num_workers=4, tasks=8)

    if not success:
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("If the test failed, try:")
        print("1. Run without profiler: python test_multiprocessing.py")
        print("2. Reduce workers: test_multiprocessing(num_workers=2)")
        print("3. Use parallel=False in your GA config")
        print("4. Check Windows firewall/antivirus settings")
        print("="*60)
