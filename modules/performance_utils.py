import time
import psutil
import os
import threading

def measure_time_and_memory(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB
        max_memory = start_memory
        stop_thread = False

        def memory_monitor():
            nonlocal max_memory
            while not stop_thread:
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                time.sleep(0.1)  # Check every 100ms

        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()

        try:
            result = func(*args, **kwargs)
        finally:
            stop_thread = True
            monitor_thread.join()

        end_time = time.time()
        execution_time = end_time - start_time
        memory_used = max_memory - start_memory

        return result, execution_time, memory_used

    return wrapper