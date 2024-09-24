import time
import psutil
import os

def measure_time_and_memory(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB
        result = func(*args, **kwargs)
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        return result, execution_time, memory_used
    return wrapper