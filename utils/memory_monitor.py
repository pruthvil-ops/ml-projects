import psutil
import os
import time

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")

def monitor_memory(interval=10):
    """Monitor memory usage at intervals"""
    while True:
        print_memory_usage()
        time.sleep(interval)