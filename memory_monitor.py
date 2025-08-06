#!/usr/bin/env python3
"""
Docker Container Memory Monitor

This script monitors memory usage of the current process over time and logs it.
It can be run inside a Docker container to track memory growth.
"""

import os
import time
import psutil
import logging
import argparse
import gc
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('container_memory.log')
    ]
)

def get_process_memory():
    """Get memory usage of current process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_system_memory():
    """Get system memory usage in MB"""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / 1024 / 1024,
        'available': memory.available / 1024 / 1024,
        'used': memory.used / 1024 / 1024,
        'percent': memory.percent
    }

def force_cleanup():
    """Force garbage collection"""
    collected = gc.collect()
    logging.info(f"Garbage collector: collected {collected} objects.")
    
def monitor_memory(duration_minutes=60, interval_seconds=60, plot=False):
    """
    Monitor memory usage over time
    
    Args:
        duration_minutes: How long to monitor (in minutes)
        interval_seconds: How often to check memory (in seconds)
        plot: Whether to generate a plot of memory usage
    """
    timestamps = []
    process_memory = []
    system_memory_used = []
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    logging.info(f"Starting memory monitoring for {duration_minutes} minutes")
    logging.info(f"Initial process memory: {get_process_memory():.2f} MB")
    logging.info(f"Initial system memory: {get_system_memory()}")
    
    try:
        while time.time() < end_time:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            proc_mem = get_process_memory()
            sys_mem = get_system_memory()
            
            timestamps.append(current_time)
            process_memory.append(proc_mem)
            system_memory_used.append(sys_mem['used'])
            
            logging.info(f"Time: {current_time}")
            logging.info(f"Process memory: {proc_mem:.2f} MB")
            logging.info(f"System memory used: {sys_mem['used']:.2f} MB ({sys_mem['percent']}%)")
            
            # Wait for the next check
            time.sleep(interval_seconds)
            
        # Final memory check
        logging.info("Final memory check:")
        logging.info(f"Process memory: {get_process_memory():.2f} MB")
        logging.info(f"System memory: {get_system_memory()}")
        
        if plot and len(timestamps) > 1:
            plot_memory_usage(timestamps, process_memory, system_memory_used)
            
    except KeyboardInterrupt:
        logging.info("Monitoring interrupted by user")
    
    logging.info("Memory monitoring complete")

def plot_memory_usage(timestamps, process_memory, system_memory):
    """Generate a plot of memory usage over time"""
    plt.figure(figsize=(12, 6))
    
    # Convert string timestamps to numbers for plotting
    x = np.arange(len(timestamps))
    
    plt.plot(x, process_memory, 'b-', label='Process Memory (MB)')
    plt.plot(x, system_memory, 'r-', label='System Memory (MB)')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    
    # Use every Nth timestamp for readability
    step = max(1, len(timestamps) // 10)
    plt.xticks(x[::step], [timestamps[i] for i in range(0, len(timestamps), step)], rotation=45)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('memory_usage.png')
    logging.info("Memory usage plot saved as 'memory_usage.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor container memory usage over time")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in minutes")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--plot", action="store_true", help="Generate memory usage plot")
    args = parser.parse_args()
    
    monitor_memory(args.duration, args.interval, args.plot)
