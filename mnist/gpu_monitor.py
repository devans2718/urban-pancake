"""
GPU Monitoring Utility
Monitor GPU usage, memory, and temperature during training
"""

import torch
import time
import subprocess
from datetime import datetime


def get_gpu_info():
    """
    Get detailed GPU information
    """
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    
    # Basic info
    print(f"\nCUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Info for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        
        print(f"\nMemory Info:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Free: {total_memory - allocated:.2f} GB")


def monitor_gpu_realtime(duration=60, interval=1):
    """
    Monitor GPU usage in real-time
    
    Args:
        duration: How long to monitor (seconds)
        interval: Update interval (seconds)
    """
    if not torch.cuda.is_available():
        print("No GPU available for monitoring")
        return
    
    print("\n" + "=" * 60)
    print("REAL-TIME GPU MONITORING")
    print("=" * 60)
    print(f"Duration: {duration}s | Interval: {interval}s")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Clear line
            print("\r" + " " * 100 + "\r", end="")
            
            # Get current stats
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            utilization = get_gpu_utilization()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] "
                  f"Memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved | "
                  f"Utilization: {utilization}%",
                  end="", flush=True)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    
    print("\n")


def get_gpu_utilization():
    """
    Get GPU utilization percentage using nvidia-smi
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return "N/A"


def memory_summary():
    """
    Print detailed memory summary
    """
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    print("\n" + "=" * 60)
    print("MEMORY SUMMARY")
    print("=" * 60 + "\n")
    
    print(torch.cuda.memory_summary(device=0, abbreviated=False))


def test_gpu_speed():
    """
    Quick benchmark to test GPU performance
    """
    if not torch.cuda.is_available():
        print("No GPU available for testing")
        return
    
    print("\n" + "=" * 60)
    print("GPU SPEED TEST")
    print("=" * 60)
    
    device = torch.device('cuda')
    
    # Test different matrix sizes
    sizes = [1000, 2000, 5000, 10000]
    
    print(f"\n{'Matrix Size':<15} {'GPU Time (ms)':<20} {'Operations/sec':<20}")
    print("-" * 60)
    
    for size in sizes:
        # Create random matrices
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 10 * 1000  # Convert to ms
        ops_per_sec = 1000 / avg_time
        
        print(f"{size}x{size:<10} {avg_time:<20.2f} {ops_per_sec:<20.2f}")
    
    print()


def check_cuda_errors():
    """
    Check for CUDA errors and display last error if any
    """
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    print("\n" + "=" * 60)
    print("CUDA ERROR CHECK")
    print("=" * 60)
    
    try:
        # Try a simple operation
        x = torch.randn(100, 100, device='cuda')
        y = x @ x
        torch.cuda.synchronize()
        print("\n✓ No CUDA errors detected")
    except Exception as e:
        print(f"\n✗ CUDA Error: {e}")
    
    print()


def main():
    """
    Main monitoring function - run all diagnostics
    """
    print("\n" + "=" * 60)
    print("GPU DIAGNOSTICS AND MONITORING")
    print("=" * 60)
    
    # 1. Get GPU info
    get_gpu_info()
    
    # 2. Check for errors
    check_cuda_errors()
    
    # 3. Speed test
    if torch.cuda.is_available():
        test_gpu_speed()
    
    # 4. Offer real-time monitoring
    if torch.cuda.is_available():
        response = input("\nStart real-time monitoring? (y/n): ")
        if response.lower() == 'y':
            duration = input("Duration in seconds (default 60): ")
            duration = int(duration) if duration else 60
            monitor_gpu_realtime(duration=duration)


if __name__ == '__main__':
    main()
