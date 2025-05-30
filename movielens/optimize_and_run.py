#!/usr/bin/env python3
"""
Auto-optimization script that detects system capabilities and runs with optimal settings.
"""

import os
import sys
import psutil
import subprocess
import platform
from pathlib import Path
import multiprocessing as mp

def get_system_info():
    """Get detailed system information."""
    return {
        'cpu_count': mp.cpu_count(),
        'cpu_physical': psutil.cpu_count(logical=False),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'ram_available_gb': psutil.virtual_memory().available / (1024**3),
        'platform': platform.system(),
        'python_version': sys.version,
        'has_ssd': check_ssd(),
        'has_gpu': check_gpu(),
        'numa_nodes': get_numa_nodes()
    }

def check_ssd():
    """Check if data directory is on SSD."""
    try:
        data_path = Path("data").resolve()
        if platform.system() == "Linux":
            # Check if rotational (0 = SSD, 1 = HDD)
            device = subprocess.check_output(
                f"df {data_path} | tail -1 | awk '{{print $1}}'", 
                shell=True, text=True
            ).strip()
            if '/dev/' in device:
                base_device = device.replace('/dev/', '').rstrip('0123456789')
                rotational = Path(f"/sys/block/{base_device}/queue/rotational").read_text().strip()
                return rotational == "0"
    except:
        pass
    return False  # Conservative assumption

def check_gpu():
    """Check if CUDA-capable GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        return result.returncode == 0
    except:
        return False

def get_numa_nodes():
    """Get number of NUMA nodes."""
    try:
        if platform.system() == "Linux":
            result = subprocess.check_output("lscpu | grep 'NUMA node(s):' | awk '{print $3}'", 
                                           shell=True, text=True)
            return int(result.strip())
    except:
        pass
    return 1

def set_optimal_environment(system_info):
    """Set optimal environment variables based on system."""
    # CPU optimization
    n_threads = str(system_info['cpu_count'])
    os.environ['OMP_NUM_THREADS'] = n_threads
    os.environ['NUMBA_NUM_THREADS'] = n_threads
    os.environ['MKL_NUM_THREADS'] = n_threads
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads
    os.environ['VECLIB_MAXIMUM_THREADS'] = n_threads
    
    # Memory optimization for large RAM
    if system_info['ram_gb'] >= 32:
        os.environ['MALLOC_TRIM_THRESHOLD_'] = '-1'
        os.environ['MALLOC_MMAP_MAX_'] = '1000000'
        os.environ['MALLOC_ARENA_MAX'] = '8'
    
    # Python optimizations
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONOPTIMIZE'] = '1'
    
    # Numba cache
    os.environ['NUMBA_CACHE_DIR'] = '.numba_cache'
    
    print(f"✓ Environment optimized for {system_info['cpu_count']} cores and {system_info['ram_gb']:.1f}GB RAM")

def get_optimal_command(system_info):
    """Determine optimal command based on system capabilities."""
    ram_gb = system_info['ram_gb']
    cpu_count = system_info['cpu_count']
    
    # Determine mode based on RAM
    if ram_gb >= 32:
        mode = "ultrafast"
        extra_args = []
        
        if system_info['has_gpu']:
            extra_args.append("--gpu")
            
        if system_info['has_ssd']:
            extra_args.append("--cache-dir .cache_ssd")
            
        return f"python analyze.py {mode} {' '.join(extra_args)}"
    
    elif ram_gb >= 16:
        # Fast mode for 16-32GB RAM
        batch_size = min(500000, int(ram_gb * 50000))
        return f"python analyze.py preprocess-fast --performance-mode speed --batch-size {batch_size}"
    
    elif ram_gb >= 8:
        # Balanced mode for 8-16GB RAM
        batch_size = min(200000, int(ram_gb * 25000))
        n_jobs = min(cpu_count - 1, 4)
        return f"python analyze.py preprocess --batch-size {batch_size} --n-jobs {n_jobs}"
    
    else:
        # Memory-efficient mode for <8GB RAM
        return "python analyze.py preprocess-fast --performance-mode memory --batch-size 50000"

def optimize_system():
    """Apply system-level optimizations."""
    if platform.system() == "Linux" and os.geteuid() == 0:  # Running as root
        try:
            # Set CPU governor to performance
            subprocess.run("cpupower frequency-set -g performance", shell=True)
            
            # Disable transparent huge pages
            with open("/sys/kernel/mm/transparent_hugepage/enabled", "w") as f:
                f.write("never")
            
            # Increase file descriptor limit
            subprocess.run("ulimit -n 65536", shell=True)
            
            print("✓ System-level optimizations applied")
        except:
            print("⚠ Could not apply all system optimizations")

def main():
    """Main optimization and execution flow."""
    print("🚀 MovieLens Auto-Optimization Script")
    print("="*50)
    
    # Get system info
    system_info = get_system_info()
    
    # Display system info
    print(f"\n📊 System Information:")
    print(f"  CPU Cores: {system_info['cpu_count']} ({system_info['cpu_physical']} physical)")
    print(f"  RAM: {system_info['ram_gb']:.1f}GB total, {system_info['ram_available_gb']:.1f}GB available")
    print(f"  Platform: {system_info['platform']}")
    print(f"  SSD Detected: {'Yes' if system_info['has_ssd'] else 'No'}")
    print(f"  GPU Available: {'Yes' if system_info['has_gpu'] else 'No'}")
    

    # Set optimal environment
    print(f"\n⚙️  Configuring environment...")
    set_optimal_environment(system_info)
    
    # Apply system optimizations if possible
    if platform.system() == "Linux":
        optimize_system()
    
    # Determine optimal command
    command = get_optimal_command(system_info)
    print(f"\n🎯 Optimal command: {command}")
    
    # Confirm execution
    response = input("\n▶️  Run with these settings? [Y/n]: ").strip().lower()
    if response in ['', 'y', 'yes']:
        print("\n🏃 Starting optimized preprocessing...\n")
        
        # Use NUMA-aware execution if multiple nodes
        if system_info['numa_nodes'] > 1:
            command = f"numactl --interleave=all {command}"
        
        # Execute
        subprocess.run(command, shell=True)
    else:
        print("Cancelled.")

if __name__ == "__main__":
    main()