#!/usr/bin/env python3
"""
Auto-optimization script that detects system capabilities and runs with optimal settings.
Supports multiple conda environments for different processing needs.
"""

import os
import sys
import psutil
import subprocess
import platform
from pathlib import Path
import multiprocessing as mp
import json
from typing import Dict, List, Optional

class EnvironmentManager:
    """Manages conda environments and their configurations."""
    
    def __init__(self):
        self.environments = {
            'base': {
                'name': 'movielens-base',
                'description': 'Base environment for common data processing',
                'requirements': 'requirements/base.txt'
            },
            'gpu': {
                'name': 'movielens-rapids',
                'description': 'GPU-accelerated environment for deep learning',
                'requirements': 'requirements/gpu.txt'
            },
            'profiling': {
                'name': 'movielens-profiling',
                'description': 'Environment for performance profiling',
                'requirements': 'requirements/profiling.txt'
            },
            'optimization': {
                'name': 'movielens-optimization',
                'description': 'Environment for performance optimization',
                'requirements': 'requirements/optimization.txt'
            },
            'dev': {
                'name': 'movielens-dev',
                'description': 'Development environment with all tools',
                'requirements': ['requirements/base.txt', 'requirements/gpu.txt', 
                               'requirements/profiling.txt', 'requirements/optimization.txt']
            }
        }
        
    def get_available_environments(self) -> List[str]:
        """Get list of available environments."""
        return list(self.environments.keys())
    
    def get_environment_info(self, env_name: str) -> Dict:
        """Get information about a specific environment."""
        return self.environments.get(env_name, {})
    
    def activate_environment(self, env_name: str) -> bool:
        """Activate a conda environment."""
        if env_name not in self.environments:
            print(f"❌ Environment {env_name} not found")
            return False
            
        env_info = self.environments[env_name]
        try:
            # Activate the environment
            if platform.system() == "Windows":
                activate_cmd = f"conda activate {env_info['name']}"
            else:
                activate_cmd = f"source activate {env_info['name']}"
            
            subprocess.run(activate_cmd, shell=True, check=True)
            print(f"✓ Activated environment: {env_info['name']}")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to activate environment: {env_info['name']}")
            return False

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

def set_optimal_environment(system_info, env_manager: EnvironmentManager):
    """Set optimal environment variables based on system and selected environment."""
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

def get_optimal_command(system_info, env_name: str) -> str:
    """Determine optimal command based on system capabilities and environment."""
    ram_gb = system_info['ram_gb']
    cpu_count = system_info['cpu_count']
    
    # Base command with environment activation
    base_cmd = f"conda run -n movielens-{env_name} python analyze.py"
    
    # Determine mode based on RAM and environment
    if env_name == 'gpu' and system_info['has_gpu']:
        return f"{base_cmd} preprocess-fast --performance-mode ultrafast --gpu"
    
    elif env_name == 'optimization':
        batch_size = min(500000, int(ram_gb * 50000))
        return f"{base_cmd} preprocess-fast --performance-mode speed --batch-size {batch_size}"
    
    elif env_name == 'profiling':
        return f"{base_cmd} preprocess --profile"
    
    elif ram_gb >= 32:
        return f"{base_cmd} preprocess-fast --performance-mode ultrafast"
    
    elif ram_gb >= 16:
        batch_size = min(500000, int(ram_gb * 50000))
        return f"{base_cmd} preprocess-fast --performance-mode speed --batch-size {batch_size}"
    
    elif ram_gb >= 8:
        batch_size = min(200000, int(ram_gb * 25000))
        n_jobs = min(cpu_count - 1, 4)
        return f"{base_cmd} preprocess --batch-size {batch_size} --n-jobs {n_jobs}"
    
    else:
        return f"{base_cmd} preprocess-fast --performance-mode memory --batch-size 50000"

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
    
    # Initialize environment manager
    env_manager = EnvironmentManager()
    
    # Get system info
    system_info = get_system_info()
    
    # Display system info
    print(f"\n📊 System Information:")
    print(f"  CPU Cores: {system_info['cpu_count']} ({system_info['cpu_physical']} physical)")
    print(f"  RAM: {system_info['ram_gb']:.1f}GB total, {system_info['ram_available_gb']:.1f}GB available")
    print(f"  Platform: {system_info['platform']}")
    print(f"  SSD Detected: {'Yes' if system_info['has_ssd'] else 'No'}")
    print(f"  GPU Available: {'Yes' if system_info['has_gpu'] else 'No'}")
    
    # Display available environments
    print("\n🌍 Available Environments:")
    for env_name, env_info in env_manager.environments.items():
        print(f"  • {env_name}: {env_info['description']}")
    
    # Get environment selection
    while True:
        env_name = input("\n▶️  Select environment to use: ").strip().lower()
        if env_name in env_manager.environments:
            break
        print(f"❌ Invalid environment. Choose from: {', '.join(env_manager.get_available_environments())}")
    
    # Set optimal environment
    print(f"\n⚙️  Configuring environment...")
    set_optimal_environment(system_info, env_manager)
    
    # Apply system optimizations if possible
    if platform.system() == "Linux":
        optimize_system()
    
    # Determine optimal command
    command = get_optimal_command(system_info, env_name)
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