# Conda Environment Setup for MovieLens Project

This project uses multiple conda environments to separate different aspects of development and ensure optimal performance for each task.

## Available Environments

### 1. Base Environment (`movielens-base`)
- Contains common dependencies for data processing and basic analysis
- Used for general development and running basic scripts
- Includes: pandas, numpy, scipy, matplotlib, seaborn, plotly, and CLI tools

### 2. GPU Environment (`movielens-gpu`)
- Optimized for deep learning and GPU-accelerated computations
- Includes TensorFlow and PyTorch with GPU support
- Use this environment for training deep learning models and heavy computations

### 3. Profiling Environment (`movielens-profiling`)
- Contains tools for performance analysis and profiling
- Includes: line_profiler, memory_profiler, py-spy, and snakeviz
- Use this environment when you need to analyze code performance

### 4. Optimization Environment (`movielens-optimization`)
- Contains tools for code optimization
- Includes: numba, cython, dask, and distributed
- Use this environment when working on performance optimization

### 5. Documentation Environment (`movielens-docs`)
- Contains tools for generating documentation
- Includes: sphinx, mkdocs, and mkdocs-material
- Use this environment when working on project documentation

### 6. Testing Environment (`movielens-testing`)
- Contains tools for testing and benchmarking
- Includes: pytest-cov, pytest-benchmark, and pytest-xdist
- Use this environment when running tests and benchmarks

## Setup Instructions

1. Make sure you have Conda installed on your system
2. Run the setup script:
   ```bash
   chmod +x setup_conda_envs.sh
   ./setup_conda_envs.sh
   ```

## Usage

To activate an environment:
```bash
conda activate <env-name>
```

For example:
```bash
conda activate movielens-gpu  # For GPU-accelerated work
conda activate movielens-profiling  # For performance profiling
```

## Environment Selection Guide

- Use `movielens-base` for general development and running basic scripts
- Use `movielens-gpu` when working with deep learning models or GPU-accelerated computations
- Use `movielens-profiling` when you need to analyze code performance
- Use `movielens-optimization` when working on performance optimization
- Use `movielens-docs` when generating or updating documentation
- Use `movielens-testing` when running tests or benchmarks

## Requirements Files

Each environment has its own requirements file in the `requirements/` directory:
- `base.txt`: Common dependencies
- `gpu.txt`: GPU-accelerated computing dependencies
- `profiling.txt`: Performance profiling tools
- `optimization.txt`: Performance optimization tools
- `docs.txt`: Documentation tools
- `testing.txt`: Testing and benchmarking tools

## Updating Environments

To update an environment with its requirements:
```bash
conda activate <env-name>
pip install -r requirements/<env-name>.txt
``` 