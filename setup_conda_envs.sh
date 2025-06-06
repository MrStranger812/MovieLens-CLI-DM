#!/bin/bash

# Function to remove existing environment if it exists
remove_env() {
    local env_name=$1
    if conda env list | grep -q "^$env_name "; then
        echo "Removing existing environment: $env_name"
        conda env remove -n $env_name
    fi
}

# Function to create and configure a conda environment
create_env() {
    local env_name=$1
    local python_version=$2
    local packages=$3
    local pip_packages=$4
    local requirements_file=$5
    
    echo "Creating conda environment: $env_name"
    conda create -y -n $env_name python=$python_version
    conda activate $env_name
    
    # Install packages
    if [ ! -z "$packages" ]; then
        conda install -y $packages
    fi
    
    # Install pip packages if specified
    if [ ! -z "$pip_packages" ]; then
        pip install $pip_packages
    fi
    
    # Install requirements file if specified
    if [ ! -z "$requirements_file" ]; then
        pip install -r requirements/$requirements_file.txt
    fi
    
    conda deactivate
    echo "Environment $env_name created successfully"
}

# Remove existing environments
echo "Cleaning up existing environments..."
remove_env "movielens-base"
remove_env "movielens-gpu"
remove_env "movielens-profiling"
remove_env "movielens-optimization"
remove_env "movielens-docs"
remove_env "movielens-testing"
remove_env "movielens-dev"
remove_env "movielens-rapids"

# Base environment for common dependencies
create_env "movielens-base" "3.11" \
    "pandas=2.2.0 numpy=1.26.4 scipy=1.12.0 matplotlib=3.8.3 seaborn=0.13.2 plotly=5.18.0 pytest=8.0.2 hypothesis=6.98.0" \
    "" "base"

# GPU environment with RAPIDS and PyTorch
echo "Creating GPU environment with RAPIDS..."
conda create -y -n movielens-rapids -c rapidsai -c conda-forge -c nvidia \
    cudf=25.04 cuml=25.04 python=3.11 'cuda-version>=12.0,<=12.8' \
    'pytorch=*=*cuda*' tensorflow dash jupyterlab
conda activate movielens-rapids
pip install -r requirements/gpu.txt
conda deactivate

# Profiling environment
create_env "movielens-profiling" "3.11" \
    "pandas=2.2.0 numpy=1.26.4 scipy=1.12.0 line_profiler=4.1.1 memory_profiler=0.61.0 snakeviz=2.2.1" \
    "" "profiling"

# Optimization environment
create_env "movielens-optimization" "3.11" \
    "pandas=2.2.0 numpy=1.26.4 scipy=1.12.0 numba=0.58.1 dask=2024.2.1 pyarrow=15.0.0" \
    "" "optimization"

# Documentation environment
create_env "movielens-docs" "3.11" \
    "pandas=2.2.0 numpy=1.26.4 scipy=1.12.0 sphinx=7.2.6 sphinx-rtd-theme=2.0.0 mkdocs=1.5.3" \
    "" "docs"

# Testing environment
create_env "movielens-testing" "3.11" \
    "pandas=2.2.0 numpy=1.26.4 scipy=1.12.0 pytest=8.0.2 hypothesis=6.98.0 coverage=7.4.1" \
    "" "testing"

# Create a development environment that combines all tools
create_env "movielens-dev" "3.11" \
    "pandas=2.2.0 numpy=1.26.4 scipy=1.12.0 matplotlib=3.8.3 seaborn=0.13.2 plotly=5.18.0 pytest=8.0.2 hypothesis=6.98.0" \
    "" "base"
conda activate movielens-dev
pip install -r requirements/gpu.txt
pip install -r requirements/profiling.txt
pip install -r requirements/optimization.txt
pip install -r requirements/docs.txt
pip install -r requirements/testing.txt
conda deactivate

echo "All conda environments have been created successfully!"
echo "To activate an environment, use: conda activate <env-name>"
echo "Available environments:"
echo "- movielens-base: Base environment with common dependencies"
echo "- movielens-rapids: GPU-accelerated environment with RAPIDS and PyTorch"
echo "- movielens-profiling: Environment for code profiling and performance analysis"
echo "- movielens-optimization: Environment for performance optimization"
echo "- movielens-docs: Environment for documentation generation"
echo "- movielens-testing: Environment for testing and benchmarking"
echo "- movielens-dev: Development environment with all tools combined" 