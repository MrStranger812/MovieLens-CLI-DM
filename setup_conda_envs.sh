#!/bin/bash

# =============================================================================
# MovieLens Multi-Environment Setup Script
# =============================================================================
# This script creates multiple specialized conda environments to handle
# package conflicts and optimize for different tasks.
#
# Environments:
# 1. movielens-base: Core data processing (CPU-only, minimal conflicts)
# 2. movielens-gpu: GPU acceleration with RAPIDS and CuPy (no TensorFlow)
# 3. movielens-profile: Performance profiling and analysis
# 4. movielens-optimize: Performance optimization (Numba, Dask)
# 5. movielens-viz: Visualization and reporting
# 6. movielens-test: Testing and benchmarking
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Miniconda or Anaconda first."
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# Function to remove existing environment
remove_env() {
    local env_name=$1
    if conda env list | grep -q "^$env_name "; then
        print_status "Removing existing environment: $env_name"
        conda env remove -n $env_name -y > /dev/null 2>&1 || true
    fi
}

# Function to create requirements files
create_requirements_files() {
    print_status "Creating requirements files..."
    
    mkdir -p requirements
    
    # Base requirements (CPU-only, minimal)
    cat > requirements/base.txt << 'EOF'
# Core data processing
pandas==2.1.4
numpy==1.24.4
scipy==1.11.4
scikit-learn==1.3.2

# CLI and utilities
click==8.1.7
rich==13.7.0
typer==0.9.0

# Basic visualization
matplotlib==3.8.2
seaborn==0.13.1

# Testing basics
pytest==7.4.3
hypothesis==6.92.1

# I/O optimization
pyarrow==14.0.2
fastparquet==2023.10.1
tables==3.9.2

# Utilities
joblib==1.3.2
tqdm==4.66.1
python-dateutil==2.8.2
EOF

    # GPU requirements (RAPIDS + CuPy, no TensorFlow)
    cat > requirements/gpu.txt << 'EOF'
# GPU arrays and computing
cupy-cuda12x==13.0.0

# Note: RAPIDS (cudf, cuml) will be installed via conda
# Additional GPU utilities
rmm==23.10.00

# GPU monitoring
gpustat==1.1.1
nvidia-ml-py==12.535.133
EOF

    # Profiling requirements
    cat > requirements/profile.txt << 'EOF'
# Profiling tools
line_profiler==4.1.1
memory_profiler==0.61.0
py-spy==0.3.14
snakeviz==2.2.0
guppy3==3.1.4

# System monitoring
psutil==5.9.6
gputil==1.4.0
py-cpuinfo==9.0.0

# Profiling visualization
viztracer==0.16.1
scalene==1.5.38
EOF

    # Optimization requirements
    cat > requirements/optimize.txt << 'EOF'
# JIT compilation
numba==0.58.1

# Parallel processing
dask[complete]==2023.12.1
distributed==2023.12.1
swifter==1.4.0

# Fast operations
bottleneck==1.3.7
numexpr==2.8.8

# Cython (optional)
cython==3.0.7
EOF

    # Visualization requirements
    cat > requirements/viz.txt << 'EOF'
# Advanced visualization
plotly==5.18.0
bokeh==3.3.2
altair==5.2.0
holoviews==1.18.1

# Network visualization
networkx==3.2.1
graphviz==0.20.1

# Dashboard
dash==2.14.2
streamlit==1.29.0
EOF

    # Testing requirements
    cat > requirements/test.txt << 'EOF'
# Testing framework
pytest==7.4.3
pytest-cov==4.1.0
pytest-benchmark==4.0.0
pytest-xdist==3.5.0
pytest-mock==3.12.0

# Testing utilities
hypothesis==6.92.1
faker==21.0.0
factory-boy==3.3.0

# Code quality
black==23.12.1
isort==5.13.2
flake8==7.0.0
mypy==1.8.0
EOF

    print_success "Requirements files created"
}

# Function to create and setup base environment
setup_base_env() {
    local env_name="movielens-base"
    print_status "Creating $env_name environment..."
    
    remove_env $env_name
    
    # Create environment with Python 3.10 (best compatibility)
    conda create -n $env_name python=3.10 -y > /dev/null
    
    # Activate and install packages
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install conda packages first (more stable)
    print_status "Installing conda packages..."
    conda install -c conda-forge \
        pandas numpy scipy scikit-learn matplotlib seaborn \
        ipython jupyter jupyterlab \
        -y > /dev/null
    
    # Install pip packages
    print_status "Installing pip packages..."
    pip install -r requirements/base.txt --quiet
    
    # Install the project in development mode
    pip install -e . --quiet
    
    conda deactivate
    print_success "$env_name environment created successfully"
}

# Function to create and setup GPU environment
setup_gpu_env() {
    local env_name="movielens-gpu"
    print_status "Creating $env_name environment..."
    
    remove_env $env_name
    
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        print_success "CUDA detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    else
        print_warning "CUDA not detected. GPU environment will be created but may not function properly."
    fi
    
    # Create environment with RAPIDS
    print_status "Installing RAPIDS and GPU packages (this may take a while)..."
    
    # RAPIDS requires specific CUDA versions
    conda create -n $env_name -c rapidsai -c conda-forge -c nvidia \
        python=3.10 \
        rapids=23.10 \
        cudatoolkit=11.8 \
        -y > /dev/null
    
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install additional GPU packages
    print_status "Installing additional GPU packages..."
    pip install -r requirements/gpu.txt --quiet
    
    # Install base requirements (excluding conflicting packages)
    pip install pandas scikit-learn click rich typer matplotlib seaborn pytest joblib tqdm pyarrow --quiet
    
    # Install the project
    pip install -e . --quiet
    
    conda deactivate
    print_success "$env_name environment created successfully"
}

# Function to create and setup profiling environment
setup_profile_env() {
    local env_name="movielens-profile"
    print_status "Creating $env_name environment..."
    
    remove_env $env_name
    
    conda create -n $env_name python=3.10 -y > /dev/null
    
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install base packages first
    conda install -c conda-forge pandas numpy scipy scikit-learn matplotlib -y > /dev/null
    
    # Install profiling packages
    print_status "Installing profiling packages..."
    pip install -r requirements/profile.txt --quiet
    pip install -r requirements/base.txt --quiet
    
    # Install the project
    pip install -e . --quiet
    
    conda deactivate
    print_success "$env_name environment created successfully"
}

# Function to create and setup optimization environment
setup_optimize_env() {
    local env_name="movielens-optimize"
    print_status "Creating $env_name environment..."
    
    remove_env $env_name
    
    conda create -n $env_name python=3.10 -y > /dev/null
    
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install numba from conda (better LLVM integration)
    conda install -c conda-forge numba dask distributed -y > /dev/null
    
    # Install other packages
    print_status "Installing optimization packages..."
    pip install -r requirements/optimize.txt --quiet
    pip install -r requirements/base.txt --quiet
    
    # Install the project
    pip install -e . --quiet
    
    conda deactivate
    print_success "$env_name environment created successfully"
}

# Function to create and setup visualization environment
setup_viz_env() {
    local env_name="movielens-viz"
    print_status "Creating $env_name environment..."
    
    remove_env $env_name
    
    conda create -n $env_name python=3.10 -y > /dev/null
    
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install visualization packages
    print_status "Installing visualization packages..."
    conda install -c conda-forge pandas numpy matplotlib seaborn plotly -y > /dev/null
    pip install -r requirements/viz.txt --quiet
    pip install click rich typer joblib --quiet
    
    # Install the project
    pip install -e . --quiet
    
    conda deactivate
    print_success "$env_name environment created successfully"
}

# Function to create and setup testing environment
setup_test_env() {
    local env_name="movielens-test"
    print_status "Creating $env_name environment..."
    
    remove_env $env_name
    
    conda create -n $env_name python=3.10 -y > /dev/null
    
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install testing packages
    print_status "Installing testing packages..."
    pip install -r requirements/test.txt --quiet
    pip install -r requirements/base.txt --quiet
    
    # Install the project
    pip install -e . --quiet
    
    conda deactivate
    print_success "$env_name environment created successfully"
}

# Function to validate environment
validate_env() {
    local env_name=$1
    local test_imports=$2
    
    print_status "Validating $env_name environment..."
    
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Test imports
    for package in $test_imports; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package imported successfully"
        else
            print_error "Failed to import $package"
        fi
    done
    
    conda deactivate
}

# Main execution
main() {
    print_status "MovieLens Multi-Environment Setup"
    print_status "================================="
    
    # Check prerequisites
    check_conda
    
    # Create requirements files
    create_requirements_files
    
    # Setup environments
    # setup_base_env
    setup_gpu_env
    setup_profile_env
    setup_optimize_env
    setup_viz_env
    setup_test_env
    
    # Validate environments
    print_status "\nValidating environments..."
    validate_env "movielens-base" "pandas numpy scipy sklearn click rich"
    validate_env "movielens-gpu" "cupy cudf cuml"
    validate_env "movielens-profile" "line_profiler memory_profiler psutil"
    validate_env "movielens-optimize" "numba dask distributed"
    validate_env "movielens-viz" "plotly bokeh altair"
    validate_env "movielens-test" "pytest hypothesis"
    
    # Print summary
    print_status "\n===== Setup Complete ====="
    print_success "All environments created successfully!"
    
    echo -e "\n${GREEN}Available environments:${NC}"
    echo "1. movielens-base    - Core data processing (CPU-only)"
    echo "2. movielens-gpu     - GPU acceleration with RAPIDS"
    echo "3. movielens-profile - Performance profiling"
    echo "4. movielens-optimize - Performance optimization"
    echo "5. movielens-viz     - Visualization and dashboards"
    echo "6. movielens-test    - Testing and benchmarking"
    
    echo -e "\n${YELLOW}Usage examples:${NC}"
    echo "conda activate movielens-base"
    echo "python analyze.py preprocess"
    echo ""
    echo "conda activate movielens-gpu"
    echo "python analyze.py preprocess --use-gpu"
    echo ""
    echo "conda activate movielens-profile"
    echo "python analyze.py preprocess --profile"
}

# Run main function
main
