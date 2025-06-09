#!/usr/bin/env python3
"""
Comprehensive environment validation for MovieLens project.
Tests all environments and their specific functionalities.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Rich imports for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich not available. Install with: pip install rich")

console = Console() if RICH_AVAILABLE else None

class EnvironmentValidator:
    """Validates all MovieLens conda environments comprehensively."""
    
    def __init__(self):
        self.environments = {
            'movielens-base': {
                'description': 'Core data processing (CPU-only)',
                'test_imports': [
                    'pandas', 'numpy', 'scipy', 'sklearn',
                    'click', 'rich', 'matplotlib', 'seaborn'
                ],
                'test_commands': [
                    'python -c "import pandas; print(f\'Pandas {pandas.__version__}\')"',
                    'python -c "import numpy; print(f\'NumPy {numpy.__version__}\')"',
                    'python analyze.py --help'
                ]
            },
            'movielens-gpu': {
                'description': 'GPU acceleration with RAPIDS',
                'test_imports': [
                    'cupy', 'cudf', 'cuml', 'rmm'
                ],
                'test_commands': [
                    'python -c "import cupy; print(f\'CuPy {cupy.__version__}\')"',
                    'python -c "import cudf; print(f\'cuDF {cudf.__version__}\')"',
                    'python -c "import cupy; print(f\'GPU count: {cupy.cuda.runtime.getDeviceCount()}\')"'
                ],
                'gpu_required': True
            },
            'movielens-profile': {
                'description': 'Performance profiling tools',
                'test_imports': [
                    'line_profiler', 'memory_profiler', 'psutil'
                ],
                'test_commands': [
                    'python -c "import psutil; print(f\'CPU: {psutil.cpu_count()} cores\')"',
                    'python -c "import psutil; print(f\'Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB\')"'
                ]
            },
            'movielens-optimize': {
                'description': 'CPU optimization tools',
                'test_imports': [
                    'numba', 'dask', 'distributed', 'swifter'
                ],
                'test_commands': [
                    'python -c "import numba; print(f\'Numba {numba.__version__}\')"',
                    'python -c "import dask; print(f\'Dask {dask.__version__}\')"'
                ]
            },
            'movielens-viz': {
                'description': 'Visualization and dashboards',
                'test_imports': [
                    'plotly', 'bokeh', 'altair'
                ],
                'test_commands': [
                    'python -c "import plotly; print(f\'Plotly {plotly.__version__}\')"'
                ]
            },
            'movielens-test': {
                'description': 'Testing and code quality',
                'test_imports': [
                    'pytest', 'hypothesis', 'black', 'flake8'
                ],
                'test_commands': [
                    'pytest --version',
                    'black --version'
                ]
            }
        }
        
        self.results = {}
    
    def print_header(self):
        """Print validation header."""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]MovieLens Environment Validator[/bold blue]\n"
                f"Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="blue"
            ))
        else:
            print("=" * 60)
            print("MovieLens Environment Validator")
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
    
    def check_conda(self) -> bool:
        """Check if conda is available."""
        try:
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                if RICH_AVAILABLE:
                    console.print(f"[green]✓ Conda found: {version}[/green]")
                else:
                    print(f"✓ Conda found: {version}")
                return True
        except FileNotFoundError:
            pass
        
        if RICH_AVAILABLE:
            console.print("[red]✗ Conda not found. Please install Miniconda or Anaconda.[/red]")
        else:
            print("✗ Conda not found. Please install Miniconda or Anaconda.")
        return False
    
    def check_gpu(self) -> Tuple[bool, str]:
        """Check GPU availability."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                return True, gpu_info
        except FileNotFoundError:
            pass
        return False, "No GPU detected"
    
    def check_environment_exists(self, env_name: str) -> bool:
        """Check if environment exists."""
        try:
            result = subprocess.run(['conda', 'env', 'list'], 
                                  capture_output=True, text=True)
            return env_name in result.stdout
        except:
            return False
    
    def test_imports(self, env_name: str, imports: List[str]) -> Dict[str, bool]:
        """Test package imports in environment."""
        results = {}
        
        for package in imports:
            cmd = f'conda run -n {env_name} python -c "import {package}"'
            try:
                result = subprocess.run(cmd, shell=True, 
                                      capture_output=True, text=True)
                results[package] = result.returncode == 0
            except:
                results[package] = False
        
        return results
    
    def test_commands(self, env_name: str, commands: List[str]) -> Dict[str, Tuple[bool, str]]:
        """Test commands in environment."""
        results = {}
        
        for command in commands:
            full_cmd = f'conda run -n {env_name} {command}'
            try:
                result = subprocess.run(full_cmd, shell=True,
                                      capture_output=True, text=True, timeout=30)
                success = result.returncode == 0
                output = result.stdout.strip() if success else result.stderr.strip()
                results[command] = (success, output)
            except subprocess.TimeoutExpired:
                results[command] = (False, "Command timed out")
            except Exception as e:
                results[command] = (False, str(e))
        
        return results
    
    def validate_environment(self, env_name: str) -> Dict:
        """Validate a single environment."""
        env_config = self.environments[env_name]
        
        # Check if environment exists
        exists = self.check_environment_exists(env_name)
        if not exists:
            return {
                'exists': False,
                'description': env_config['description'],
                'imports': {},
                'commands': {},
                'errors': ['Environment does not exist']
            }
        
        # Test imports
        import_results = self.test_imports(env_name, env_config['test_imports'])
        
        # Test commands
        command_results = self.test_commands(env_name, env_config['test_commands'])
        
        # Check for errors
        errors = []
        failed_imports = [pkg for pkg, success in import_results.items() if not success]
        if failed_imports:
            errors.append(f"Failed imports: {', '.join(failed_imports)}")
        
        failed_commands = [cmd for cmd, (success, _) in command_results.items() if not success]
        if failed_commands:
            errors.append(f"Failed commands: {len(failed_commands)}")
        
        return {
            'exists': True,
            'description': env_config['description'],
            'imports': import_results,
            'commands': command_results,
            'errors': errors,
            'gpu_required': env_config.get('gpu_required', False)
        }
    
    def run_validation(self):
        """Run validation for all environments."""
        self.print_header()
        
        # Check prerequisites
        if not self.check_conda():
            return
        
        # Check GPU
        gpu_available, gpu_info = self.check_gpu()
        if RICH_AVAILABLE:
            if gpu_available:
                console.print(f"[green]✓ GPU detected: {gpu_info}[/green]")
            else:
                console.print("[yellow]⚠ No GPU detected (GPU environments may not work)[/yellow]")
        else:
            print(f"GPU: {gpu_info}")
        
        # Validate each environment
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Validating environments...", 
                                       total=len(self.environments))
                
                for env_name in self.environments:
                    progress.update(task, description=f"Validating {env_name}...")
                    self.results[env_name] = self.validate_environment(env_name)
                    progress.advance(task)
        else:
            for env_name in self.environments:
                print(f"\nValidating {env_name}...")
                self.results[env_name] = self.validate_environment(env_name)
        
        # Display results
        self.display_results(gpu_available)
    
    def display_results(self, gpu_available: bool):
        """Display validation results."""
        if RICH_AVAILABLE:
            # Create summary table
            table = Table(title="Environment Validation Summary", box=box.ROUNDED)
            table.add_column("Environment", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("Packages", justify="center")
            table.add_column("Commands", justify="center")
            table.add_column("Notes", style="dim")
            
            for env_name, result in self.results.items():
                if not result['exists']:
                    status = "[red]Missing[/red]"
                    packages = "—"
                    commands = "—"
                    notes = "Run setup_conda_envs.sh"
                else:
                    # Calculate success rates
                    import_success = sum(1 for s in result['imports'].values() if s)
                    import_total = len(result['imports'])
                    cmd_success = sum(1 for s, _ in result['commands'].values() if s)
                    cmd_total = len(result['commands'])
                    
                    # Status
                    if result['errors']:
                        status = "[yellow]Issues[/yellow]"
                    else:
                        status = "[green]Ready[/green]"
                    
                    # GPU check for GPU environment
                    if result.get('gpu_required') and not gpu_available:
                        status = "[yellow]No GPU[/yellow]"
                        notes = "GPU required but not available"
                    else:
                        notes = result['description']
                    
                    packages = f"{import_success}/{import_total}"
                    commands = f"{cmd_success}/{cmd_total}"
                
                table.add_row(env_name, status, packages, commands, notes)
            
            console.print("\n")
            console.print(table)
            
            # Show detailed errors if any
            self.show_detailed_errors()
            
            # Show recommendations
            self.show_recommendations(gpu_available)
            
        else:
            # Simple text output
            print("\n" + "=" * 60)
            print("VALIDATION RESULTS")
            print("=" * 60)
            
            for env_name, result in self.results.items():
                print(f"\n{env_name}:")
                print(f"  Description: {result['description']}")
                print(f"  Exists: {'Yes' if result['exists'] else 'No'}")
                
                if result['exists']:
                    import_success = sum(1 for s in result['imports'].values() if s)
                    import_total = len(result['imports'])
                    print(f"  Package imports: {import_success}/{import_total} successful")
                    
                    if result['errors']:
                        print(f"  Errors: {'; '.join(result['errors'])}")
    
    def show_detailed_errors(self):
        """Show detailed error information."""
        has_errors = False
        
        for env_name, result in self.results.items():
            if result['exists'] and result['errors']:
                has_errors = True
                
                if RICH_AVAILABLE:
                    # Create error tree
                    tree = Tree(f"[yellow]{env_name}[/yellow]")
                    
                    # Failed imports
                    failed_imports = [pkg for pkg, success in result['imports'].items() 
                                    if not success]
                    if failed_imports:
                        import_branch = tree.add("[red]Failed Imports[/red]")
                        for pkg in failed_imports:
                            import_branch.add(f"❌ {pkg}")
                    
                    # Failed commands
                    for cmd, (success, output) in result['commands'].items():
                        if not success:
                            cmd_branch = tree.add(f"[red]Failed Command[/red]")
                            cmd_branch.add(f"Command: {cmd[:50]}...")
                            cmd_branch.add(f"Error: {output[:100]}...")
                    
                    console.print(tree)
        
        if not has_errors and RICH_AVAILABLE:
            console.print("\n[green]✅ No errors detected![/green]")
    
    def show_recommendations(self, gpu_available: bool):
        """Show recommendations based on validation results."""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📋 Recommendations[/bold cyan]")
            
            # Check if any environments are missing
            missing_envs = [env for env, result in self.results.items() 
                          if not result['exists']]
            
            if missing_envs:
                console.print(f"\n[yellow]⚠ Missing environments: {', '.join(missing_envs)}[/yellow]")
                console.print("Run: [bold]./setup_conda_envs.sh[/bold]")
            
            # GPU recommendations
            if not gpu_available:
                console.print("\n[yellow]⚠ No GPU detected[/yellow]")
                console.print("GPU environment (movielens-gpu) will not work properly.")
                console.print("Use movielens-optimize for CPU-based acceleration instead.")
            
            # Success message
            all_exist = all(result['exists'] for result in self.results.values())
            if all_exist and gpu_available:
                console.print("\n[green]✅ All environments are properly configured![/green]")
                console.print("\nYou can now run:")
                console.print("  [bold]conda activate movielens-base[/bold]")
                console.print("  [bold]python analyze.py preprocess[/bold]")

def create_test_script():
    """Create a quick test script for preprocessing."""
    test_script = '''#!/usr/bin/env python3
"""Quick test of preprocessing in different environments."""

import time
import subprocess
import pandas as pd

def test_environment(env_name, command):
    """Test preprocessing in an environment."""
    print(f"\\nTesting {env_name}...")
    start_time = time.time()
    
    full_command = f"conda run -n {env_name} {command}"
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ Success in {elapsed:.2f} seconds")
    else:
        print(f"✗ Failed: {result.stderr[:200]}")
    
    return elapsed if result.returncode == 0 else None

# Test commands for each environment
tests = [
    ("movielens-base", "python analyze.py preprocess --batch-size 1000 --no-save"),
    ("movielens-optimize", "python analyze.py preprocess-fast --batch-size 1000 --no-save"),
    ("movielens-gpu", "python analyze.py preprocess --use-gpu --batch-size 1000 --no-save"),
]

print("Running preprocessing tests...")
results = {}

for env, cmd in tests:
    time_taken = test_environment(env, cmd)
    if time_taken:
        results[env] = time_taken

# Show comparison
if results:
    print("\\n" + "="*50)
    print("Performance Comparison:")
    print("="*50)
    
    baseline = results.get("movielens-base", 1)
    for env, time_taken in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline / time_taken
        print(f"{env:20} {time_taken:8.2f}s  ({speedup:.2f}x speedup)")
'''
    
    with open('test_environments.py', 'w') as f:
        f.write(test_script)
    
    if RICH_AVAILABLE:
        console.print("\n[green]Created test_environments.py[/green]")
        console.print("Run it with: [bold]python test_environments.py[/bold]")

def main():
    """Main validation routine."""
    validator = EnvironmentValidator()
    validator.run_validation()
    
    # Create test script
    create_test_script()
    
    if RICH_AVAILABLE:
        console.print("\n[bold green]Validation complete![/bold green]")

if __name__ == "__main__":
    main()