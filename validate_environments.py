#!/usr/bin/env python3
"""
Environment validation script for MovieLens project.
Checks all conda environments and their dependencies.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

class EnvironmentValidator:
    """Validates conda environments and their dependencies."""
    
    def __init__(self):
        self.environments = {
            'base': {
                'name': 'movielens-base',
                'required_packages': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'pytest', 'hypothesis'],
                'required_files': ['requirements/base.txt']
            },
            'gpu': {
                'name': 'movielens-rapids',
                'required_packages': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'torch', 'cuda-python'],
                'required_files': ['requirements/gpu.txt']
            },
            'profiling': {
                'name': 'movielens-profiling',
                'required_packages': ['pandas', 'numpy', 'scipy', 'line_profiler', 'memory_profiler', 'snakeviz'],
                'required_files': ['requirements/profiling.txt']
            },
            'optimization': {
                'name': 'movielens-optimization',
                'required_packages': ['pandas', 'numpy', 'scipy', 'numba', 'dask', 'pyarrow'],
                'required_files': ['requirements/optimization.txt']
            },
            'docs': {
                'name': 'movielens-docs',
                'required_packages': ['pandas', 'numpy', 'scipy', 'sphinx', 'sphinx-rtd-theme', 'mkdocs'],
                'required_files': ['requirements/docs.txt']
            },
            'testing': {
                'name': 'movielens-testing',
                'required_packages': ['pandas', 'numpy', 'scipy', 'pytest', 'hypothesis', 'coverage'],
                'required_files': ['requirements/testing.txt']
            },
            'dev': {
                'name': 'movielens-dev',
                'required_packages': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 
                                    'pytest', 'hypothesis', 'torch', 'numba', 'dask', 'sphinx'],
                'required_files': ['requirements/base.txt', 'requirements/gpu.txt', 
                                 'requirements/profiling.txt', 'requirements/optimization.txt',
                                 'requirements/docs.txt', 'requirements/testing.txt']
            }
        }
    
    def check_environment_exists(self, env_name: str) -> bool:
        """Check if a conda environment exists."""
        try:
            result = subprocess.run(['conda', 'env', 'list'], 
                                 capture_output=True, text=True)
            return env_name in result.stdout
        except:
            return False
    
    def check_package_installed(self, env_name: str, package: str) -> bool:
        """Check if a package is installed in the environment."""
        try:
            result = subprocess.run(
                ['conda', 'run', '-n', env_name, 'python', '-c', f'import {package}'],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
    
    def check_requirements_file(self, file_path: str) -> bool:
        """Check if a requirements file exists."""
        return Path(file_path).exists()
    
    def validate_environment(self, env_name: str) -> Dict:
        """Validate a specific environment."""
        env_info = self.environments.get(env_name)
        if not env_info:
            return {'exists': False, 'error': f'Environment {env_name} not defined'}
        
        results = {
            'exists': self.check_environment_exists(env_info['name']),
            'packages': {},
            'requirements_files': {}
        }
        
        # Check packages
        for package in env_info['required_packages']:
            results['packages'][package] = self.check_package_installed(env_info['name'], package)
        
        # Check requirements files
        for req_file in env_info['required_files']:
            results['requirements_files'][req_file] = self.check_requirements_file(req_file)
        
        return results

@click.command()
@click.option('--env', help='Specific environment to validate')
@click.option('--fix', is_flag=True, help='Attempt to fix missing dependencies')
def validate(env: Optional[str], fix: bool):
    """Validate conda environments and their dependencies."""
    validator = EnvironmentValidator()
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]MovieLens Environment Validator[/bold blue]\n"
        "Validates all conda environments and their dependencies",
        border_style="blue"
    ))
    
    # Determine which environments to validate
    envs_to_check = [env] if env else validator.environments.keys()
    
    # Create results table
    results_table = Table(title="Environment Validation Results", box=box.ROUNDED)
    results_table.add_column("Environment", style="cyan")
    results_table.add_column("Status", style="bold")
    results_table.add_column("Packages", style="green")
    results_table.add_column("Requirements", style="yellow")
    results_table.add_column("Actions", style="dim")
    
    for env_name in envs_to_check:
        results = validator.validate_environment(env_name)
        
        if not results['exists']:
            status = "[red]❌ Missing[/red]"
            packages = "N/A"
            requirements = "N/A"
            actions = "Run setup_conda_envs.sh"
        else:
            # Check package status
            missing_packages = [pkg for pkg, installed in results['packages'].items() if not installed]
            if missing_packages:
                status = "[yellow]⚠ Incomplete[/yellow]"
                packages = f"[red]{len(missing_packages)} missing[/red]"
            else:
                status = "[green]✓ Complete[/green]"
                packages = f"[green]{len(results['packages'])} installed[/green]"
            
            # Check requirements files
            missing_files = [f for f, exists in results['requirements_files'].items() if not exists]
            if missing_files:
                requirements = f"[red]{len(missing_files)} missing[/red]"
            else:
                requirements = f"[green]{len(results['requirements_files'])} found[/green]"
            
            # Suggest actions
            actions = []
            if missing_packages:
                actions.append("Install missing packages")
            if missing_files:
                actions.append("Create missing requirements files")
            actions = "\n".join(actions) if actions else "None needed"
        
        results_table.add_row(env_name, status, packages, requirements, actions)
    
    console.print(results_table)
    
    # Display summary
    console.print("\n[bold cyan]Environment Usage Guide[/bold cyan]")
    console.print("""
[bold]Base Environment (movielens-base)[/bold]
• Basic data processing and analysis
• Command: python analyze.py preprocess --performance-mode balanced

[bold]GPU Environment (movielens-rapids)[/bold]
• Deep learning and GPU-accelerated processing
• Command: python analyze.py preprocess --performance-mode speed --use-gpu

[bold]Profiling Environment (movielens-profiling)[/bold]
• Performance profiling and analysis
• Command: python analyze.py preprocess --profile

[bold]Optimization Environment (movielens-optimization)[/bold]
• Performance optimization and benchmarking
• Command: python analyze.py preprocess --performance-mode speed

[bold]Documentation Environment (movielens-docs)[/bold]
• Documentation generation
• Command: python analyze.py docs

[bold]Testing Environment (movielens-testing)[/bold]
• Running tests and benchmarks
• Command: python analyze.py test

[bold]Development Environment (movielens-dev)[/bold]
• Complete development environment with all tools
• Use for development and testing all features
""")

if __name__ == '__main__':
    validate() 