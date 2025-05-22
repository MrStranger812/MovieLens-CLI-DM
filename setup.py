from setuptools import setup, find_packages

setup(
    name="movielens-analytics",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "click>=8.0.0",
        "rich>=12.0.0",
    ],
    entry_points={
        'console_scripts': [
            'movielens=movielens.cli:cli',
        ],
    },
)