"""
Setup script for momentest.
"""

from setuptools import setup, find_packages

setup(
    name="momentest",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "momentest": ["data/*.txt", "py.typed"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "hypothesis>=6.0", "matplotlib>=3.5", "ruff>=0.1.0"],
        "plot": ["matplotlib>=3.5"],
    },
)
