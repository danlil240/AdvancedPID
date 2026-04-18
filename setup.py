#!/usr/bin/env python3
"""
Setup script for Advanced PID Control Library.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="advanced-pid-control",
    version="1.0.0",
    author="Advanced PID Control Project",
    description="Professional PID control library with tuning, analysis, and simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/advanced-pid-control",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.9",
        "matplotlib>=3.5",
        "control>=0.10,<0.12",
        "gymnasium>=0.29",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "hypothesis>=6.80",
            "ruff>=0.4",
            "mypy>=1.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "pidtune=pid_control.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pid control automation tuning simulation autotune",
)
