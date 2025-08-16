"""Setup configuration for inflection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name = "inflection",
    version = "0.2.0",
    author = "Jim Java",
    author_email = "james.j.java@gmail.com",
    description = "Find inflection points of curves using ESE and EDE methods",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/priscian/inflection",
    packages = find_packages(),
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires = ">=3.7",
    install_requires = [
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "joblib>=1.0.0",
    ],
    extras_require = {
        'dev': [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        'minimal': [
            "numpy>=1.19.0",
            "scipy>=1.5.0",
        ]
    },
)
