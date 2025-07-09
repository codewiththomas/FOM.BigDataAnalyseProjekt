from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "ResearchRAG - A modular RAG system for research purposes"

# Read requirements
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="research-rag",
    version="1.0.0",
    author="FOM Research Team",
    author_email="research@fom.de",
    description="A modular RAG system for research and experimentation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/fom/research-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch>=1.12.0+cu117",
            "faiss-gpu>=1.7.0",
        ],
        "full": [
            "sentence-transformers[all]>=2.2.0",
            "spacy[de]>=3.4.0",
            "spacy[en]>=3.4.0",
            "wandb>=0.13.0",
            "mlflow>=1.20.0",
            "tensorboard>=2.8.0",
        ],
        "colab": [
            "google-colab",
            "google-auth",
            "google-auth-oauthlib",
            "google-auth-httplib2",
        ]
    },
    entry_points={
        "console_scripts": [
            "research-rag=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "research_rag": [
            "data/raw/*.txt",
            "data/evaluation/*.json",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    data_files=[
        ("config", ["src/config/baseline_config.py"]),
        ("data", ["data/raw/dsgvo.txt", "data/evaluation/qa_pairs.json"]),
    ],
    project_urls={
        "Bug Reports": "https://github.com/fom/research-rag/issues",
        "Source": "https://github.com/fom/research-rag",
        "Documentation": "https://github.com/fom/research-rag/docs",
    },
    keywords="rag, retrieval, augmented, generation, nlp, ai, research, modular",
    zip_safe=False,
)

# Additional setup for development
if __name__ == "__main__":
    import sys

    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    # Print installation instructions
    print("\n" + "="*60)
    print("ResearchRAG Setup")
    print("="*60)
    print("\nTo install in development mode:")
    print("  pip install -e .")
    print("\nTo install with GPU support:")
    print("  pip install -e .[gpu]")
    print("\nTo install with all features:")
    print("  pip install -e .[full]")
    print("\nTo install for Google Colab:")
    print("  pip install -e .[colab]")
    print("\nTo run tests:")
    print("  pytest tests/")
    print("\nTo format code:")
    print("  black src/")
    print("\nTo check code style:")
    print("  flake8 src/")
    print("\nTo run type checking:")
    print("  mypy src/")
    print("\n" + "="*60)