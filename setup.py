from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="research-rag",
    version="0.1.0",
    description="Modulares RAG-System für wissenschaftliche Studien",
    author="FOM Research Team",
    author_email="research@fom.de",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="rag, retrieval, augmented generation, research, nlp",
    project_urls={
        "Bug Reports": "https://github.com/codewiththomas/rrag/issues",
        "Source": "https://github.com/codewiththomas/rrag",
    },
) 