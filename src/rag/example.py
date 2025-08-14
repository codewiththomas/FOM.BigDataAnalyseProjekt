#!/usr/bin/env python3
"""
ResearchRAG Example Usage
Demonstrates how to use the ResearchRAG system programmatically
"""

import logging
from pathlib import Path
from .factory import RAGFactory
from .dataset import DSGVODataset

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    """Example usage of ResearchRAG"""

    # Configuration and dataset paths
    config_path = "configs/baseline.yaml"
    dataset_path = "data/output/dsgvo_crawled_2025-08-14_1535.jsonl"

    print("🚀 ResearchRAG Example")
    print("=" * 50)

    try:
        # 1. Create RAG factory and pipeline
        print("1. Creating RAG pipeline...")
        factory = RAGFactory(config_path)
        pipeline = factory.create_pipeline()

        print(f"✅ Pipeline created with components:")
        pipeline_info = pipeline.get_pipeline_info()
        for component, info in pipeline_info.items():
            if component != 'is_indexed':
                print(f"   - {component}: {info.get('name', 'unknown')}")

        # 2. Load and prepare dataset
        print("\n2. Loading DSGVO dataset...")
        dataset = DSGVODataset(dataset_path)
        documents = dataset.get_documents()
        qa_pairs = dataset.get_qa_pairs()

        print(f"✅ Loaded {len(documents)} documents")
        print(f"✅ Generated {len(qa_pairs)} QA pairs")

        # 3. Index documents
        print("\n3. Indexing documents...")
        pipeline.index_documents(documents)
        print("✅ Documents indexed successfully")

        # 4. Run sample queries
        print("\n4. Running sample queries...")
        sample_qa = qa_pairs[:3]  # First 3 QA pairs

        for i, qa in enumerate(sample_qa, 1):
            print(f"\n--- Query {i} ---")
            print(f"Question: {qa['question']}")

            # Query the pipeline
            result = pipeline.query(qa['question'])

            print(f"Response: {result.response[:200]}...")
            print(f"Retrieved chunks: {len(result.chunks)}")
            print(f"Query time: {result.metadata.get('query_time', 0):.2f}s")

        # 5. Show pipeline information
        print("\n5. Pipeline Information:")
        print("=" * 30)
        for key, value in pipeline_info.items():
            if key != 'is_indexed':
                print(f"{key}: {value}")

        print("\n🎉 Example completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error("Example failed", exc_info=True)


if __name__ == "__main__":
    main()
