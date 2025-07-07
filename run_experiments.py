#!/usr/bin/env python3
"""
Hauptskript für RAG-System Experimente mit DSGVO-Daten.

Dieses Skript führt umfassende Experimente mit verschiedenen RAG-Konfigurationen durch:
- Verschiedene Chunker (Line vs. Recursive Character)
- Verschiedene Embeddings (OpenAI vs. Sentence Transformers)
- Verschiedene Vector Stores (In-Memory vs. Chroma)
- Verschiedene Language Models (OpenAI vs. Lokale Modelle)
"""

import os
import sys
from dotenv import load_dotenv
from typing import List

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('src')

from config import RAGConfig
from experiments.comprehensive_experiment_runner import ComprehensiveExperimentRunner
from data_loader import DataLoader


def create_experiment_configs() -> List[RAGConfig]:
    """
    Erstellt verschiedene RAG-Konfigurationen für Experimente.
    """
    configs = []

    # OpenAI API Key laden
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("⚠️  OpenAI API Key nicht gefunden! Verwende 'your-api-key-here' als Platzhalter.")
        openai_key = "your-api-key-here"

    # Konfiguration 1: Baseline (Line Chunker + Sentence Transformers + In-Memory + OpenAI)
    configs.append(RAGConfig(
        chunker_type="line",
        chunker_params={},
        embedding_type="sentence_transformers",
        embedding_params={},
        vector_store_type="in_memory",
        vector_store_params={},
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    ))

    # Konfiguration 2: Recursive Chunker
    configs.append(RAGConfig(
        chunker_type="recursive",
        chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
        embedding_type="sentence_transformers",
        embedding_params={},
        vector_store_type="in_memory",
        vector_store_params={},
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    ))

    # Konfiguration 3: OpenAI Embeddings
    configs.append(RAGConfig(
        chunker_type="line",
        chunker_params={},
        embedding_type="openai",
        embedding_params={"api_key": openai_key, "model": "text-embedding-ada-002"},
        vector_store_type="in_memory",
        vector_store_params={},
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    ))

    # Konfiguration 4: Chroma Vector Store
    configs.append(RAGConfig(
        chunker_type="line",
        chunker_params={},
        embedding_type="sentence_transformers",
        embedding_params={},
        vector_store_type="chroma",
        vector_store_params={"persist_directory": "./chroma_db", "collection_name": "dsgvo"},
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    ))

    # Konfiguration 5: Lokales Language Model (falls verfügbar)
    configs.append(RAGConfig(
        chunker_type="line",
        chunker_params={},
        embedding_type="sentence_transformers",
        embedding_params={},
        vector_store_type="in_memory",
        vector_store_params={},
        language_model_type="local",
        language_model_params={"model_name": "llama2", "api_url": "http://localhost:11434"}
    ))

    # Konfiguration 6: Beste Kombination (Recursive + OpenAI + Chroma)
    configs.append(RAGConfig(
        chunker_type="recursive",
        chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
        embedding_type="openai",
        embedding_params={"api_key": openai_key, "model": "text-embedding-ada-002"},
        vector_store_type="chroma",
        vector_store_params={"persist_directory": "./chroma_db", "collection_name": "dsgvo_best"},
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    ))

    return configs


def main():
    """
    Hauptfunktion für die Durchführung der Experimente.
    """
    print("🚀 RAG-System Experimente mit DSGVO-Daten")
    print("=" * 50)

    # Data Loader testen
    print("\n📄 Teste Data Loader...")
    try:
        data_loader = DataLoader()
        dsgvo_text = data_loader.load_dsgvo_document()
        stats = data_loader.get_dsgvo_statistics(dsgvo_text)
        print(f"✅ DSGVO-Dokument geladen: {stats['total_words']} Wörter, {stats['article_count']} Artikel")
    except Exception as e:
        print(f"❌ Fehler beim Laden der DSGVO-Daten: {str(e)}")
        return

    # Konfigurationen erstellen
    print("\n⚙️  Erstelle Experiment-Konfigurationen...")
    configs = create_experiment_configs()
    print(f"✅ {len(configs)} Konfigurationen erstellt")

    # Experiment Runner erstellen
    runner = ComprehensiveExperimentRunner()

    # Testfragen laden
    test_questions = data_loader.load_test_questions()
    print(f"✅ {len(test_questions)} Testfragen geladen")

    # Experimente durchführen
    print(f"\n🔬 Starte {len(configs)} Experimente...")
    results = runner.run_comprehensive_experiment(
        configs=configs,
        test_questions=test_questions,
        use_dsgvo=True
    )

    # Bericht erstellen
    print("\n📊 Erstelle Vergleichsbericht...")
    report = runner.create_comparison_report(results)

    # Bericht speichern
    report_file = "experiment_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Bericht gespeichert: {report_file}")

    # Zusammenfassung ausgeben
    print("\n" + "=" * 50)
    print("📋 EXPERIMENT-ZUSAMMENFASSUNG")
    print("=" * 50)

    overall = results.get("overall_results", {})
    successful_configs = [c for c in results.get("configurations", []) if c.get("status") == "completed"]

    print(f"✅ Erfolgreiche Konfigurationen: {len(successful_configs)}/{len(configs)}")
    print(f"📊 Durchschnittliche Relevanz: {overall.get('avg_relevance', 0):.3f}")
    print(f"📏 Durchschnittliche Antwortlänge: {overall.get('avg_response_length', 0):.1f} Zeichen")

    if successful_configs:
        best_config = max(successful_configs,
                         key=lambda x: x.get("metrics", {}).get("avg_relevance", 0))
        best_metrics = best_config.get("metrics", {})
        print(f"🏆 Beste Konfiguration: {best_config['config']['chunker_type']} + {best_config['config']['embedding_type']} + {best_config['config']['vector_store_type']} + {best_config['config']['language_model_type']}")
        print(f"   Relevanz: {best_metrics.get('avg_relevance', 0):.3f}")

    print("\n🎉 Experimente abgeschlossen!")


if __name__ == "__main__":
    main()