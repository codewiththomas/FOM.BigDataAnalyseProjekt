#!/usr/bin/env python3
"""
Beispiel für die Verwendung des RAG-Systems mit DSGVO-Daten.

Dieses Skript demonstriert:
1. Laden und Verarbeiten der DSGVO-Daten
2. Erstellen verschiedener RAG-Konfigurationen
3. Durchführung von Experimenten
4. Evaluierung der Ergebnisse
"""

import os
import sys
from dotenv import load_dotenv

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('src')

from config import RAGConfig
from rag_system import RAGSystem
from data_loader import DataLoader
from evaluations.rag_metrics import RAGMetrics


def example_basic_usage():
    """
    Grundlegende Verwendung des RAG-Systems.
    """
    print("🔍 Beispiel: Grundlegende RAG-Verwendung")
    print("=" * 50)

    # OpenAI API Key laden
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    # Konfiguration erstellen
    config = RAGConfig(
        chunker_type="line",
        embedding_type="sentence_transformers",
        vector_store_type="in_memory",
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    )

    # RAG-System erstellen
    rag = RAGSystem(config)

    # DSGVO-Daten laden
    data_loader = DataLoader()
    dsgvo_text = data_loader.load_dsgvo_document()

    # Dokumente verarbeiten
    print("📄 Verarbeite DSGVO-Dokument...")
    rag.process_documents([dsgvo_text])

    # Fragen stellen
    questions = [
        "Was ist die DSGVO?",
        "Welche Rechte haben betroffene Personen?",
        "Was ist ein Verantwortlicher?"
    ]

    print("\n❓ Teste Fragen:")
    for question in questions:
        try:
            answer = rag.query(question)
            print(f"\nFrage: {question}")
            print(f"Antwort: {answer[:200]}...")
        except Exception as e:
            print(f"Fehler bei Frage '{question}': {str(e)}")


def example_comparison():
    """
    Vergleich verschiedener Konfigurationen.
    """
    print("\n\n🔬 Beispiel: Konfigurationsvergleich")
    print("=" * 50)

    # OpenAI API Key laden
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    # Verschiedene Konfigurationen
    configs = [
        ("Baseline", RAGConfig(
            chunker_type="line",
            embedding_type="sentence_transformers",
            vector_store_type="in_memory",
            language_model_type="openai",
            language_model_params={"api_key": openai_key}
        )),
        ("Recursive Chunker", RAGConfig(
            chunker_type="recursive",
            chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
            embedding_type="sentence_transformers",
            vector_store_type="in_memory",
            language_model_type="openai",
            language_model_params={"api_key": openai_key}
        )),
        ("OpenAI Embeddings", RAGConfig(
            chunker_type="line",
            embedding_type="openai",
            embedding_params={"api_key": openai_key, "model": "text-embedding-ada-002"},
            vector_store_type="in_memory",
            language_model_type="openai",
            language_model_params={"api_key": openai_key}
        ))
    ]

    # Testfrage
    test_question = "Was ist die DSGVO?"

    # DSGVO-Daten laden
    data_loader = DataLoader()
    dsgvo_text = data_loader.load_dsgvo_document()

    # Vergleich durchführen
    for name, config in configs:
        print(f"\n🧪 Teste {name}...")
        try:
            rag = RAGSystem(config)
            rag.process_documents([dsgvo_text])
            answer = rag.query(test_question)
            print(f"Antwort: {answer[:150]}...")
        except Exception as e:
            print(f"Fehler: {str(e)}")


def example_evaluation():
    """
    Beispiel für die Evaluierung von RAG-Systemen.
    """
    print("\n\n📊 Beispiel: RAG-Evaluierung")
    print("=" * 50)

    # OpenAI API Key laden
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    # Konfiguration
    config = RAGConfig(
        chunker_type="line",
        embedding_type="sentence_transformers",
        vector_store_type="in_memory",
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    )

    # RAG-System erstellen
    rag = RAGSystem(config)

    # DSGVO-Daten laden
    data_loader = DataLoader()
    dsgvo_text = data_loader.load_dsgvo_document()
    rag.process_documents([dsgvo_text])

    # Testfragen
    test_questions = [
        "Was ist die DSGVO?",
        "Welche Rechte haben betroffene Personen?",
        "Was ist ein Verantwortlicher?",
        "Was ist die Einwilligung?"
    ]

    # Antworten generieren
    answers = []
    for question in test_questions:
        try:
            answer = rag.query(question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Fehler: {str(e)}")

    # Metriken berechnen
    metrics = RAGMetrics()

    # Einfache Relevanz-Metriken
    relevance_scores = []
    for question, answer in zip(test_questions, answers):
        if not answer.startswith("Fehler"):
            relevance = metrics._calculate_simple_relevance(question, answer)
            relevance_scores.append(relevance)
            print(f"Frage: {question}")
            print(f"Relevanz: {relevance:.3f}")
            print(f"Antwort: {answer[:100]}...")
            print()

    if relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        print(f"📊 Durchschnittliche Relevanz: {avg_relevance:.3f}")
        print(f"📊 Anzahl erfolgreicher Antworten: {len(relevance_scores)}/{len(test_questions)}")


def example_custom_questions():
    """
    Beispiel für benutzerdefinierte Fragen.
    """
    print("\n\n💭 Beispiel: Benutzerdefinierte Fragen")
    print("=" * 50)

    # OpenAI API Key laden
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    # Konfiguration
    config = RAGConfig(
        chunker_type="recursive",
        chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
        embedding_type="sentence_transformers",
        vector_store_type="in_memory",
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    )

    # RAG-System erstellen
    rag = RAGSystem(config)

    # DSGVO-Daten laden
    data_loader = DataLoader()
    dsgvo_text = data_loader.load_dsgvo_document()
    rag.process_documents([dsgvo_text])

    # Interaktive Fragen
    custom_questions = [
        "Was passiert bei einem Datenleck?",
        "Kann ich meine Daten löschen lassen?",
        "Was ist die Datenschutz-Folgenabschätzung?",
        "Welche Sanktionen gibt es bei DSGVO-Verstößen?",
        "Was ist das Recht auf Datenübertragbarkeit?"
    ]

    print("🔍 Benutzerdefinierte DSGVO-Fragen:")
    for question in custom_questions:
        try:
            answer = rag.query(question)
            print(f"\n❓ {question}")
            print(f"💡 {answer}")
        except Exception as e:
            print(f"\n❓ {question}")
            print(f"❌ Fehler: {str(e)}")


def main():
    """
    Hauptfunktion für alle Beispiele.
    """
    print("🚀 RAG-System Beispiele mit DSGVO-Daten")
    print("=" * 60)

    try:
        # Grundlegende Verwendung
        example_basic_usage()

        # Konfigurationsvergleich
        example_comparison()

        # Evaluierung
        example_evaluation()

        # Benutzerdefinierte Fragen
        example_custom_questions()

        print("\n✅ Alle Beispiele erfolgreich ausgeführt!")

    except Exception as e:
        print(f"\n❌ Fehler beim Ausführen der Beispiele: {str(e)}")
        print("Stellen Sie sicher, dass:")
        print("1. Alle Abhängigkeiten installiert sind (pip install -r requirements.txt)")
        print("2. Ein OpenAI API Key in der .env-Datei konfiguriert ist")
        print("3. Die DSGVO-Datei in data/raw/dsgvo.txt vorhanden ist")


if __name__ == "__main__":
    main()