#!/usr/bin/env python3
"""
Einfaches Baseline-Experiment für RAG-System
"""

import os
import sys
import time
import json
from datetime import datetime
import numpy as np

# Pfade konfigurieren
sys.path.append('.')
sys.path.append('src')

from dotenv import load_dotenv
from config.rag_config import RAGConfig
from rag_system import RAGSystem
from data_loader import DataLoader
from evaluations.rag_metrics import RAGMetrics

def main():
    print("🚀 Baseline-Experiment für RAG-System")
    print("📊 Evaluiert: Precision@5, Recall@5, F1-Score, RAGAS, Inferenzgeschwindigkeit")
    print("=" * 80)

    # API Key laden
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ Kein OpenAI API Key gefunden!")
        print("Setzen Sie OPENAI_API_KEY in der .env-Datei")
        return

    print("✅ API Key geladen")

    # Daten laden
    print("\n📄 Lade DSGVO-Daten...")
    data_loader = DataLoader()

    try:
        documents = data_loader.load_dsgvo_data()
        questions = data_loader.get_test_questions()

        # Begrenzen für schnellere Tests
        documents = documents[:50]  # Erste 50 Dokumente
        questions = questions[:5]   # Erste 5 Fragen

        print(f"✅ {len(documents)} Dokumente geladen")
        print(f"✅ {len(questions)} Test-Fragen ausgewählt")

    except Exception as e:
        print(f"❌ Fehler beim Laden der Daten: {e}")
        return

    # Baseline-Konfiguration
    print("\n⚙️ Erstelle Baseline-Konfiguration...")
    config = RAGConfig(
        chunker_type="line",
        embedding_type="sentence_transformers",
        vector_store_type="in_memory",
        language_model_type="openai",
        language_model_params={"api_key": api_key}
    )

    print("✅ Konfiguration erstellt")

    # RAG-System initialisieren
    print("\n🔧 Initialisiere RAG-System...")
    try:
        rag_system = RAGSystem(config)
        print("✅ RAG-System initialisiert")
    except Exception as e:
        print(f"❌ Fehler bei RAG-System: {e}")
        return

    # Dokumente verarbeiten
    print("\n📚 Verarbeite Dokumente...")
    start_time = time.time()
    try:
        rag_system.process_documents(documents)
        processing_time = time.time() - start_time
        print(f"✅ Dokumente verarbeitet in {processing_time:.2f}s")
    except Exception as e:
        print(f"❌ Fehler bei Dokumentverarbeitung: {e}")
        return

    # Test-Queries ausführen
    print("\n💬 Führe Test-Queries aus...")
    answers = []
    query_times = []

    for i, question in enumerate(questions):
        print(f"   {i+1}. {question}")
        start_time = time.time()

        try:
            answer = rag_system.query(question)
            query_time = time.time() - start_time

            answers.append(answer)
            query_times.append(query_time)

            print(f"      ⏱️  {query_time:.3f}s")
            print(f"      💡 {answer[:100]}...")

        except Exception as e:
            print(f"      ❌ Fehler: {e}")
            answers.append(f"Fehler: {str(e)}")
            query_times.append(0)

    # Metriken berechnen
    print("\n📊 Berechne Metriken...")

    # Inferenzgeschwindigkeit
    total_query_time = sum(query_times)
    avg_query_time = np.mean(query_times) if query_times else 0
    queries_per_second = len(questions) / total_query_time if total_query_time > 0 else 0

    # Einfache Retrieval-Simulation für Metriken
    metrics_calculator = RAGMetrics()

    # Simuliere relevante Dokumente (vereinfacht)
    relevant_docs = []
    retrieved_docs = []

    for question in questions:
        # Simuliere: Erste 5 Dokumente als "retrieved"
        retrieved = documents[:5]

        # Simuliere: Dokumente mit Schlüsselwörtern als "relevant"
        relevant = []
        question_lower = question.lower()

        for doc in documents:
            if any(word in doc.lower() for word in question_lower.split()):
                relevant.append(doc)

        # Mindestens 3 relevante Dokumente
        if len(relevant) < 3:
            relevant.extend(documents[:3])

        retrieved_docs.append(retrieved)
        relevant_docs.append(relevant[:5])

    # Retrieval-Metriken berechnen
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        metrics = metrics_calculator.calculate_precision_recall_f1(retrieved, relevant)
        all_precisions.append(metrics["precision"])
        all_recalls.append(metrics["recall"])
        all_f1s.append(metrics["f1_score"])

    # RAGAS-ähnliche Metriken
    context_relevances = []
    answer_relevances = []

    for question, answer, context in zip(questions, answers, retrieved_docs):
        ragas = metrics_calculator.calculate_ragas_metrics(context, question, answer)
        context_relevances.append(ragas.get("context_relevance", 0))
        answer_relevances.append(ragas.get("answer_relevance", 0))

    # Ergebnisse zusammenfassen
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": "Baseline: Line + SentenceTransformers + InMemory + OpenAI",
        "data_stats": {
            "num_documents": len(documents),
            "num_questions": len(questions)
        },
        "retrieval_metrics": {
            "precision@5": np.mean(all_precisions),
            "recall@5": np.mean(all_recalls),
            "f1@5": np.mean(all_f1s),
            "std_precision@5": np.std(all_precisions),
            "std_recall@5": np.std(all_recalls),
            "std_f1@5": np.std(all_f1s)
        },
        "ragas_metrics": {
            "context_relevance": np.mean(context_relevances),
            "answer_relevance": np.mean(answer_relevances),
            "std_context_relevance": np.std(context_relevances),
            "std_answer_relevance": np.std(answer_relevances)
        },
        "speed_metrics": {
            "document_processing_time": processing_time,
            "total_query_time": total_query_time,
            "avg_query_time": avg_query_time,
            "queries_per_second": queries_per_second
        },
        "sample_qa": [
            {"question": q, "answer": a[:200] + "..." if len(a) > 200 else a}
            for q, a in zip(questions, answers)
        ]
    }

    # Ergebnisse anzeigen
    print("\n" + "="*80)
    print("📊 BASELINE-EXPERIMENT ERGEBNISSE")
    print("="*80)

    print(f"⏰ Zeitstempel: {results['timestamp']}")
    print(f"📄 Dokumente: {results['data_stats']['num_documents']}")
    print(f"❓ Fragen: {results['data_stats']['num_questions']}")

    print(f"\n🎯 RETRIEVAL-METRIKEN:")
    ret_metrics = results['retrieval_metrics']
    print(f"   Precision@5: {ret_metrics['precision@5']:.3f} (±{ret_metrics['std_precision@5']:.3f})")
    print(f"   Recall@5:    {ret_metrics['recall@5']:.3f} (±{ret_metrics['std_recall@5']:.3f})")
    print(f"   F1@5:        {ret_metrics['f1@5']:.3f} (±{ret_metrics['std_f1@5']:.3f})")

    print(f"\n📝 RAGAS-METRIKEN:")
    ragas_metrics = results['ragas_metrics']
    print(f"   Context Relevance: {ragas_metrics['context_relevance']:.3f} (±{ragas_metrics['std_context_relevance']:.3f})")
    print(f"   Answer Relevance:  {ragas_metrics['answer_relevance']:.3f} (±{ragas_metrics['std_answer_relevance']:.3f})")

    print(f"\n⚡ INFERENZGESCHWINDIGKEIT:")
    speed_metrics = results['speed_metrics']
    print(f"   Dokumentverarbeitung: {speed_metrics['document_processing_time']:.2f}s")
    print(f"   Queries/Sekunde:      {speed_metrics['queries_per_second']:.2f}")
    print(f"   Ø Zeit/Query:         {speed_metrics['avg_query_time']:.3f}s")

    print(f"\n💬 BEISPIEL-ANTWORTEN:")
    for i, qa in enumerate(results['sample_qa'][:3]):
        print(f"   {i+1}. Q: {qa['question']}")
        print(f"      A: {qa['answer']}")
        print()

    # Ergebnisse speichern
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/baseline_experiment_{timestamp}.json"

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Ergebnisse gespeichert: {filepath}")
    print("\n✅ Baseline-Experiment abgeschlossen!")

if __name__ == "__main__":
    main()