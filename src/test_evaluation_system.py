#!/usr/bin/env python3
"""
Testskript für das Evaluierungssystem.

Testet alle Evaluierungskomponenten einzeln ohne komplexe RAG-Tests.
"""

import os
import sys
import json
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluations import (
    RetrievalEvaluator, GenerationEvaluator,
    PerformanceEvaluator
)


def test_retrieval_evaluator():
    """Testet den RetrievalEvaluator."""
    print("🔍 Teste RetrievalEvaluator...")

    evaluator = RetrievalEvaluator()

    # Test-Daten
    predictions = [
        ["doc1", "doc2", "doc3"],
        ["doc2", "doc4", "doc5"],
        ["doc1", "doc3", "doc6"]
    ]

    ground_truth = [
        ["doc1", "doc2"],
        ["doc2", "doc3"],
        ["doc1", "doc4"]
    ]

    results = evaluator.evaluate(predictions, ground_truth)

    print(f"  ✅ Precision@5: {results['precision@5']:.3f}")
    print(f"  ✅ Recall@5: {results['recall@5']:.3f}")
    print(f"  ✅ F1@5: {results['f1@5']:.3f}")
    print(f"  ✅ MRR: {results['mrr']:.3f}")
    print(f"  ✅ NDCG@5: {results['ndcg@5']:.3f}")

    return results


def test_generation_evaluator():
    """Testet den GenerationEvaluator."""
    print("📝 Teste GenerationEvaluator...")

    evaluator = GenerationEvaluator()

    # Test-Daten
    predictions = [
        "Die maximale Geldbuße beträgt 20 Millionen Euro.",
        "Betroffene Personen haben verschiedene Rechte.",
        "Eine Einwilligung muss freiwillig sein."
    ]

    ground_truth = [
        "Die maximale Geldbuße nach DSGVO beträgt 20 Millionen Euro oder 4% des Jahresumsatzes.",
        "Betroffene haben Rechte auf Auskunft, Berichtigung und Löschung.",
        "Eine Einwilligung muss freiwillig, informiert und eindeutig sein."
    ]

    results = evaluator.evaluate(predictions, ground_truth)

    print(f"  ✅ ROUGE-L: {results['rouge_l']:.3f}")
    print(f"  ✅ BLEU: {results['bleu']:.3f}")
    print(f"  ✅ Exact Match: {results['exact_match']:.3f}")
    print(f"  ✅ Semantic Similarity: {results['semantic_similarity']:.3f}")

    return results


def test_performance_evaluator():
    """Testet den PerformanceEvaluator."""
    print("⚡ Teste PerformanceEvaluator...")

    evaluator = PerformanceEvaluator()

    # Test-Daten
    latencies = [0.5, 0.8, 0.6, 1.2, 0.9]
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]

    results = evaluator.evaluate(
        [], [],  # Dummy-Daten
        latencies=latencies,
        timestamps=timestamps
    )

    print(f"  ✅ Durchschnittliche Latenz: {results['avg_latency']:.3f}s")
    print(f"  ✅ Median Latenz: {results['median_latency']:.3f}s")
    print(f"  ✅ P95 Latenz: {results['p95_latency']:.3f}s")
    print(f"  ✅ Throughput: {results['throughput_qps']:.1f} QPS")

    return results


def test_simple_rag_evaluator():
    """Testet den RAGEvaluator mit einfachen Daten."""
    print("🎯 Teste RAGEvaluator (vereinfacht)...")

    from src.evaluations import RAGEvaluator
    evaluator = RAGEvaluator()

    # Einfache Test-Daten ohne Kategorien
    predictions = [
        {
            "question": "Was ist die maximale Geldbuße?",
            "answer": "Die maximale Geldbuße beträgt 20 Millionen Euro.",
            "retrieved_contexts": [
                {"chunk_id": "1", "text": "Art. 83 DSGVO regelt Geldbußen von bis zu 20 Millionen Euro."},
            ],
            "query_time": 0.8,
            "timestamp": 1234567890
        }
    ]

    ground_truth = [
        {
            "question": "Was ist die maximale Geldbuße?",
            "gold_answer": "Die maximale Geldbuße nach DSGVO beträgt 20 Millionen Euro oder 4% des Jahresumsatzes.",
            "relevant_chunks": ["1"]
        }
    ]

    try:
        results = evaluator.evaluate(predictions, ground_truth)

        print(f"  ✅ RAG Score: {results.get('rag_score', 0.0):.3f}")
        print(f"  ✅ Quality Score: {results.get('quality_score', 0.0):.3f}")
        print(f"  ✅ Efficiency Score: {results.get('efficiency_score', 0.0):.3f}")
        print(f"  ✅ Overall Score: {results.get('overall_score', 0.0):.3f}")
        print(f"  ✅ Faithfulness: {results.get('faithfulness', 0.0):.3f}")
        print(f"  ✅ Groundedness: {results.get('groundedness', 0.0):.3f}")

        return results
    except Exception as e:
        print(f"  ❌ RAGEvaluator Test fehlgeschlagen: {e}")
        return None


def test_component_loader():
    """Testet den erweiterten ComponentLoader."""
    print("🔧 Teste ComponentLoader...")

    from src.core.component_loader import ComponentLoader

    loader = ComponentLoader()

    # Verfügbare Komponenten anzeigen
    available = loader.get_available_components()
    print(f"  ✅ Verfügbare Evaluatoren: {available['evaluators']}")

    # Evaluator laden
    evaluator_config = {"type": "retrieval", "k_values": [1, 3, 5]}
    evaluator = loader.load_evaluator(evaluator_config)

    print(f"  ✅ Evaluator geladen: {evaluator}")

    return evaluator


def main():
    """Hauptfunktion für alle Tests."""
    print("🚀 Starte Evaluierungssystem-Tests...\n")

    test_results = {}

    # Einzelne Evaluatoren testen
    test_results["retrieval"] = test_retrieval_evaluator()
    print()

    test_results["generation"] = test_generation_evaluator()
    print()

    test_results["performance"] = test_performance_evaluator()
    print()

    test_results["rag_simple"] = test_simple_rag_evaluator()
    print()

    # ComponentLoader testen
    test_results["component_loader"] = test_component_loader()
    print()

    # Zusammenfassung
    print("📊 Test-Zusammenfassung:")
    print("=" * 50)

    for test_name, result in test_results.items():
        status = "✅ BESTANDEN" if result is not None else "❌ FEHLGESCHLAGEN"
        print(f"{test_name:20} {status}")

    print("\n🎉 Evaluierungssystem-Tests abgeschlossen!")

    return test_results


if __name__ == "__main__":
    main()