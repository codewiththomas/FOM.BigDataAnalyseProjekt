#!/usr/bin/env python3
"""
RAGAS Evaluierungs-Test für ResearchRAG-System
Testet das vollständige RAG-System mit Generation und RAGAS-Metriken
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Pfad zum src-Verzeichnis hinzufügen
sys.path.insert(0, 'src')

def test_ragas_evaluation():
    print("🚀 Teste ResearchRAG-System mit RAGAS-Evaluierung...")

    # Ergebnis-Verzeichnis erstellen
    results_dir = Path("data/evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Zeitstempel für eindeutige Dateinamen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Imports
        from config.pipeline_configs import get_local_config
        from core.rag_pipeline import RAGPipeline
        from evaluations.rag_evaluator import RAGEvaluator

        print("✅ Module erfolgreich importiert")

        # Lokale Konfiguration mit OpenAI für Generation
        config = get_local_config()
        config_dict = config._config.copy()
        config_dict["language_model"] = {
            "type": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 500
        }

        from config.pipeline_configs import PipelineConfig
        config = PipelineConfig(config_dict)

        print(f"✅ Konfiguration geladen: {config.get_component_types()}")

        # Pipeline erstellen
        print("🔧 Erstelle RAG-Pipeline...")
        pipeline = RAGPipeline(config)
        print("✅ Pipeline erstellt")

        # Test-Dokument laden und indexieren
        print("📄 Lade und indexiere DSGVO-Dokument...")
        dsgvo_path = Path("data/raw/dsgvo.txt")
        if not dsgvo_path.exists():
            print(f"❌ DSGVO-Datei nicht gefunden: {dsgvo_path}")
            return False

        with open(dsgvo_path, 'r', encoding='utf-8') as f:
            dsgvo_text = f.read()

        pipeline.index_documents([dsgvo_text])
        print("✅ Dokument indexiert")

        # Test-Fragen aus QA-Datensatz laden
        qa_pairs_path = Path("data/evaluation/qa_pairs.json")
        if qa_pairs_path.exists():
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            test_questions = qa_data.get("questions", [])[:10]  # Erste 10 Fragen
        else:
            print("❌ QA-Datensatz nicht gefunden")
            return False

        print(f"📋 Teste {len(test_questions)} Fragen mit vollständiger RAG-Pipeline...")

        # RAG-Evaluator erstellen
        rag_evaluator = RAGEvaluator(k_values=[1, 3, 5])

        # Vorhersagen sammeln
        predictions = []
        ground_truth = []

        for i, question_data in enumerate(test_questions, 1):
            question = question_data["question"]
            print(f"\n--- Frage {i}: {question} ---")

            start_time = time.time()

            # Vollständige RAG-Abfrage (Retrieval + Generation)
            try:
                answer = pipeline.query(question)
                query_time = time.time() - start_time

                print(f"✅ Antwort generiert in {query_time:.2f}s")
                print(f"📝 Antwort: {str(answer)[:200]}...")

                # Retrieval-Kontext für RAGAS sammeln
                question_embedding = pipeline.embedding.embed_texts([question])
                similar_chunks = pipeline.vector_store.similarity_search(
                    question_embedding[0], top_k=5
                )

                # Prediction für RAGAS formatieren
                prediction = {
                    "question": question,
                    "answer": str(answer),
                    "retrieved_contexts": [
                        {
                            "chunk_id": chunk.get('id', f'chunk_{j}'),
                            "text": chunk['text'],
                            "score": chunk.get('score', 0.0),
                            "metadata": chunk.get('metadata', {})
                        }
                        for j, chunk in enumerate(similar_chunks)
                    ],
                    "query_time": query_time,
                    "timestamp": time.time()
                }
                predictions.append(prediction)

                # Ground Truth für RAGAS formatieren
                gt = {
                    "question": question,
                    "gold_answer": question_data.get("gold_answer", ""),
                    "relevant_chunks": question_data.get("relevant_chunks", []),
                    "category": question_data.get("category", "unknown"),
                    "difficulty": question_data.get("difficulty", "unknown")
                }
                ground_truth.append(gt)

            except Exception as e:
                print(f"❌ Fehler bei Frage {i}: {e}")
                continue

        print(f"\n🧪 Führe RAGAS-Evaluierung durch...")

        # RAGAS-Evaluierung durchführen
        ragas_results = rag_evaluator.evaluate(predictions, ground_truth)

        # Ergebnisse anzeigen
        print("\n📊 RAGAS-Evaluierungsergebnisse:")
        print("=" * 50)

        # Retrieval-Metriken
        print("\n🔍 Retrieval-Metriken:")
        for k in [1, 3, 5]:
            precision = ragas_results.get(f"retrieval_precision@{k}", 0)
            recall = ragas_results.get(f"retrieval_recall@{k}", 0)
            f1 = ragas_results.get(f"retrieval_f1@{k}", 0)
            print(f"  Precision@{k}: {precision:.3f}")
            print(f"  Recall@{k}: {recall:.3f}")
            print(f"  F1@{k}: {f1:.3f}")

        # Generation-Metriken
        print("\n✏️ Generation-Metriken:")
        rouge_l = ragas_results.get("generation_rouge_l", 0)
        bleu = ragas_results.get("generation_bleu", 0)
        exact_match = ragas_results.get("generation_exact_match", 0)
        semantic_sim = ragas_results.get("generation_semantic_similarity", 0)
        print(f"  ROUGE-L: {rouge_l:.3f}")
        print(f"  BLEU: {bleu:.3f}")
        print(f"  Exact Match: {exact_match:.3f}")
        print(f"  Semantic Similarity: {semantic_sim:.3f}")

        # RAGAS-spezifische Metriken
        print("\n🎯 RAGAS-spezifische Metriken:")
        faithfulness = ragas_results.get("faithfulness", 0)
        groundedness = ragas_results.get("groundedness", 0)
        answer_relevance = ragas_results.get("answer_relevance", 0)
        context_precision = ragas_results.get("context_precision", 0)
        context_recall = ragas_results.get("context_recall", 0)
        print(f"  Faithfulness: {faithfulness:.3f}")
        print(f"  Groundedness: {groundedness:.3f}")
        print(f"  Answer Relevance: {answer_relevance:.3f}")
        print(f"  Context Precision: {context_precision:.3f}")
        print(f"  Context Recall: {context_recall:.3f}")

        # Performance-Metriken
        print("\n⚡ Performance-Metriken:")
        avg_latency = ragas_results.get("performance_avg_latency", 0)
        throughput = ragas_results.get("performance_throughput_qps", 0)
        print(f"  Durchschnittliche Latenz: {avg_latency:.3f}s")
        print(f"  Durchsatz: {throughput:.2f} Fragen/s")

        # Kombinierte Scores
        print("\n🏆 Kombinierte Scores:")
        rag_score = ragas_results.get("rag_score", 0)
        quality_score = ragas_results.get("quality_score", 0)
        efficiency_score = ragas_results.get("efficiency_score", 0)
        overall_score = ragas_results.get("overall_score", 0)
        print(f"  RAG Score: {rag_score:.3f}")
        print(f"  Quality Score: {quality_score:.3f}")
        print(f"  Efficiency Score: {efficiency_score:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")

        # Kategorien-Analyse
        if "category_analysis" in ragas_results:
            print("\n📋 Kategorien-Analyse:")
            for category, stats in ragas_results["category_analysis"].items():
                print(f"  {category}: {stats['count']} Fragen, Ø{stats['avg_query_time']:.2f}s")

        # Vollständige Ergebnisse speichern
        full_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "ragas_evaluation",
                "system_config": config.get_component_types(),
                "questions_tested": len(predictions),
                "evaluation_framework": "RAGAS"
            },
            "ragas_metrics": ragas_results,
            "predictions": predictions,
            "ground_truth": ground_truth
        }

        result_file = results_dir / f"ragas_evaluation_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Vollständige RAGAS-Ergebnisse gespeichert in: {result_file}")

        # RAGAS-Bericht erstellen
        report_file = results_dir / f"ragas_report_{timestamp}.md"
        create_ragas_report(full_results, report_file)
        print(f"📄 RAGAS-Bericht erstellt: {report_file}")

        return True

    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_ragas_report(results, report_file):
    """Erstellt einen detaillierten RAGAS-Bericht."""

    ragas_metrics = results["ragas_metrics"]
    metadata = results["metadata"]

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# RAGAS Evaluierungs-Bericht\n\n")
        f.write(f"**Zeitstempel:** {metadata['timestamp']}\n")
        f.write(f"**Fragen getestet:** {metadata['questions_tested']}\n")
        f.write(f"**Evaluierungs-Framework:** {metadata['evaluation_framework']}\n\n")

        # System-Konfiguration
        f.write("## System-Konfiguration\n\n")
        config = metadata['system_config']
        for component, type_name in config.items():
            f.write(f"- **{component.title()}:** {type_name}\n")
        f.write("\n")

        # RAGAS-Metriken
        f.write("## RAGAS-Metriken\n\n")

        # Retrieval
        f.write("### 🔍 Retrieval-Performance\n\n")
        f.write("| Metrik | @1 | @3 | @5 |\n")
        f.write("|--------|----|----|----|\n")

        for metric in ["precision", "recall", "f1"]:
            f.write(f"| {metric.title()} |")
            for k in [1, 3, 5]:
                value = ragas_metrics.get(f"retrieval_{metric}@{k}", 0)
                f.write(f" {value:.3f} |")
            f.write("\n")
        f.write("\n")

        # Generation
        f.write("### ✏️ Generation-Performance\n\n")
        f.write("| Metrik | Score |\n")
        f.write("|--------|-------|\n")
        gen_metrics = ["rouge_l", "bleu", "exact_match", "semantic_similarity"]
        for metric in gen_metrics:
            value = ragas_metrics.get(f"generation_{metric}", 0)
            f.write(f"| {metric.replace('_', ' ').title()} | {value:.3f} |\n")
        f.write("\n")

        # RAGAS-spezifisch
        f.write("### 🎯 RAGAS-spezifische Metriken\n\n")
        f.write("| Metrik | Score | Beschreibung |\n")
        f.write("|--------|-------|-------------|\n")

        ragas_specific = {
            "faithfulness": "Treue zur Quelle",
            "groundedness": "Verankerung in Kontexten",
            "answer_relevance": "Antwort-Relevanz",
            "context_precision": "Kontext-Präzision",
            "context_recall": "Kontext-Recall"
        }

        for metric, desc in ragas_specific.items():
            value = ragas_metrics.get(metric, 0)
            f.write(f"| {metric.replace('_', ' ').title()} | {value:.3f} | {desc} |\n")
        f.write("\n")

        # Performance
        f.write("### ⚡ Performance-Metriken\n\n")
        avg_latency = ragas_metrics.get("performance_avg_latency", 0)
        throughput = ragas_metrics.get("performance_throughput_qps", 0)
        f.write(f"- **Durchschnittliche Latenz:** {avg_latency:.3f}s\n")
        f.write(f"- **Durchsatz:** {throughput:.2f} Fragen/s\n\n")

        # Kombinierte Scores
        f.write("### 🏆 Kombinierte Bewertung\n\n")
        f.write("| Score | Wert | Gewichtung |\n")
        f.write("|-------|------|------------|\n")

        combined_scores = {
            "rag_score": "30% Retrieval + 30% Generation + 40% Faithfulness",
            "quality_score": "50% Groundedness + 50% Answer Relevance",
            "efficiency_score": "Durchsatz / Latenz",
            "overall_score": "40% RAG + 40% Quality + 20% Efficiency"
        }

        for score, desc in combined_scores.items():
            value = ragas_metrics.get(score, 0)
            f.write(f"| {score.replace('_', ' ').title()} | {value:.3f} | {desc} |\n")
        f.write("\n")

        # Kategorien-Analyse
        if "category_analysis" in ragas_metrics:
            f.write("### 📋 Kategorien-Analyse\n\n")
            f.write("| Kategorie | Anzahl | Ø Latenz | Ø Kontexte |\n")
            f.write("|-----------|--------|----------|------------|\n")

            for category, stats in ragas_metrics["category_analysis"].items():
                f.write(f"| {category} | {stats['count']} | {stats['avg_query_time']:.2f}s | {stats['avg_contexts_retrieved']:.1f} |\n")
            f.write("\n")

        # Interpretation
        f.write("## 📊 Interpretation\n\n")
        overall = ragas_metrics.get("overall_score", 0)

        if overall >= 0.8:
            f.write("✅ **Exzellente Performance** - Das System zeigt sehr gute Ergebnisse in allen Bereichen.\n\n")
        elif overall >= 0.6:
            f.write("🟡 **Gute Performance** - Das System funktioniert gut, aber es gibt Verbesserungspotenzial.\n\n")
        elif overall >= 0.4:
            f.write("🟠 **Mittelmäßige Performance** - Das System benötigt Optimierungen.\n\n")
        else:
            f.write("🔴 **Schwache Performance** - Das System benötigt grundlegende Verbesserungen.\n\n")

        # Empfehlungen
        f.write("## 💡 Empfehlungen\n\n")

        faithfulness = ragas_metrics.get("faithfulness", 0)
        context_precision = ragas_metrics.get("context_precision", 0)
        generation_rouge = ragas_metrics.get("generation_rouge_l", 0)

        if faithfulness < 0.5:
            f.write("- **Faithfulness verbessern:** Überprüfe die Prompt-Gestaltung für bessere Quelltreue\n")
        if context_precision < 0.5:
            f.write("- **Retrieval optimieren:** Verbessere Chunking-Strategie oder Embedding-Modell\n")
        if generation_rouge < 0.3:
            f.write("- **Generation verbessern:** Experimentiere mit anderen LLM-Modellen oder Parametern\n")

        f.write("\n---\n\n")
        f.write("*Generiert mit ResearchRAG RAGAS-Evaluierung*\n")

if __name__ == "__main__":
    success = test_ragas_evaluation()
    if success:
        print("\n🎉 RAGAS-Evaluierung erfolgreich abgeschlossen!")
        print("\nDie Ergebnisse wurden in data/evaluation/results/ gespeichert.")
    else:
        print("\n❌ RAGAS-Evaluierung fehlgeschlagen!")
        sys.exit(1)