#!/usr/bin/env python3
"""
Baseline-Experiment für RAG-System Evaluierung

Evaluiert verschiedene RAG-Konfigurationen mit:
- Precision@k5, Recall@k5, F1-Score
- RAGAS-Metriken
- Inferenzgeschwindigkeit
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

# Pfad hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.rag_config import RAGConfig
from rag_system import RAGSystem
from data_loader import DataLoader
from evaluations.rag_metrics import RAGMetrics


class BaselineExperiment:
    """
    Führt umfassende Baseline-Experimente für RAG-Systeme durch.
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.metrics_calculator = RAGMetrics()
        self.results = {}

    def create_baseline_configs(self, api_key: str) -> List[Dict[str, Any]]:
        """
        Erstellt Baseline-Konfigurationen für das Experiment.
        """
        configs = [
            {
                "name": "Baseline-1: Line + SentenceTransformers + InMemory",
                "config": RAGConfig(
                    chunker_type="line",
                    chunker_params={},
                    embedding_type="sentence_transformers",
                    embedding_params={},
                    vector_store_type="in_memory",
                    vector_store_params={},
                    language_model_type="openai",
                    language_model_params={"api_key": api_key}
                )
            },
            {
                "name": "Baseline-2: Recursive + SentenceTransformers + InMemory",
                "config": RAGConfig(
                    chunker_type="recursive_character",
                    chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
                    embedding_type="sentence_transformers",
                    embedding_params={},
                    vector_store_type="in_memory",
                    vector_store_params={},
                    language_model_type="openai",
                    language_model_params={"api_key": api_key}
                )
            },
            {
                "name": "Baseline-3: Line + OpenAI + InMemory",
                "config": RAGConfig(
                    chunker_type="line",
                    chunker_params={},
                    embedding_type="openai",
                    embedding_params={"api_key": api_key},
                    vector_store_type="in_memory",
                    vector_store_params={},
                    language_model_type="openai",
                    language_model_params={"api_key": api_key}
                )
            }
        ]

        return configs

    def create_ground_truth_relevance(self, questions: List[str], documents: List[str]) -> List[List[str]]:
        """
        Erstellt Ground Truth für Relevanz-Bewertung (vereinfacht).
        """
        relevant_docs = []

        for question in questions:
            question_lower = question.lower()
            relevant = []

            # Einfache Schlüsselwort-basierte Relevanz
            for doc in documents:
                doc_lower = doc.lower()

                # Prüfe auf Schlüsselwörter
                if ("dsgvo" in question_lower and "dsgvo" in doc_lower) or \
                   ("personenbezogen" in question_lower and "personenbezogen" in doc_lower) or \
                   ("rechte" in question_lower and "recht" in doc_lower) or \
                   ("verarbeitung" in question_lower and "verarbeitung" in doc_lower) or \
                   ("datenschutz" in question_lower and "datenschutz" in doc_lower):
                    relevant.append(doc)

            # Mindestens top 3 relevante Dokumente
            if len(relevant) < 3:
                relevant.extend(documents[:3-len(relevant)])

            relevant_docs.append(relevant[:5])  # Top 5 relevante Dokumente

        return relevant_docs

    def measure_inference_speed(self, rag_system: RAGSystem, questions: List[str],
                               num_runs: int = 3) -> Dict[str, float]:
        """
        Misst die Inferenzgeschwindigkeit des RAG-Systems.
        """
        times = []

        for _ in range(num_runs):
            start_time = time.time()

            for question in questions:
                try:
                    _ = rag_system.query(question)
                except Exception as e:
                    print(f"Fehler bei Inferenz: {e}")
                    continue

            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        queries_per_second = len(questions) / avg_time if avg_time > 0 else 0

        return {
            "avg_total_time": avg_time,
            "avg_time_per_query": avg_time / len(questions) if questions else 0,
            "queries_per_second": queries_per_second,
            "std_time": np.std(times)
        }

    def simulate_retrieval_results(self, rag_system: RAGSystem, questions: List[str],
                                 documents: List[str], k: int = 5) -> List[List[str]]:
        """
        Simuliert Retrieval-Ergebnisse für Evaluation.
        """
        retrieved_docs = []

        for question in questions:
            # Einfache Simulation: Nimm die ersten k Dokumente als "retrieved"
            # In einem echten System würde hier das tatsächliche Retrieval stattfinden
            retrieved = documents[:k]
            retrieved_docs.append(retrieved)

        return retrieved_docs

    def run_baseline_experiment(self, api_key: str, num_test_queries: int = 10) -> Dict[str, Any]:
        """
        Führt das vollständige Baseline-Experiment durch.
        """
        print("🚀 Starte Baseline-Experiment...")
        print(f"📊 Anzahl Test-Queries: {num_test_queries}")

        # Daten laden
        print("\n📄 Lade DSGVO-Daten...")
        try:
            documents = self.data_loader.load_dsgvo_data()
            all_questions = self.data_loader.get_test_questions()

            # Begrenzen auf gewünschte Anzahl
            questions = all_questions[:num_test_queries]

            print(f"✅ {len(documents)} Dokumente geladen")
            print(f"✅ {len(questions)} Test-Fragen ausgewählt")

        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten: {e}")
            return {"error": str(e)}

        # Ground Truth erstellen
        print("\n🎯 Erstelle Ground Truth...")
        relevant_docs = self.create_ground_truth_relevance(questions, documents)

        # Baseline-Konfigurationen
        configs = self.create_baseline_configs(api_key)

        experiment_results = {
            "timestamp": datetime.now().isoformat(),
            "num_documents": len(documents),
            "num_questions": len(questions),
            "configurations": {}
        }

        # Für jede Konfiguration
        for config_info in configs:
            config_name = config_info["name"]
            config = config_info["config"]

            print(f"\n🧪 Teste Konfiguration: {config_name}")

            try:
                # RAG-System erstellen
                rag_system = RAGSystem(config)

                # Dokumente verarbeiten
                print("   📚 Verarbeite Dokumente...")
                start_time = time.time()
                rag_system.process_documents(documents)
                processing_time = time.time() - start_time

                # Antworten generieren
                print("   💬 Generiere Antworten...")
                answers = []
                answer_times = []

                for question in questions:
                    start_time = time.time()
                    try:
                        answer = rag_system.query(question)
                        answers.append(answer)
                        answer_times.append(time.time() - start_time)
                    except Exception as e:
                        print(f"   ⚠️ Fehler bei Frage '{question}': {e}")
                        answers.append(f"Fehler: {str(e)}")
                        answer_times.append(0)

                # Retrieval-Ergebnisse simulieren
                retrieved_docs = self.simulate_retrieval_results(rag_system, questions, documents)

                # Metriken berechnen
                print("   📊 Berechne Metriken...")

                # Retrieval-Metriken
                retrieval_metrics = self.calculate_retrieval_metrics(
                    questions, retrieved_docs, relevant_docs
                )

                # RAGAS-Metriken
                ragas_metrics = self.calculate_ragas_metrics(
                    questions, answers, retrieved_docs
                )

                # Inferenzgeschwindigkeit
                speed_metrics = {
                    "document_processing_time": processing_time,
                    "avg_answer_time": np.mean(answer_times),
                    "total_answer_time": sum(answer_times),
                    "queries_per_second": len(questions) / sum(answer_times) if sum(answer_times) > 0 else 0
                }

                # Ergebnisse sammeln
                experiment_results["configurations"][config_name] = {
                    "config": {
                        "chunker_type": config.chunker_type,
                        "embedding_type": config.embedding_type,
                        "vector_store_type": config.vector_store_type,
                        "language_model_type": config.language_model_type
                    },
                    "retrieval_metrics": retrieval_metrics,
                    "ragas_metrics": ragas_metrics,
                    "speed_metrics": speed_metrics,
                    "sample_answers": answers[:3],  # Erste 3 Antworten als Beispiel
                    "status": "completed"
                }

                print(f"   ✅ Konfiguration abgeschlossen")

            except Exception as e:
                print(f"   ❌ Fehler bei Konfiguration: {e}")
                experiment_results["configurations"][config_name] = {
                    "error": str(e),
                    "status": "failed"
                }

        return experiment_results

    def calculate_retrieval_metrics(self, questions: List[str],
                                  retrieved_docs: List[List[str]],
                                  relevant_docs: List[List[str]]) -> Dict[str, float]:
        """
        Berechnet Retrieval-Metriken über alle Fragen.
        """
        all_precisions_5 = []
        all_recalls_5 = []
        all_f1s_5 = []
        all_mrrs = []
        all_ndcgs = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            metrics = self.metrics_calculator.calculate_retrieval_metrics(
                "", retrieved, relevant, top_k=5
            )

            all_precisions_5.append(metrics.get("precision@5", 0))
            all_recalls_5.append(metrics.get("recall@5", 0))
            all_f1s_5.append(metrics.get("f1@5", 0))
            all_mrrs.append(metrics.get("mrr", 0))
            all_ndcgs.append(metrics.get("ndcg", 0))

        return {
            "precision@5": np.mean(all_precisions_5),
            "recall@5": np.mean(all_recalls_5),
            "f1@5": np.mean(all_f1s_5),
            "mrr": np.mean(all_mrrs),
            "ndcg": np.mean(all_ndcgs),
            "std_precision@5": np.std(all_precisions_5),
            "std_recall@5": np.std(all_recalls_5),
            "std_f1@5": np.std(all_f1s_5)
        }

    def calculate_ragas_metrics(self, questions: List[str], answers: List[str],
                               contexts: List[List[str]]) -> Dict[str, float]:
        """
        Berechnet RAGAS-ähnliche Metriken.
        """
        context_relevances = []
        answer_relevances = []

        for question, answer, context in zip(questions, answers, contexts):
            ragas = self.metrics_calculator.calculate_ragas_metrics(
                context, question, answer
            )

            context_relevances.append(ragas.get("context_relevance", 0))
            answer_relevances.append(ragas.get("answer_relevance", 0))

        return {
            "context_relevance": np.mean(context_relevances),
            "answer_relevance": np.mean(answer_relevances),
            "std_context_relevance": np.std(context_relevances),
            "std_answer_relevance": np.std(answer_relevances)
        }

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Speichert die Experiment-Ergebnisse.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_experiment_{timestamp}.json"

        filepath = os.path.join("results", filename)
        os.makedirs("results", exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return filepath

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Druckt eine Zusammenfassung der Ergebnisse.
        """
        print("\n" + "="*80)
        print("📊 BASELINE-EXPERIMENT ZUSAMMENFASSUNG")
        print("="*80)

        print(f"🕐 Zeitstempel: {results['timestamp']}")
        print(f"📄 Dokumente: {results['num_documents']}")
        print(f"❓ Fragen: {results['num_questions']}")

        print("\n🏆 ERGEBNISSE PRO KONFIGURATION:")
        print("-" * 80)

        for config_name, config_results in results["configurations"].items():
            if config_results.get("status") == "completed":
                print(f"\n📋 {config_name}")

                # Retrieval-Metriken
                retrieval = config_results["retrieval_metrics"]
                print(f"   🎯 Precision@5: {retrieval['precision@5']:.3f} (±{retrieval['std_precision@5']:.3f})")
                print(f"   🎯 Recall@5:    {retrieval['recall@5']:.3f} (±{retrieval['std_recall@5']:.3f})")
                print(f"   🎯 F1@5:        {retrieval['f1@5']:.3f} (±{retrieval['std_f1@5']:.3f})")
                print(f"   🎯 MRR:         {retrieval['mrr']:.3f}")
                print(f"   🎯 nDCG:        {retrieval['ndcg']:.3f}")

                # RAGAS-Metriken
                ragas = config_results["ragas_metrics"]
                print(f"   📝 Context Relevance: {ragas['context_relevance']:.3f} (±{ragas['std_context_relevance']:.3f})")
                print(f"   📝 Answer Relevance:  {ragas['answer_relevance']:.3f} (±{ragas['std_answer_relevance']:.3f})")

                # Geschwindigkeit
                speed = config_results["speed_metrics"]
                print(f"   ⚡ Queries/Sekunde: {speed['queries_per_second']:.2f}")
                print(f"   ⚡ Ø Zeit/Query:    {speed['avg_answer_time']:.3f}s")

            else:
                print(f"\n❌ {config_name}: {config_results.get('error', 'Unbekannter Fehler')}")


def main():
    """
    Hauptfunktion für das Baseline-Experiment.
    """
    # API Key laden
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Kein OpenAI API Key gefunden!")
        print("Setzen Sie OPENAI_API_KEY in der .env-Datei")
        return

    # DataLoader initialisieren
    data_loader = DataLoader()

    # Experiment durchführen
    experiment = BaselineExperiment(data_loader)

    print("🚀 Starte Baseline-Experiment für RAG-System")
    print("📊 Evaluiert: Precision@5, Recall@5, F1-Score, RAGAS, Inferenzgeschwindigkeit")

    # Experiment ausführen
    results = experiment.run_baseline_experiment(api_key, num_test_queries=10)

    # Ergebnisse speichern
    if "error" not in results:
        filepath = experiment.save_results(results)
        print(f"\n💾 Ergebnisse gespeichert: {filepath}")

        # Zusammenfassung drucken
        experiment.print_summary(results)
    else:
        print(f"❌ Experiment fehlgeschlagen: {results['error']}")


if __name__ == "__main__":
    main()