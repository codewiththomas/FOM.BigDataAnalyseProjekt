from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from config import RAGConfig
from evaluations.rag_metrics import RAGMetrics
from data_loader import DataLoader
from rag_system import RAGSystem


class ComprehensiveExperimentRunner:
    """
    Umfassender Experiment Runner für verschiedene RAG-Konfigurationen.
    """

    def __init__(self, results_directory: str = "experiment_results"):
        self.results_directory = results_directory
        self.metrics = RAGMetrics()
        self.data_loader = DataLoader()

        # Erstelle Results-Verzeichnis
        os.makedirs(results_directory, exist_ok=True)

    def run_comprehensive_experiment(self,
                                   configs: List[RAGConfig],
                                   test_questions: Optional[List[str]] = None,
                                   use_dsgvo: bool = True) -> Dict[str, Any]:
        """
        Führt umfassende Experimente mit verschiedenen Konfigurationen durch.
        """
        results = {
            "experiment_timestamp": datetime.now().isoformat(),
            "configurations": [],
            "overall_results": {}
        }

        # Lade Testdaten
        if use_dsgvo:
            dsgvo_text = self.data_loader.load_dsgvo_document()
            test_data = [dsgvo_text]
            print(f"DSGVO-Dokument geladen: {len(dsgvo_text)} Zeichen")
        else:
            test_data = [
                "Big Data Analyse ist ein Prozess zur Untersuchung großer Datenmengen.",
                "Machine Learning ist ein Teilgebiet der künstlichen Intelligenz.",
                "Datenverarbeitung umfasst die systematische Analyse von Informationen.",
                "Algorithmen sind Schritt-für-Schritt-Anweisungen zur Problemlösung."
            ]

        # Standard-Testfragen
        if test_questions is None:
            test_questions = self.data_loader.load_test_questions()

        # Führe Experimente für jede Konfiguration durch
        for i, config in enumerate(configs):
            print(f"\n🚀 Experiment {i+1}/{len(configs)}: {config.chunker_type} + {config.embedding_type} + {config.vector_store_type} + {config.language_model_type}")

            try:
                config_result = self._run_single_config_experiment(config, test_data, test_questions)
                results["configurations"].append(config_result)

            except Exception as e:
                print(f"❌ Fehler in Experiment {i+1}: {str(e)}")
                results["configurations"].append({
                    "config": config.__dict__,
                    "error": str(e),
                    "status": "failed"
                })

        # Berechne Gesamtergebnisse
        results["overall_results"] = self._calculate_overall_results(results["configurations"])

        # Speichere Ergebnisse
        self._save_experiment_results(results)

        return results

    def _run_single_config_experiment(self,
                                    config: RAGConfig,
                                    test_data: List[str],
                                    test_questions: List[str]) -> Dict[str, Any]:
        """
        Führt ein einzelnes Experiment mit einer Konfiguration durch.
        """
        # RAG-System erstellen
        rag_system = RAGSystem(config)

        # Dokumente verarbeiten
        print(f"  📄 Verarbeite {len(test_data)} Dokumente...")
        rag_system.process_documents(test_data)

        # Testfragen ausführen
        print(f"  ❓ Teste {len(test_questions)} Fragen...")
        all_responses = []
        all_contexts = []

        for question in test_questions:
            try:
                response = rag_system.query(question)
                all_responses.append(response)

                # Kontext extrahieren (falls verfügbar)
                context = self._extract_context_from_response(response)
                all_contexts.append(context)

            except Exception as e:
                print(f"    ⚠️ Fehler bei Frage '{question}': {str(e)}")
                all_responses.append(f"Fehler: {str(e)}")
                all_contexts.append("")

        # Metriken berechnen
        print(f"  📊 Berechne Metriken...")
        metrics = self._calculate_config_metrics(test_questions, all_responses, all_contexts)

        return {
            "config": config.__dict__,
            "test_data_count": len(test_data),
            "test_questions_count": len(test_questions),
            "responses": all_responses,
            "metrics": metrics,
            "status": "completed"
        }

    def _extract_context_from_response(self, response: str) -> str:
        """
        Extrahiert Kontext aus einer RAG-Antwort (vereinfacht).
        """
        # Einfache Extraktion - in der Praxis würde man den tatsächlichen Kontext verwenden
        return response[:200] if response else ""

    def _calculate_config_metrics(self,
                                questions: List[str],
                                responses: List[str],
                                contexts: List[str]) -> Dict[str, Any]:
        """
        Berechnet Metriken für eine Konfiguration.
        """
        metrics = {
            "response_lengths": [len(response) for response in responses],
            "avg_response_length": sum(len(response) for response in responses) / len(responses) if responses else 0,
            "successful_responses": len([r for r in responses if not r.startswith("Fehler")]),
            "error_rate": len([r for r in responses if r.startswith("Fehler")]) / len(responses) if responses else 0
        }

        # Einfache Relevanz-Metriken
        relevance_scores = []
        for question, response in zip(questions, responses):
            if not response.startswith("Fehler"):
                relevance = self._calculate_simple_relevance(question, response)
                relevance_scores.append(relevance)

        metrics["avg_relevance"] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        return metrics

    def _calculate_simple_relevance(self, question: str, response: str) -> float:
        """
        Berechnet eine einfache Relevanz-Metrik.
        """
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())

        if not question_words:
            return 0.0

        overlap = len(question_words.intersection(response_words))
        return overlap / len(question_words)

    def _calculate_overall_results(self, config_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Berechnet Gesamtergebnisse über alle Konfigurationen.
        """
        successful_configs = [r for r in config_results if r.get("status") == "completed"]

        if not successful_configs:
            return {"error": "Keine erfolgreichen Konfigurationen"}

        # Beste Konfiguration finden
        best_config = max(successful_configs,
                         key=lambda x: x.get("metrics", {}).get("avg_relevance", 0))

        # Durchschnittliche Metriken
        avg_relevance = sum(c.get("metrics", {}).get("avg_relevance", 0) for c in successful_configs) / len(successful_configs)
        avg_response_length = sum(c.get("metrics", {}).get("avg_response_length", 0) for c in successful_configs) / len(successful_configs)

        return {
            "total_configurations": len(config_results),
            "successful_configurations": len(successful_configs),
            "best_configuration": best_config.get("config"),
            "avg_relevance": avg_relevance,
            "avg_response_length": avg_response_length
        }

    def _save_experiment_results(self, results: Dict[str, Any]) -> None:
        """
        Speichert Experiment-Ergebnisse.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.json"
        filepath = os.path.join(self.results_directory, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Ergebnisse gespeichert: {filepath}")

    def create_comparison_report(self, results: Dict[str, Any]) -> str:
        """
        Erstellt einen Vergleichsbericht der Experimente.
        """
        report = []
        report.append("# RAG-System Experiment Vergleichsbericht")
        report.append(f"**Datum:** {results.get('experiment_timestamp', 'Unbekannt')}")
        report.append(f"**Anzahl Konfigurationen:** {len(results.get('configurations', []))}")
        report.append("")

        # Konfigurationen vergleichen
        successful_configs = [c for c in results.get("configurations", []) if c.get("status") == "completed"]

        if successful_configs:
            report.append("## Beste Konfigurationen")
            report.append("")

            # Sortiere nach Relevanz
            sorted_configs = sorted(successful_configs,
                                  key=lambda x: x.get("metrics", {}).get("avg_relevance", 0),
                                  reverse=True)

            for i, config in enumerate(sorted_configs[:5]):  # Top 5
                metrics = config.get("metrics", {})
                report.append(f"### {i+1}. {config['config']['chunker_type']} + {config['config']['embedding_type']} + {config['config']['vector_store_type']} + {config['config']['language_model_type']}")
                report.append(f"- **Durchschnittliche Relevanz:** {metrics.get('avg_relevance', 0):.3f}")
                report.append(f"- **Durchschnittliche Antwortlänge:** {metrics.get('avg_response_length', 0):.1f} Zeichen")
                report.append(f"- **Erfolgsrate:** {metrics.get('successful_responses', 0)}/{config.get('test_questions_count', 0)}")
                report.append("")

        # Gesamtergebnisse
        overall = results.get("overall_results", {})
        report.append("## Gesamtergebnisse")
        report.append(f"- **Erfolgreiche Konfigurationen:** {overall.get('successful_configurations', 0)}/{overall.get('total_configurations', 0)}")
        report.append(f"- **Durchschnittliche Relevanz:** {overall.get('avg_relevance', 0):.3f}")
        report.append(f"- **Durchschnittliche Antwortlänge:** {overall.get('avg_response_length', 0):.1f} Zeichen")

        return "\n".join(report)