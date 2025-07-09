from typing import List, Dict, Any, Optional, Callable
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from config.pipeline_configs import PipelineConfig
from evaluations import RAGEvaluator
from core.rag_pipeline import RAGPipeline


class ExperimentRunner:
    """
    Führt systematische Experimente mit verschiedenen RAG-Konfigurationen durch.

    Ermöglicht den Vergleich verschiedener Komponenten und Konfigurationen
    mit standardisierten Evaluierungsmetriken.
    """

    def __init__(self, output_dir: str = "data/evaluation/results"):
        """
        Initialisiert den Experiment Runner.

        Args:
            output_dir: Verzeichnis für Experiment-Ergebnisse
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = RAGEvaluator()
        self.experiment_history = []

    def run_experiment(self, config: PipelineConfig, qa_pairs: List[Dict[str, Any]],
                      experiment_name: str, documents: List[str] = None) -> Dict[str, Any]:
        """
        Führt ein einzelnes Experiment durch.

        Args:
            config: Pipeline-Konfiguration
            qa_pairs: QA-Datensatz für die Evaluierung
            experiment_name: Name des Experiments
            documents: Dokumente für die Indexierung (optional)

        Returns:
            Dictionary mit Experiment-Ergebnissen
        """
        print(f"🧪 Starte Experiment: {experiment_name}")
        start_time = time.time()

        # Pipeline erstellen
        pipeline = RAGPipeline(config)

        # Dokumente indexieren
        if documents:
            print("📄 Indexiere Dokumente...")
            indexing_stats = pipeline.index_documents(documents, show_progress=False)
        else:
            # Standarddokument laden
            dsgvo_file = "data/raw/dsgvo.txt"
            if Path(dsgvo_file).exists():
                documents = pipeline.load_documents_from_file(dsgvo_file)
                indexing_stats = pipeline.index_documents(documents, show_progress=False)
            else:
                raise FileNotFoundError(f"Dokument nicht gefunden: {dsgvo_file}")

        # Queries ausführen
        print("❓ Führe Queries aus...")
        predictions = []
        for qa_pair in qa_pairs:
            question = qa_pair["question"]
            result = pipeline.query(question, return_context=True)

            # Struktur für Evaluierung anpassen
            prediction = {
                "question": question,
                "answer": result["answer"],
                "retrieved_contexts": result.get("retrieved_contexts", []),
                "query_time": result["query_time"],
                "timestamp": time.time(),
                "metadata": result.get("metadata", {})
            }
            predictions.append(prediction)

        # Evaluierung durchführen
        print("📊 Evaluiere Ergebnisse...")
        evaluation_results = self.evaluator.evaluate_with_metadata(
            predictions, qa_pairs,
            metadata={
                "experiment_name": experiment_name,
                "config": config.to_dict(),
                "indexing_stats": indexing_stats
            }
        )

        # Experiment-Ergebnisse zusammenstellen
        experiment_results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - start_time,
            "config": config.to_dict(),
            "indexing_stats": indexing_stats,
            "num_queries": len(qa_pairs),
            "evaluation_results": evaluation_results,
            "predictions": predictions  # Für detaillierte Analyse
        }

        # Ergebnisse speichern
        self._save_experiment_results(experiment_results)

        # Zur Historie hinzufügen
        self.experiment_history.append(experiment_results)

        print(f"✅ Experiment abgeschlossen in {experiment_results['duration']:.2f}s")
        print(f"📈 RAG Score: {evaluation_results.get('rag_score', 0.0):.3f}")

        return experiment_results

    def compare_configurations(self, configs: List[Dict[str, Any]],
                             qa_pairs: List[Dict[str, Any]],
                             experiment_name: str = "config_comparison") -> Dict[str, Any]:
        """
        Vergleicht mehrere Konfigurationen systematisch.

        Args:
            configs: Liste von Konfigurationen mit Namen
                [{"name": "config1", "config": PipelineConfig}, ...]
            qa_pairs: QA-Datensatz für die Evaluierung
            experiment_name: Name des Vergleichs-Experiments

        Returns:
            Dictionary mit Vergleichsergebnissen
        """
        print(f"🔬 Starte Konfigurationsvergleich: {experiment_name}")

        comparison_results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "num_configs": len(configs),
            "num_queries": len(qa_pairs),
            "results": {},
            "comparison": {},
            "ranking": []
        }

        # Führe Experimente für alle Konfigurationen durch
        for config_info in configs:
            config_name = config_info["name"]
            config = config_info["config"]

            print(f"\n🔧 Teste Konfiguration: {config_name}")

            # Experiment durchführen
            experiment_result = self.run_experiment(
                config, qa_pairs, f"{experiment_name}_{config_name}"
            )

            comparison_results["results"][config_name] = experiment_result

        # Vergleichsanalyse durchführen
        comparison_results["comparison"] = self._analyze_comparison(comparison_results["results"])
        comparison_results["ranking"] = self._rank_configurations(comparison_results["results"])

        # Vergleichsergebnisse speichern
        self._save_comparison_results(comparison_results)

        print(f"\n🏆 Beste Konfiguration: {comparison_results['ranking'][0]['name']}")

        return comparison_results

    def run_ablation_study(self, base_config: PipelineConfig,
                          component_variants: Dict[str, List[Dict[str, Any]]],
                          qa_pairs: List[Dict[str, Any]],
                          study_name: str = "ablation_study") -> Dict[str, Any]:
        """
        Führt eine Ablationsstudie durch.

        Args:
            base_config: Basis-Konfiguration
            component_variants: Komponenten-Varianten
                {"chunker": [{"type": "line_chunker"}, {"type": "recursive_chunker"}]}
            qa_pairs: QA-Datensatz
            study_name: Name der Studie

        Returns:
            Dictionary mit Ablationsstudie-Ergebnissen
        """
        print(f"🔍 Starte Ablationsstudie: {study_name}")

        configs_to_test = []

        # Baseline-Konfiguration
        configs_to_test.append({
            "name": "baseline",
            "config": base_config
        })

        # Varianten für jede Komponente
        for component_type, variants in component_variants.items():
            for variant in variants:
                # Kopiere Basis-Konfiguration
                variant_config = PipelineConfig(base_config.to_dict())

                # Ändere spezifische Komponente
                if component_type == "chunker":
                    variant_config.config["chunker"].update(variant)
                elif component_type == "embedding":
                    variant_config.config["embedding"].update(variant)
                elif component_type == "vector_store":
                    variant_config.config["vector_store"].update(variant)
                elif component_type == "language_model":
                    variant_config.config["language_model"].update(variant)

                variant_name = f"{component_type}_{variant.get('type', 'variant')}"
                configs_to_test.append({
                    "name": variant_name,
                    "config": variant_config
                })

        # Vergleiche alle Konfigurationen
        return self.compare_configurations(configs_to_test, qa_pairs, study_name)

    def benchmark_performance(self, config: PipelineConfig,
                             queries: List[str],
                             benchmark_name: str = "performance_benchmark") -> Dict[str, Any]:
        """
        Führt ein Performance-Benchmark durch.

        Args:
            config: Pipeline-Konfiguration
            queries: Liste von Test-Queries
            benchmark_name: Name des Benchmarks

        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        print(f"⚡ Starte Performance-Benchmark: {benchmark_name}")

        # Pipeline erstellen
        pipeline = RAGPipeline(config)

        # Dokumente indexieren
        dsgvo_file = "data/raw/dsgvo.txt"
        if Path(dsgvo_file).exists():
            documents = pipeline.load_documents_from_file(dsgvo_file)
            pipeline.index_documents(documents, show_progress=False)

        # Performance-Evaluator verwenden
        from evaluations import PerformanceEvaluator
        performance_evaluator = PerformanceEvaluator()

        # Benchmark durchführen
        benchmark_results = performance_evaluator.benchmark_pipeline(
            lambda q: pipeline.query(q), queries
        )

        # Ergebnisse erweitern
        benchmark_results.update({
            "benchmark_name": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "num_queries": len(queries),
            "system_info": performance_evaluator.get_system_info()
        })

        # Ergebnisse speichern
        self._save_benchmark_results(benchmark_results)

        print(f"⚡ Benchmark abgeschlossen")
        print(f"📊 Durchschnittliche Latenz: {benchmark_results.get('avg_latency', 0.0):.3f}s")
        print(f"🚀 Throughput: {benchmark_results.get('throughput_qps', 0.0):.1f} QPS")

        return benchmark_results

    def _analyze_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert Vergleichsergebnisse.

        Args:
            results: Experiment-Ergebnisse

        Returns:
            Dictionary mit Vergleichsanalyse
        """
        analysis = {
            "metric_comparison": {},
            "statistical_significance": {},
            "best_performers": {},
            "worst_performers": {}
        }

        # Metriken extrahieren
        metrics = {}
        for config_name, result in results.items():
            eval_results = result["evaluation_results"]
            metrics[config_name] = {
                "rag_score": eval_results.get("rag_score", 0.0),
                "quality_score": eval_results.get("quality_score", 0.0),
                "efficiency_score": eval_results.get("efficiency_score", 0.0),
                "overall_score": eval_results.get("overall_score", 0.0)
            }

        # Vergleichsanalyse für jede Metrik
        for metric_name in ["rag_score", "quality_score", "efficiency_score", "overall_score"]:
            metric_values = {config: metrics[config][metric_name] for config in metrics}

            analysis["metric_comparison"][metric_name] = {
                "values": metric_values,
                "best": max(metric_values, key=metric_values.get),
                "worst": min(metric_values, key=metric_values.get),
                "mean": np.mean(list(metric_values.values())),
                "std": np.std(list(metric_values.values()))
            }

        return analysis

    def _rank_configurations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rankt Konfigurationen nach Performance.

        Args:
            results: Experiment-Ergebnisse

        Returns:
            Liste von gerankten Konfigurationen
        """
        ranking = []

        for config_name, result in results.items():
            eval_results = result["evaluation_results"]

            ranking.append({
                "name": config_name,
                "overall_score": eval_results.get("overall_score", 0.0),
                "rag_score": eval_results.get("rag_score", 0.0),
                "quality_score": eval_results.get("quality_score", 0.0),
                "efficiency_score": eval_results.get("efficiency_score", 0.0),
                "duration": result["duration"]
            })

        # Sortiere nach Overall Score
        ranking.sort(key=lambda x: x["overall_score"], reverse=True)

        return ranking

    def _save_experiment_results(self, results: Dict[str, Any]) -> None:
        """
        Speichert Experiment-Ergebnisse.

        Args:
            results: Experiment-Ergebnisse
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{results['experiment_name']}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def _save_comparison_results(self, results: Dict[str, Any]) -> None:
        """
        Speichert Vergleichsergebnisse.

        Args:
            results: Vergleichsergebnisse
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{results['experiment_name']}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """
        Speichert Benchmark-Ergebnisse.

        Args:
            results: Benchmark-Ergebnisse
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{results['benchmark_name']}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def load_experiment_results(self, filepath: str) -> Dict[str, Any]:
        """
        Lädt Experiment-Ergebnisse aus einer Datei.

        Args:
            filepath: Pfad zur Ergebnisdatei

        Returns:
            Dictionary mit Experiment-Ergebnissen
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """
        Gibt die Historie aller Experimente zurück.

        Returns:
            Liste aller Experiment-Ergebnisse
        """
        return self.experiment_history.copy()

    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """
        Generiert einen Bericht aus Experiment-Ergebnissen.

        Args:
            results: Experiment-Ergebnisse
            output_file: Optionaler Ausgabedatei-Pfad

        Returns:
            Bericht als String
        """
        report = []
        report.append(f"# Experiment Report: {results.get('experiment_name', 'Unknown')}")
        report.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        report.append(f"Duration: {results.get('duration', 0.0):.2f}s")
        report.append("")

        # Konfiguration
        report.append("## Configuration")
        config = results.get("config", {})
        for component, settings in config.items():
            if isinstance(settings, dict):
                report.append(f"- {component}: {settings.get('type', 'Unknown')}")
        report.append("")

        # Evaluierungsergebnisse
        report.append("## Evaluation Results")
        eval_results = results.get("evaluation_results", {})

        key_metrics = ["rag_score", "quality_score", "efficiency_score", "overall_score"]
        for metric in key_metrics:
            if metric in eval_results:
                report.append(f"- {metric}: {eval_results[metric]:.3f}")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text