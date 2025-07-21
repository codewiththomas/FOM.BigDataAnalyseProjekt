"""
Evaluator module for RAG system evaluation using RAGAS and BERTScore.
"""

import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore


class RAGEvaluator:
    """Evaluator for RAG systems with multiple metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_config = config.get("evaluation", {})
        self.metrics = self.evaluation_config.get("metrics", ["ragas", "bertscore", "performance"])
        self.batch_size = self.evaluation_config.get("batch_size", 5)

        # Initialize evaluators
        self.ragas_evaluator = None
        self.bertscore_evaluator = None

        if "ragas" in self.metrics:
            self._init_ragas()
        if "bertscore" in self.metrics:
            self._init_bertscore()

    def _init_ragas(self):
        """Initialize RAGAS evaluator."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall
            )
            from datasets import Dataset

            self.ragas_evaluate = evaluate
            self.ragas_Dataset = Dataset

            # Select metrics based on config
            ragas_metrics_config = self.evaluation_config.get("ragas_metrics", [
                "answer_relevancy", "faithfulness", "context_precision"
            ])

            self.ragas_metrics = []
            metric_map = {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall
            }

            for metric_name in ragas_metrics_config:
                if metric_name in metric_map:
                    self.ragas_metrics.append(metric_map[metric_name])

            print(f"RAGAS initialized with metrics: {ragas_metrics_config}")

        except ImportError as e:
            print(f"RAGAS not available: {e}")
            self.ragas_metrics = []
            self.ragas_evaluate = None
            self.ragas_Dataset = None

    def _init_bertscore(self):
        """Initialize BERTScore evaluator."""
        try:
            from bert_score import score
            self.bertscore_func = score

            self.bertscore_model = self.evaluation_config.get("bertscore", {}).get(
                "model", "microsoft/deberta-xlarge-mnli"
            )
            self.bertscore_lang = self.evaluation_config.get("bertscore", {}).get("lang", "de")

            print(f"BERTScore initialized with model: {self.bertscore_model}")

        except ImportError as e:
            print(f"BERTScore not available: {e}")
            self.bertscore_evaluator = None

    def evaluate_single(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        retrieved_contexts: List[str],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.

        Args:
            question: The question asked
            generated_answer: Generated answer from RAG system
            reference_answer: Ground truth reference answer
            retrieved_contexts: List of retrieved context chunks
            metadata: Additional metadata (model info, timing, etc.)

        Returns:
            Dictionary with evaluation scores
        """
        results = {
            "question": question,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "retrieved_contexts": retrieved_contexts,
            "metadata": metadata or {}
        }

        # RAGAS evaluation
        if "ragas" in self.metrics and hasattr(self, 'ragas_metrics') and self.ragas_metrics:
            ragas_scores = self._evaluate_ragas_single(
                question, generated_answer, reference_answer, retrieved_contexts
            )
            results.update(ragas_scores)

        # BERTScore evaluation
        if "bertscore" in self.metrics and self.bertscore_func:
            bertscore_scores = self._evaluate_bertscore_single(generated_answer, reference_answer)
            results.update(bertscore_scores)

        # Performance metrics (if available in metadata)
        if "performance" in self.metrics and metadata:
            performance_scores = self._extract_performance_metrics(metadata)
            results.update(performance_scores)

        return results

    def _evaluate_ragas_single(
        self, question: str, answer: str, reference: str, contexts: List[str]
    ) -> Dict[str, float]:
        """Evaluate single item with RAGAS."""
        try:
            # Ensure we have contexts for RAGAS
            if not contexts or len(contexts) == 0:
                print("Warning: No contexts provided for RAGAS evaluation")
                return {f"ragas_{metric.name}": 0.0 for metric in self.ragas_metrics}

            # Prepare data for RAGAS 0.1.20
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truths": [reference]
            }

            dataset = self.ragas_Dataset.from_dict(data)

            # Run RAGAS evaluation with error handling
            try:
                result = self.ragas_evaluate(dataset, metrics=self.ragas_metrics)

                # Extract scores - handle different result formats
                ragas_scores = {}
                for metric in self.ragas_metrics:
                    metric_name = metric.name
                    if hasattr(result, metric_name):
                        # Handle pandas Series/DataFrame format
                        score_series = getattr(result, metric_name)
                        if hasattr(score_series, 'iloc') and len(score_series) > 0:
                            score = float(score_series.iloc[0])
                        else:
                            score = float(score_series) if score_series is not None else 0.0
                    elif metric_name in result:
                        # Handle dictionary format
                        score_value = result[metric_name]
                        if isinstance(score_value, list) and len(score_value) > 0:
                            score = float(score_value[0])
                        else:
                            score = float(score_value) if score_value is not None else 0.0
                    else:
                        score = 0.0

                    ragas_scores[f"ragas_{metric_name}"] = score

                return ragas_scores

            except Exception as e:
                print(f"Error during RAGAS evaluation execution: {e}")
                return {f"ragas_{metric.name}": 0.0 for metric in self.ragas_metrics}

        except Exception as e:
            print(f"Error in RAGAS evaluation setup: {e}")
            return {f"ragas_{metric.name}": 0.0 for metric in self.ragas_metrics}

    def _evaluate_bertscore_single(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate single item with BERTScore."""
        try:
            P, R, F1 = self.bertscore_func(
                cands=[generated],
                refs=[reference],
                model_type=self.bertscore_model,
                lang=self.bertscore_lang,
                verbose=False
            )

            return {
                "bertscore_precision": float(P[0]),
                "bertscore_recall": float(R[0]),
                "bertscore_f1": float(F1[0])
            }

        except Exception as e:
            print(f"Error in BERTScore evaluation: {e}")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0
            }

    def _extract_performance_metrics(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from metadata."""
        performance = {}

        if "execution_time" in metadata:
            performance["execution_time"] = metadata["execution_time"]

        if "tokens_per_second" in metadata:
            performance["tokens_per_second"] = metadata["tokens_per_second"]

        if "cost_estimate" in metadata:
            performance["cost_estimate"] = metadata["cost_estimate"]

        if "usage" in metadata:
            usage = metadata["usage"]
            performance["total_tokens"] = usage.get("total_tokens", 0)
            performance["prompt_tokens"] = usage.get("prompt_tokens", 0)
            performance["completion_tokens"] = usage.get("completion_tokens", 0)

        return performance

    def evaluate_batch(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of test questions.

        Args:
            test_data: List of test items with keys: question, generated_answer,
                      reference_answer, retrieved_contexts, metadata

        Returns:
            List of evaluation results
        """
        results = []

        print(f"Evaluating {len(test_data)} items...")

        for item in tqdm(test_data, desc="Evaluating"):
            result = self.evaluate_single(
                question=item["question"],
                generated_answer=item["generated_answer"],
                reference_answer=item["reference_answer"],
                retrieved_contexts=item["retrieved_contexts"],
                metadata=item.get("metadata", {})
            )
            results.append(result)

        return results

    def compute_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate metrics across all results."""
        if not results:
            return {}

        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(results)

        aggregates = {}

        # RAGAS metrics
        ragas_columns = [col for col in df.columns if col.startswith("ragas_")]
        for col in ragas_columns:
            if col in df.columns and df[col].notna().any():
                aggregates[f"{col}_mean"] = df[col].mean()
                aggregates[f"{col}_std"] = df[col].std()

        # BERTScore metrics
        bertscore_columns = [col for col in df.columns if col.startswith("bertscore_")]
        for col in bertscore_columns:
            if col in df.columns and df[col].notna().any():
                aggregates[f"{col}_mean"] = df[col].mean()
                aggregates[f"{col}_std"] = df[col].std()

        # Performance metrics
        performance_columns = ["execution_time", "tokens_per_second", "cost_estimate", "total_tokens"]
        for col in performance_columns:
            if col in df.columns and df[col].notna().any():
                aggregates[f"{col}_mean"] = df[col].mean()
                aggregates[f"{col}_total"] = df[col].sum() if col == "cost_estimate" else df[col].mean()

        return aggregates

    def save_results(self, results: List[Dict[str, Any]], output_dir: str, model_name: str):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = output_path / f"{model_name}_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save aggregated metrics
        aggregates = self.compute_aggregate_metrics(results)
        aggregates_file = output_path / f"{model_name}_aggregates_{timestamp}.json"
        with open(aggregates_file, 'w', encoding='utf-8') as f:
            json.dump(aggregates, f, ensure_ascii=False, indent=2)

        # Save CSV for easy analysis
        df = pd.DataFrame(results)
        csv_file = output_path / f"{model_name}_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')

        print(f"Results saved to {output_path}")
        return {
            "detailed": detailed_file,
            "aggregates": aggregates_file,
            "csv": csv_file
        }

    def compare_models(self, results_dir: str) -> pd.DataFrame:
        """Compare results across different models."""
        results_path = Path(results_dir)

        if not results_path.exists():
            print(f"Results directory {results_dir} does not exist")
            return pd.DataFrame()

        # Find all aggregate files
        aggregate_files = list(results_path.glob("*_aggregates_*.json"))

        comparison_data = []

        for file in aggregate_files:
            # Extract model name from filename
            model_name = file.name.split("_aggregates_")[0]

            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                data["model"] = model_name
                comparison_data.append(data)

            except Exception as e:
                print(f"Error loading {file}: {e}")

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            # Reorder columns to put model first
            cols = ["model"] + [col for col in df.columns if col != "model"]
            df = df[cols]
            return df

        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    config = {
        "evaluation": {
            "metrics": ["ragas", "bertscore", "performance"],
            "ragas_metrics": ["answer_relevancy", "faithfulness"],
            "bertscore": {
                "model": "microsoft/deberta-xlarge-mnli",
                "lang": "de"
            }
        }
    }

    evaluator = RAGEvaluator(config)

    # Sample evaluation
    sample_data = [{
        "question": "Was ist die DSGVO?",
        "generated_answer": "Die DSGVO ist eine EU-Verordnung zum Schutz personenbezogener Daten.",
        "reference_answer": "Die DSGVO ist die Datenschutz-Grundverordnung der EU.",
        "retrieved_contexts": ["Die DSGVO regelt den Datenschutz in der EU."],
        "metadata": {"execution_time": 1.5, "cost_estimate": 0.001}
    }]

    try:
        results = evaluator.evaluate_batch(sample_data)
        print("Sample evaluation completed!")

        aggregates = evaluator.compute_aggregate_metrics(results)
        print(f"Aggregates: {aggregates}")

    except Exception as e:
        print(f"Error in evaluation: {e}")

    print("Evaluator test complete!")