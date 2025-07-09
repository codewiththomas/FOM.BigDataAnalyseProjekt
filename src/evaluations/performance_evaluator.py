from typing import List, Dict, Any, Optional, Callable
import time
import psutil
import os
from functools import wraps
from .base_evaluator import BaseEvaluator


class PerformanceEvaluator(BaseEvaluator):
    """
    Evaluator für Performance-Metriken.

    Berechnet Latenz, Throughput, Memory Usage und Cost per Query.
    """

    def __init__(self, name: str = "performance_evaluator", **kwargs):
        """
        Initialisiert den Performance-Evaluator.

        Args:
            name: Name des Evaluators
            **kwargs: Zusätzliche Parameter
        """
        super().__init__(name, **kwargs)
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self._get_memory_usage()

    def evaluate(self, predictions: List[Any], ground_truth: List[Any],
                **kwargs) -> Dict[str, Any]:
        """
        Führt die Performance-Evaluierung durch.

        Args:
            predictions: Liste der Vorhersagen (nicht verwendet, für Interface-Kompatibilität)
            ground_truth: Liste der Ground-Truth-Daten (nicht verwendet)
            **kwargs: Zusätzliche Parameter mit Performance-Daten

        Returns:
            Dictionary mit Performance-Metriken
        """
        # Performance-Daten aus kwargs extrahieren
        latencies = kwargs.get("latencies", [])
        memory_usage = kwargs.get("memory_usage", [])
        timestamps = kwargs.get("timestamps", [])
        costs = kwargs.get("costs", [])

        results = {}

        # Latenz-Metriken
        if latencies:
            results["avg_latency"] = sum(latencies) / len(latencies)
            results["min_latency"] = min(latencies)
            results["max_latency"] = max(latencies)
            results["median_latency"] = self._calculate_median(latencies)
            results["p95_latency"] = self._calculate_percentile(latencies, 95)
            results["p99_latency"] = self._calculate_percentile(latencies, 99)

        # Throughput-Metriken
        if timestamps and len(timestamps) > 1:
            total_time = timestamps[-1] - timestamps[0]
            if total_time > 0:
                results["throughput_qps"] = len(timestamps) / total_time
                results["total_processing_time"] = total_time

        # Memory-Metriken
        if memory_usage:
            results["avg_memory_mb"] = sum(memory_usage) / len(memory_usage)
            results["peak_memory_mb"] = max(memory_usage)
            results["memory_increase_mb"] = max(memory_usage) - min(memory_usage)

        # Cost-Metriken
        if costs:
            results["total_cost"] = sum(costs)
            results["avg_cost_per_query"] = sum(costs) / len(costs)
            results["cost_efficiency"] = len(costs) / sum(costs) if sum(costs) > 0 else 0

        # Zusätzliche Metriken
        results["num_queries"] = len(predictions) if predictions else len(latencies)
        results["current_memory_mb"] = self._get_memory_usage()
        results["memory_overhead_mb"] = self._get_memory_usage() - self.baseline_memory

        return results

    def measure_function_performance(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Misst die Performance einer Funktion.

        Args:
            func: Zu messende Funktion
            *args: Argumente für die Funktion
            **kwargs: Keyword-Argumente für die Funktion

        Returns:
            Dictionary mit Performance-Metriken und Funktionsergebnis
        """
        # Memory vor Ausführung
        memory_before = self._get_memory_usage()

        # Zeit messen
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Memory nach Ausführung
        memory_after = self._get_memory_usage()

        performance_data = {
            "execution_time": end_time - start_time,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_delta_mb": memory_after - memory_before,
            "result": result
        }

        return performance_data

    def benchmark_pipeline(self, pipeline_func: Callable, queries: List[str],
                          warmup_queries: int = 5) -> Dict[str, Any]:
        """
        Führt ein Benchmark einer Pipeline durch.

        Args:
            pipeline_func: Pipeline-Funktion zum Testen
            queries: Liste der Test-Queries
            warmup_queries: Anzahl der Warmup-Queries

        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        # Warmup
        warmup_data = queries[:warmup_queries]
        for query in warmup_data:
            pipeline_func(query)

        # Benchmark
        test_queries = queries[warmup_queries:]
        latencies = []
        memory_usage = []
        timestamps = []
        costs = []

        for query in test_queries:
            start_time = time.time()
            memory_before = self._get_memory_usage()

            result = pipeline_func(query)

            end_time = time.time()
            memory_after = self._get_memory_usage()

            latency = end_time - start_time
            latencies.append(latency)
            memory_usage.append(memory_after)
            timestamps.append(end_time)

            # Cost schätzen (falls verfügbar)
            if hasattr(result, 'get') and 'cost' in result:
                costs.append(result['cost'])

        # Evaluierung
        performance_results = self.evaluate(
            test_queries, test_queries,
            latencies=latencies,
            memory_usage=memory_usage,
            timestamps=timestamps,
            costs=costs
        )

        return performance_results

    def create_performance_decorator(self):
        """
        Erstellt einen Decorator für Performance-Messung.

        Returns:
            Decorator-Funktion
        """
        def performance_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.measure_function_performance(func, *args, **kwargs)
            return wrapper
        return performance_decorator

    def _get_memory_usage(self) -> float:
        """
        Gibt die aktuelle Memory-Nutzung in MB zurück.

        Returns:
            Memory-Nutzung in MB
        """
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Bytes zu MB
        except:
            return 0.0

    def _calculate_median(self, values: List[float]) -> float:
        """
        Berechnet den Median einer Liste von Werten.

        Args:
            values: Liste von Werten

        Returns:
            Median
        """
        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """
        Berechnet ein Perzentil einer Liste von Werten.

        Args:
            values: Liste von Werten
            percentile: Perzentil (0-100)

        Returns:
            Perzentil-Wert
        """
        sorted_values = sorted(values)
        n = len(sorted_values)

        if n == 0:
            return 0.0

        index = (percentile / 100) * (n - 1)

        if index == int(index):
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1

            if upper_index >= n:
                return sorted_values[lower_index]

            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def get_system_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das System zurück.

        Returns:
            Dictionary mit System-Informationen
        """
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "python_version": os.sys.version
        }

    def get_metric_names(self) -> List[str]:
        """
        Gibt die Namen der berechneten Metriken zurück.

        Returns:
            Liste der Metrik-Namen
        """
        return [
            "avg_latency",
            "min_latency",
            "max_latency",
            "median_latency",
            "p95_latency",
            "p99_latency",
            "throughput_qps",
            "total_processing_time",
            "avg_memory_mb",
            "peak_memory_mb",
            "memory_increase_mb",
            "total_cost",
            "avg_cost_per_query",
            "cost_efficiency",
            "num_queries",
            "current_memory_mb",
            "memory_overhead_mb"
        ]

    def compare_performance(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vergleicht zwei Performance-Ergebnisse.

        Args:
            results1: Erste Performance-Ergebnisse
            results2: Zweite Performance-Ergebnisse

        Returns:
            Dictionary mit Vergleichsergebnissen
        """
        comparison = {}

        for metric in self.get_metric_names():
            if metric in results1 and metric in results2:
                val1 = results1[metric]
                val2 = results2[metric]

                if val1 != 0:
                    improvement = (val2 - val1) / val1 * 100
                    comparison[f"{metric}_improvement_percent"] = improvement

                comparison[f"{metric}_ratio"] = val2 / val1 if val1 != 0 else float('inf')

        return comparison