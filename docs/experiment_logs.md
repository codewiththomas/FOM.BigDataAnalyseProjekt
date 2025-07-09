# Experiment Logs

## Overview

This document provides guidelines for experiment logging, analysis, and reproducibility in the ResearchRAG system. Proper experiment logging is crucial for research reproducibility and systematic comparison of different RAG configurations.

## Experiment Structure

### Experiment Metadata

Each experiment should include comprehensive metadata:

```json
{
  "experiment_id": "exp_001_baseline_comparison",
  "timestamp": "2025-01-08T10:30:00Z",
  "researcher": "FOM Research Team",
  "description": "Baseline comparison of different chunking strategies",
  "tags": ["chunking", "baseline", "comparison"],
  "version": "1.0.0",
  "git_commit": "abc123def456",
  "environment": {
    "python_version": "3.10.8",
    "cuda_version": "11.8",
    "gpu_model": "NVIDIA RTX 4090",
    "memory_gb": 32
  }
}
```

### Configuration Logging

All configurations should be logged with full parameter details:

```json
{
  "configuration": {
    "chunker": {
      "type": "recursive_chunker",
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "separators": ["\n\n", "\n", ". ", " "],
      "keep_separator": true
    },
    "embedding": {
      "type": "sentence_transformer",
      "model": "all-MiniLM-L6-v2",
      "normalize_embeddings": true,
      "batch_size": 32,
      "device": "cuda"
    },
    "vector_store": {
      "type": "chroma",
      "collection_name": "rag_documents",
      "persist_directory": "./data/chroma_db",
      "distance_metric": "cosine"
    },
    "language_model": {
      "type": "ollama",
      "model": "llama3.2",
      "temperature": 0.1,
      "max_tokens": 500,
      "base_url": "http://localhost:11434"
    }
  }
}
```

### Dataset Information

Document the dataset used for evaluation:

```json
{
  "dataset": {
    "name": "DSGVO_QA_Dataset",
    "version": "1.0.0",
    "total_questions": 55,
    "categories": {
      "faktenfragen": 20,
      "prozessfragen": 15,
      "problemlösungsfragen": 15,
      "interpretationsfragen": 5
    },
    "difficulty_levels": {
      "easy": 18,
      "medium": 25,
      "hard": 12
    },
    "source_documents": ["dsgvo.txt"],
    "document_stats": {
      "total_characters": 245680,
      "total_words": 35240,
      "total_paragraphs": 890
    }
  }
}
```

## Results Logging

### Evaluation Metrics

Log comprehensive evaluation results:

```json
{
  "results": {
    "retrieval_metrics": {
      "precision_at_1": 0.85,
      "precision_at_3": 0.78,
      "precision_at_5": 0.72,
      "precision_at_10": 0.65,
      "recall_at_1": 0.45,
      "recall_at_3": 0.68,
      "recall_at_5": 0.78,
      "recall_at_10": 0.85,
      "f1_at_1": 0.59,
      "f1_at_3": 0.73,
      "f1_at_5": 0.75,
      "f1_at_10": 0.74,
      "mrr": 0.72,
      "ndcg_at_5": 0.76,
      "ndcg_at_10": 0.78
    },
    "generation_metrics": {
      "rouge_l": 0.68,
      "bleu_score": 0.45,
      "exact_match": 0.32,
      "semantic_similarity": 0.75,
      "avg_length": 85.4,
      "length_ratio": 0.92
    },
    "performance_metrics": {
      "avg_latency_ms": 450.2,
      "min_latency_ms": 320.1,
      "max_latency_ms": 680.5,
      "p95_latency_ms": 625.3,
      "p99_latency_ms": 668.9,
      "throughput_qps": 2.22,
      "memory_usage_mb": 512.8,
      "cost_per_query": 0.0,
      "total_cost": 0.0
    },
    "rag_metrics": {
      "faithfulness": 0.82,
      "groundedness": 0.78,
      "answer_relevance": 0.85,
      "context_precision": 0.76,
      "context_recall": 0.82,
      "rag_score": 0.78,
      "quality_score": 0.75,
      "efficiency_score": 0.88,
      "overall_score": 0.80
    }
  }
}
```

### Statistical Analysis

Include statistical significance testing:

```json
{
  "statistical_analysis": {
    "sample_size": 55,
    "confidence_interval": 0.95,
    "significance_tests": {
      "retrieval_vs_baseline": {
        "test_type": "paired_t_test",
        "p_value": 0.0032,
        "statistically_significant": true,
        "effect_size": 0.68
      },
      "generation_vs_baseline": {
        "test_type": "wilcoxon_signed_rank",
        "p_value": 0.0156,
        "statistically_significant": true,
        "effect_size": 0.45
      }
    },
    "confidence_intervals": {
      "rouge_l": [0.64, 0.72],
      "precision_at_5": [0.68, 0.76],
      "rag_score": [0.74, 0.82]
    }
  }
}
```

## Experiment Types

### 1. Baseline Experiments

Establish baseline performance with standard configurations:

```json
{
  "experiment_type": "baseline",
  "purpose": "Establish baseline performance metrics",
  "configurations": [
    {
      "name": "openai_baseline",
      "description": "OpenAI embedding + OpenAI LLM baseline",
      "config": {
        "chunker": {"type": "line_chunker", "chunk_size": 512},
        "embedding": {"type": "openai", "model": "text-embedding-3-small"},
        "vector_store": {"type": "in_memory"},
        "language_model": {"type": "openai", "model": "gpt-4o-mini"}
      }
    }
  ]
}
```

### 2. Component Comparison

Compare different implementations of the same component:

```json
{
  "experiment_type": "component_comparison",
  "component": "chunker",
  "purpose": "Compare different chunking strategies",
  "configurations": [
    {
      "name": "line_chunker",
      "config": {"chunker": {"type": "line_chunker", "chunk_size": 1000}}
    },
    {
      "name": "recursive_chunker",
      "config": {"chunker": {"type": "recursive_chunker", "chunk_size": 1000}}
    },
    {
      "name": "semantic_chunker",
      "config": {"chunker": {"type": "semantic_chunker", "chunk_size": 1000}}
    }
  ]
}
```

### 3. Ablation Studies

Systematic removal or modification of components:

```json
{
  "experiment_type": "ablation_study",
  "purpose": "Identify contribution of each component",
  "base_config": {
    "chunker": {"type": "recursive_chunker", "chunk_size": 1000},
    "embedding": {"type": "sentence_transformer", "model": "all-MiniLM-L6-v2"},
    "vector_store": {"type": "chroma"},
    "language_model": {"type": "ollama", "model": "llama3.2"}
  },
  "ablations": [
    {
      "name": "no_overlap",
      "modification": {"chunker": {"chunk_overlap": 0}},
      "hypothesis": "Overlap improves retrieval quality"
    },
    {
      "name": "larger_chunks",
      "modification": {"chunker": {"chunk_size": 2000}},
      "hypothesis": "Larger chunks provide better context"
    }
  ]
}
```

### 4. Hyperparameter Optimization

Systematic parameter tuning:

```json
{
  "experiment_type": "hyperparameter_optimization",
  "purpose": "Find optimal parameters for chunk size and overlap",
  "parameter_grid": {
    "chunk_size": [500, 1000, 1500, 2000],
    "chunk_overlap": [0, 100, 200, 300],
    "temperature": [0.0, 0.1, 0.3, 0.5]
  },
  "optimization_metric": "rag_score",
  "search_strategy": "grid_search"
}
```

### 5. Performance Benchmarks

System performance under different conditions:

```json
{
  "experiment_type": "performance_benchmark",
  "purpose": "Evaluate system performance under load",
  "test_conditions": [
    {
      "name": "single_query",
      "concurrent_queries": 1,
      "total_queries": 100
    },
    {
      "name": "moderate_load",
      "concurrent_queries": 5,
      "total_queries": 500
    },
    {
      "name": "high_load",
      "concurrent_queries": 20,
      "total_queries": 1000
    }
  ]
}
```

## Best Practices

### 1. Reproducibility

Ensure experiments can be reproduced:

```python
# Set random seeds
import random
import numpy as np
import torch

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Log environment information
import sys
import platform
import torch

environment_info = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
}
```

### 2. Version Control

Track code and data versions:

```python
import subprocess
import os

def get_git_info():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        is_dirty = len(subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()) > 0

        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "is_dirty": is_dirty,
            "repository": os.path.basename(os.getcwd())
        }
    except:
        return {"error": "Git information not available"}
```

### 3. Error Handling

Log errors and failures:

```python
import traceback
import logging

def log_experiment_error(experiment_id, error, context=None):
    error_info = {
        "experiment_id": experiment_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context,
        "timestamp": datetime.utcnow().isoformat()
    }

    logging.error(f"Experiment {experiment_id} failed: {error_info}")
    return error_info
```

### 4. Progress Tracking

Monitor long-running experiments:

```python
from tqdm import tqdm
import time

def run_experiment_with_progress(configs, qa_pairs):
    results = {}

    for config_name, config in tqdm(configs.items(), desc="Configurations"):
        config_results = {}

        for i, qa_pair in enumerate(tqdm(qa_pairs, desc=f"Processing {config_name}", leave=False)):
            try:
                # Run experiment
                result = process_qa_pair(config, qa_pair)
                config_results[f"qa_{i}"] = result

                # Log progress every 10 questions
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i+1}/{len(qa_pairs)} questions for {config_name}")

            except Exception as e:
                logging.error(f"Error processing QA pair {i} for {config_name}: {e}")
                config_results[f"qa_{i}"] = {"error": str(e)}

        results[config_name] = config_results

    return results
```

## Analysis and Reporting

### 1. Comparative Analysis

Compare multiple experiments:

```python
def compare_experiments(experiment_results):
    comparison = {
        "summary": {},
        "detailed_comparison": {},
        "statistical_tests": {},
        "recommendations": []
    }

    # Extract key metrics
    metrics = ["rag_score", "rouge_l", "precision_at_5", "avg_latency_ms"]

    for metric in metrics:
        comparison["summary"][metric] = {}

        for exp_name, results in experiment_results.items():
            if metric in results["results"]:
                comparison["summary"][metric][exp_name] = results["results"][metric]

    # Statistical significance testing
    for metric in metrics:
        if len(comparison["summary"][metric]) > 1:
            # Perform statistical tests
            comparison["statistical_tests"][metric] = perform_statistical_test(
                comparison["summary"][metric]
            )

    return comparison
```

### 2. Visualization

Create visualizations for experiment results:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_experiment_visualizations(results, output_dir):
    # Performance comparison
    plt.figure(figsize=(12, 8))

    # Extract metrics for plotting
    experiments = list(results.keys())
    rag_scores = [results[exp]["results"]["rag_score"] for exp in experiments]
    latencies = [results[exp]["results"]["avg_latency_ms"] for exp in experiments]

    # Create scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(latencies, rag_scores)
    plt.xlabel("Average Latency (ms)")
    plt.ylabel("RAG Score")
    plt.title("Performance vs Quality Trade-off")

    for i, exp in enumerate(experiments):
        plt.annotate(exp, (latencies[i], rag_scores[i]))

    # Metric comparison
    plt.subplot(2, 2, 2)
    metrics = ["precision_at_5", "recall_at_5", "rouge_l", "bleu_score"]

    for i, exp in enumerate(experiments):
        values = [results[exp]["results"][metric] for metric in metrics]
        plt.plot(metrics, values, marker='o', label=exp)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Metric Comparison")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/experiment_comparison.png")
    plt.close()
```

### 3. Report Generation

Generate comprehensive reports:

```python
def generate_experiment_report(results, output_path):
    report = {
        "executive_summary": generate_executive_summary(results),
        "methodology": generate_methodology_section(results),
        "results": generate_results_section(results),
        "analysis": generate_analysis_section(results),
        "conclusions": generate_conclusions(results),
        "recommendations": generate_recommendations(results),
        "appendices": generate_appendices(results)
    }

    # Save as JSON
    with open(f"{output_path}/experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    markdown_report = generate_markdown_report(report)
    with open(f"{output_path}/experiment_report.md", "w") as f:
        f.write(markdown_report)

    return report
```

## Experiment Templates

### Template 1: Component Evaluation

```python
def evaluate_component(component_type, implementations, base_config, qa_pairs):
    """
    Template for evaluating different implementations of a component.
    """
    results = {}

    for impl_name, impl_config in implementations.items():
        # Create configuration
        config = base_config.copy()
        config[component_type] = impl_config

        # Run experiment
        experiment_name = f"{component_type}_{impl_name}"
        result = run_single_experiment(config, qa_pairs, experiment_name)

        results[impl_name] = result

    # Generate comparison report
    comparison = compare_component_results(results)

    return {
        "individual_results": results,
        "comparison": comparison,
        "recommendation": select_best_implementation(comparison)
    }
```

### Template 2: Parameter Optimization

```python
def optimize_parameters(parameter_grid, base_config, qa_pairs, optimization_metric):
    """
    Template for parameter optimization experiments.
    """
    from itertools import product

    results = {}
    best_score = -1
    best_config = None

    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())

    for combination in product(*param_values):
        # Create configuration
        config = base_config.copy()
        param_config = dict(zip(param_names, combination))

        # Update configuration with parameters
        update_config_with_params(config, param_config)

        # Run experiment
        experiment_name = f"param_{'_'.join(map(str, combination))}"
        result = run_single_experiment(config, qa_pairs, experiment_name)

        results[experiment_name] = {
            "parameters": param_config,
            "results": result
        }

        # Track best configuration
        if result[optimization_metric] > best_score:
            best_score = result[optimization_metric]
            best_config = param_config

    return {
        "all_results": results,
        "best_configuration": best_config,
        "best_score": best_score,
        "optimization_metric": optimization_metric
    }
```

## Quality Assurance

### 1. Validation Checks

Implement validation for experiment integrity:

```python
def validate_experiment_results(results):
    """
    Validate experiment results for consistency and completeness.
    """
    validation_errors = []

    # Check required fields
    required_fields = ["configuration", "results", "metadata"]
    for field in required_fields:
        if field not in results:
            validation_errors.append(f"Missing required field: {field}")

    # Check metric ranges
    if "results" in results:
        metrics = results["results"]

        # Check that scores are in valid ranges
        for metric, value in metrics.items():
            if metric.endswith("_score") or metric.startswith("precision") or metric.startswith("recall"):
                if not (0 <= value <= 1):
                    validation_errors.append(f"Invalid score range for {metric}: {value}")

    # Check for statistical significance
    if "statistical_analysis" in results:
        stats = results["statistical_analysis"]
        if "p_value" in stats and stats["p_value"] < 0.05:
            if not stats.get("statistically_significant", False):
                validation_errors.append("P-value < 0.05 but not marked as statistically significant")

    return validation_errors
```

### 2. Automated Testing

Set up automated experiment testing:

```python
def test_experiment_pipeline():
    """
    Automated test for the experiment pipeline.
    """
    # Create minimal test configuration
    test_config = {
        "chunker": {"type": "line_chunker", "chunk_size": 100},
        "embedding": {"type": "sentence_transformer", "model": "all-MiniLM-L6-v2"},
        "vector_store": {"type": "in_memory"},
        "language_model": {"type": "ollama", "model": "llama3.2"}
    }

    # Create test QA pairs
    test_qa_pairs = [
        {
            "id": "test_001",
            "question": "What is the test question?",
            "gold_answer": "This is the test answer.",
            "category": "test"
        }
    ]

    # Run test experiment
    try:
        results = run_single_experiment(test_config, test_qa_pairs, "pipeline_test")

        # Validate results
        validation_errors = validate_experiment_results(results)

        if validation_errors:
            raise Exception(f"Validation errors: {validation_errors}")

        print("✓ Experiment pipeline test passed")
        return True

    except Exception as e:
        print(f"✗ Experiment pipeline test failed: {e}")
        return False
```

## Conclusion

Proper experiment logging and analysis are essential for reproducible research in RAG systems. This framework provides:

1. **Structured logging** of experiments, configurations, and results
2. **Statistical analysis** for significance testing
3. **Visualization tools** for result interpretation
4. **Templates** for common experiment types
5. **Quality assurance** mechanisms

By following these guidelines, researchers can ensure their experiments are reproducible, comparable, and contribute to the advancement of RAG technology.

For more information, see:
- [Architecture Documentation](architecture.md)
- [API Reference](api_reference.md)
- [Results Analysis](results_analysis.md)