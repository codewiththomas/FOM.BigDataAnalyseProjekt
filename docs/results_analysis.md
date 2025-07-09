# Results Analysis Guide

## Overview

This document provides comprehensive guidance for analyzing experimental results in the ResearchRAG system. It covers statistical methods, visualization techniques, and interpretation guidelines for RAG system evaluation.

## Statistical Analysis Framework

### 1. Descriptive Statistics

Calculate basic statistics for all metrics:

```python
import numpy as np
import pandas as pd
from scipy import stats

def calculate_descriptive_stats(results):
    """
    Calculate descriptive statistics for experiment results.
    """
    stats_summary = {}

    for metric_name, values in results.items():
        if isinstance(values, list) and len(values) > 0:
            stats_summary[metric_name] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "iqr": np.percentile(values, 75) - np.percentile(values, 25),
                "count": len(values),
                "variance": np.var(values),
                "skewness": stats.skew(values),
                "kurtosis": stats.kurtosis(values)
            }

    return stats_summary
```

### 2. Significance Testing

Perform statistical significance tests:

```python
def perform_significance_tests(baseline_results, treatment_results, alpha=0.05):
    """
    Perform statistical significance tests between baseline and treatment.
    """
    test_results = {}

    for metric in baseline_results.keys():
        if metric in treatment_results:
            baseline_values = baseline_results[metric]
            treatment_values = treatment_results[metric]

            # Normality test
            _, baseline_normal = stats.shapiro(baseline_values)
            _, treatment_normal = stats.shapiro(treatment_values)

            both_normal = baseline_normal > alpha and treatment_normal > alpha

            if both_normal and len(baseline_values) == len(treatment_values):
                # Paired t-test for normal data
                statistic, p_value = stats.ttest_rel(treatment_values, baseline_values)
                test_type = "paired_t_test"
            elif both_normal:
                # Independent t-test for normal data
                statistic, p_value = stats.ttest_ind(treatment_values, baseline_values)
                test_type = "independent_t_test"
            else:
                # Wilcoxon signed-rank test for non-normal data
                statistic, p_value = stats.wilcoxon(treatment_values, baseline_values)
                test_type = "wilcoxon_signed_rank"

            # Effect size calculation
            effect_size = calculate_effect_size(baseline_values, treatment_values)

            test_results[metric] = {
                "test_type": test_type,
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < alpha,
                "effect_size": effect_size,
                "interpretation": interpret_effect_size(effect_size)
            }

    return test_results

def calculate_effect_size(baseline, treatment):
    """
    Calculate Cohen's d effect size.
    """
    pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline) +
                         (len(treatment) - 1) * np.var(treatment)) /
                        (len(baseline) + len(treatment) - 2))

    return (np.mean(treatment) - np.mean(baseline)) / pooled_std

def interpret_effect_size(effect_size):
    """
    Interpret effect size magnitude.
    """
    abs_effect = abs(effect_size)

    if abs_effect < 0.2:
        return "negligible"
    elif abs_effect < 0.5:
        return "small"
    elif abs_effect < 0.8:
        return "medium"
    else:
        return "large"
```

### 3. Confidence Intervals

Calculate confidence intervals for metrics:

```python
def calculate_confidence_intervals(values, confidence_level=0.95):
    """
    Calculate confidence intervals for given values.
    """
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean

    # t-distribution for small samples
    t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)

    margin_of_error = t_critical * std_err

    return {
        "mean": mean,
        "lower_bound": mean - margin_of_error,
        "upper_bound": mean + margin_of_error,
        "margin_of_error": margin_of_error,
        "confidence_level": confidence_level
    }
```

### 4. Multiple Comparisons

Handle multiple comparisons with correction:

```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_comparisons(p_values, method='bonferroni'):
    """
    Correct for multiple comparisons.
    """
    rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
        p_values, method=method
    )

    return {
        "original_p_values": p_values,
        "corrected_p_values": corrected_p_values,
        "rejected_hypotheses": rejected,
        "correction_method": method,
        "alpha_bonferroni": alpha_bonf,
        "alpha_sidak": alpha_sidak
    }
```

## Performance Analysis

### 1. Latency Analysis

Analyze response time distributions:

```python
def analyze_latency(latency_data):
    """
    Comprehensive latency analysis.
    """
    analysis = {
        "descriptive_stats": calculate_descriptive_stats({"latency": latency_data}),
        "percentiles": {
            "p50": np.percentile(latency_data, 50),
            "p90": np.percentile(latency_data, 90),
            "p95": np.percentile(latency_data, 95),
            "p99": np.percentile(latency_data, 99),
            "p99.9": np.percentile(latency_data, 99.9)
        },
        "outlier_analysis": detect_outliers(latency_data),
        "distribution_fit": fit_distribution(latency_data)
    }

    return analysis

def detect_outliers(data, method='iqr'):
    """
    Detect outliers using IQR method.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [x for x in data if x < lower_bound or x > upper_bound]

    return {
        "outliers": outliers,
        "outlier_count": len(outliers),
        "outlier_percentage": len(outliers) / len(data) * 100,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }
```

### 2. Throughput Analysis

Analyze system throughput characteristics:

```python
def analyze_throughput(throughput_data, time_series=None):
    """
    Analyze system throughput.
    """
    analysis = {
        "average_qps": np.mean(throughput_data),
        "peak_qps": np.max(throughput_data),
        "min_qps": np.min(throughput_data),
        "throughput_stability": calculate_stability(throughput_data),
        "capacity_utilization": calculate_capacity_utilization(throughput_data)
    }

    if time_series:
        analysis["time_series_analysis"] = analyze_time_series(throughput_data, time_series)

    return analysis

def calculate_stability(data):
    """
    Calculate throughput stability using coefficient of variation.
    """
    cv = np.std(data) / np.mean(data)

    if cv < 0.1:
        stability = "very_stable"
    elif cv < 0.2:
        stability = "stable"
    elif cv < 0.3:
        stability = "moderately_stable"
    else:
        stability = "unstable"

    return {
        "coefficient_of_variation": cv,
        "stability_rating": stability
    }
```

### 3. Resource Utilization

Analyze memory and CPU usage:

```python
def analyze_resource_usage(memory_data, cpu_data=None):
    """
    Analyze resource utilization patterns.
    """
    analysis = {
        "memory_analysis": {
            "peak_usage_mb": np.max(memory_data),
            "average_usage_mb": np.mean(memory_data),
            "memory_efficiency": calculate_memory_efficiency(memory_data),
            "memory_leaks": detect_memory_leaks(memory_data)
        }
    }

    if cpu_data:
        analysis["cpu_analysis"] = {
            "peak_cpu_percent": np.max(cpu_data),
            "average_cpu_percent": np.mean(cpu_data),
            "cpu_efficiency": calculate_cpu_efficiency(cpu_data)
        }

    return analysis

def detect_memory_leaks(memory_data):
    """
    Detect potential memory leaks using trend analysis.
    """
    # Simple linear regression to detect upward trend
    x = np.arange(len(memory_data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, memory_data)

    return {
        "trend_slope": slope,
        "correlation": r_value,
        "p_value": p_value,
        "potential_leak": slope > 0 and p_value < 0.05 and r_value > 0.7
    }
```

## Quality Metrics Analysis

### 1. Retrieval Quality

Analyze retrieval performance:

```python
def analyze_retrieval_quality(precision_at_k, recall_at_k, mrr_scores, ndcg_scores):
    """
    Comprehensive retrieval quality analysis.
    """
    analysis = {
        "precision_analysis": {
            "k_values": list(precision_at_k.keys()),
            "precision_degradation": calculate_precision_degradation(precision_at_k),
            "optimal_k": find_optimal_k(precision_at_k, recall_at_k)
        },
        "recall_analysis": {
            "recall_saturation": calculate_recall_saturation(recall_at_k),
            "recall_improvement": calculate_recall_improvement(recall_at_k)
        },
        "ranking_quality": {
            "mrr_analysis": analyze_mrr(mrr_scores),
            "ndcg_analysis": analyze_ndcg(ndcg_scores)
        }
    }

    return analysis

def calculate_precision_degradation(precision_at_k):
    """
    Calculate how precision degrades as k increases.
    """
    k_values = sorted(precision_at_k.keys())
    degradation_rates = []

    for i in range(1, len(k_values)):
        prev_k = k_values[i-1]
        curr_k = k_values[i]

        degradation = (precision_at_k[prev_k] - precision_at_k[curr_k]) / precision_at_k[prev_k]
        degradation_rates.append(degradation)

    return {
        "degradation_rates": degradation_rates,
        "average_degradation": np.mean(degradation_rates),
        "max_degradation": np.max(degradation_rates)
    }

def find_optimal_k(precision_at_k, recall_at_k):
    """
    Find optimal k value balancing precision and recall.
    """
    f1_scores = {}

    for k in precision_at_k.keys():
        if k in recall_at_k:
            p = precision_at_k[k]
            r = recall_at_k[k]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores[k] = f1

    optimal_k = max(f1_scores, key=f1_scores.get)

    return {
        "optimal_k": optimal_k,
        "optimal_f1": f1_scores[optimal_k],
        "all_f1_scores": f1_scores
    }
```

### 2. Generation Quality

Analyze text generation quality:

```python
def analyze_generation_quality(rouge_scores, bleu_scores, semantic_scores):
    """
    Analyze text generation quality metrics.
    """
    analysis = {
        "rouge_analysis": {
            "distribution": calculate_descriptive_stats({"rouge_l": rouge_scores}),
            "quality_categories": categorize_rouge_scores(rouge_scores)
        },
        "bleu_analysis": {
            "distribution": calculate_descriptive_stats({"bleu": bleu_scores}),
            "quality_assessment": assess_bleu_quality(bleu_scores)
        },
        "semantic_analysis": {
            "distribution": calculate_descriptive_stats({"semantic": semantic_scores}),
            "correlation_with_rouge": stats.pearsonr(rouge_scores, semantic_scores)[0]
        }
    }

    return analysis

def categorize_rouge_scores(rouge_scores):
    """
    Categorize ROUGE scores into quality levels.
    """
    categories = {
        "excellent": len([s for s in rouge_scores if s >= 0.8]),
        "good": len([s for s in rouge_scores if 0.6 <= s < 0.8]),
        "fair": len([s for s in rouge_scores if 0.4 <= s < 0.6]),
        "poor": len([s for s in rouge_scores if s < 0.4])
    }

    total = len(rouge_scores)
    percentages = {k: (v / total) * 100 for k, v in categories.items()}

    return {
        "counts": categories,
        "percentages": percentages
    }
```

### 3. End-to-End RAG Analysis

Comprehensive RAG system analysis:

```python
def analyze_rag_performance(rag_results):
    """
    Comprehensive end-to-end RAG analysis.
    """
    analysis = {
        "overall_performance": {
            "rag_score_distribution": calculate_descriptive_stats({"rag_score": rag_results["rag_scores"]}),
            "performance_categories": categorize_rag_performance(rag_results["rag_scores"])
        },
        "component_analysis": {
            "retrieval_contribution": analyze_component_contribution(
                rag_results["retrieval_scores"], rag_results["rag_scores"]
            ),
            "generation_contribution": analyze_component_contribution(
                rag_results["generation_scores"], rag_results["rag_scores"]
            )
        },
        "error_analysis": analyze_rag_errors(rag_results),
        "improvement_opportunities": identify_improvement_opportunities(rag_results)
    }

    return analysis

def analyze_component_contribution(component_scores, overall_scores):
    """
    Analyze how component performance contributes to overall performance.
    """
    correlation, p_value = stats.pearsonr(component_scores, overall_scores)

    return {
        "correlation": correlation,
        "p_value": p_value,
        "contribution_strength": interpret_correlation(correlation),
        "explained_variance": correlation ** 2
    }

def identify_improvement_opportunities(rag_results):
    """
    Identify areas for improvement based on performance analysis.
    """
    opportunities = []

    # Check retrieval performance
    if np.mean(rag_results["retrieval_scores"]) < 0.7:
        opportunities.append({
            "area": "retrieval",
            "issue": "low_precision",
            "recommendation": "Consider better chunking strategy or embedding model"
        })

    # Check generation performance
    if np.mean(rag_results["generation_scores"]) < 0.6:
        opportunities.append({
            "area": "generation",
            "issue": "low_quality",
            "recommendation": "Improve prompting strategy or use better language model"
        })

    # Check consistency
    if np.std(rag_results["rag_scores"]) > 0.2:
        opportunities.append({
            "area": "consistency",
            "issue": "high_variance",
            "recommendation": "Investigate and reduce performance variability"
        })

    return opportunities
```

## Visualization Framework

### 1. Performance Visualizations

Create comprehensive performance visualizations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(results, output_dir):
    """
    Create a comprehensive performance dashboard.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RAG System Performance Dashboard', fontsize=16)

    # 1. Latency distribution
    axes[0, 0].hist(results["latency_data"], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Latency Distribution')
    axes[0, 0].set_xlabel('Latency (ms)')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Throughput over time
    if "throughput_time_series" in results:
        axes[0, 1].plot(results["throughput_time_series"], color='green')
        axes[0, 1].set_title('Throughput Over Time')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Queries/Second')

    # 3. Memory usage
    axes[0, 2].plot(results["memory_usage"], color='red')
    axes[0, 2].set_title('Memory Usage')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Memory (MB)')

    # 4. Quality metrics comparison
    metrics = ['precision_at_5', 'recall_at_5', 'rouge_l', 'bleu_score']
    values = [results[metric] for metric in metrics]

    axes[1, 0].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    axes[1, 0].set_title('Quality Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Performance vs Quality scatter
    axes[1, 1].scatter(results["latency_data"], results["rag_scores"], alpha=0.6)
    axes[1, 1].set_title('Performance vs Quality')
    axes[1, 1].set_xlabel('Latency (ms)')
    axes[1, 1].set_ylabel('RAG Score')

    # 6. Error analysis
    error_categories = list(results["error_analysis"].keys())
    error_counts = list(results["error_analysis"].values())

    axes[1, 2].pie(error_counts, labels=error_categories, autopct='%1.1f%%')
    axes[1, 2].set_title('Error Distribution')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_visualizations(experiment_results, output_dir):
    """
    Create visualizations comparing multiple experiments.
    """
    # Radar chart for multi-dimensional comparison
    create_radar_chart(experiment_results, output_dir)

    # Box plots for metric distributions
    create_metric_boxplots(experiment_results, output_dir)

    # Heatmap for correlation analysis
    create_correlation_heatmap(experiment_results, output_dir)

def create_radar_chart(experiment_results, output_dir):
    """
    Create radar chart for multi-dimensional comparison.
    """
    import numpy as np

    # Define metrics for radar chart
    metrics = ['precision_at_5', 'recall_at_5', 'rouge_l', 'bleu_score', 'avg_latency_normalized']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (exp_name, results) in enumerate(experiment_results.items()):
        values = []
        for metric in metrics:
            if metric == 'avg_latency_normalized':
                # Normalize latency (lower is better)
                max_latency = max([r['avg_latency_ms'] for r in experiment_results.values()])
                normalized = 1 - (results['avg_latency_ms'] / max_latency)
                values.append(normalized)
            else:
                values.append(results[metric])

        values = np.concatenate((values, [values[0]]))  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=exp_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Dimensional Performance Comparison', size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.savefig(f"{output_dir}/radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
```

### 2. Statistical Visualizations

Create statistical analysis visualizations:

```python
def create_statistical_visualizations(statistical_results, output_dir):
    """
    Create visualizations for statistical analysis results.
    """
    # Confidence intervals plot
    create_confidence_intervals_plot(statistical_results, output_dir)

    # P-value heatmap
    create_pvalue_heatmap(statistical_results, output_dir)

    # Effect size visualization
    create_effect_size_plot(statistical_results, output_dir)

def create_confidence_intervals_plot(statistical_results, output_dir):
    """
    Create confidence intervals visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    experiments = list(statistical_results.keys())
    metrics = list(statistical_results[experiments[0]]['confidence_intervals'].keys())

    y_pos = np.arange(len(metrics))

    for i, exp in enumerate(experiments):
        means = []
        lower_bounds = []
        upper_bounds = []

        for metric in metrics:
            ci = statistical_results[exp]['confidence_intervals'][metric]
            means.append(ci['mean'])
            lower_bounds.append(ci['lower_bound'])
            upper_bounds.append(ci['upper_bound'])

        # Calculate error bars
        lower_errors = [means[j] - lower_bounds[j] for j in range(len(means))]
        upper_errors = [upper_bounds[j] - means[j] for j in range(len(means))]

        ax.errorbar(means, y_pos + i * 0.1,
                   xerr=[lower_errors, upper_errors],
                   fmt='o', label=exp, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Metric Value')
    ax.set_title('Confidence Intervals Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_intervals.png", dpi=300, bbox_inches='tight')
    plt.close()
```

## Reporting Framework

### 1. Executive Summary Generation

Generate executive summaries:

```python
def generate_executive_summary(analysis_results):
    """
    Generate executive summary of analysis results.
    """
    summary = {
        "key_findings": [],
        "performance_highlights": {},
        "recommendations": [],
        "statistical_significance": {}
    }

    # Key findings
    best_config = find_best_configuration(analysis_results)
    summary["key_findings"].append(f"Best performing configuration: {best_config['name']}")

    # Performance highlights
    summary["performance_highlights"] = {
        "highest_rag_score": max([r['rag_score'] for r in analysis_results.values()]),
        "lowest_latency": min([r['avg_latency_ms'] for r in analysis_results.values()]),
        "most_consistent": find_most_consistent_config(analysis_results)
    }

    # Recommendations
    summary["recommendations"] = generate_recommendations(analysis_results)

    return summary

def generate_recommendations(analysis_results):
    """
    Generate actionable recommendations based on analysis.
    """
    recommendations = []

    # Performance recommendations
    high_latency_configs = [name for name, results in analysis_results.items()
                           if results['avg_latency_ms'] > 1000]

    if high_latency_configs:
        recommendations.append({
            "category": "performance",
            "priority": "high",
            "issue": "High latency detected",
            "affected_configs": high_latency_configs,
            "recommendation": "Consider using faster embedding models or optimizing vector store configuration"
        })

    # Quality recommendations
    low_quality_configs = [name for name, results in analysis_results.items()
                          if results['rag_score'] < 0.6]

    if low_quality_configs:
        recommendations.append({
            "category": "quality",
            "priority": "medium",
            "issue": "Low RAG scores detected",
            "affected_configs": low_quality_configs,
            "recommendation": "Improve chunking strategy or fine-tune language model prompts"
        })

    return recommendations
```

### 2. Detailed Report Generation

Generate comprehensive reports:

```python
def generate_detailed_report(analysis_results, output_path):
    """
    Generate detailed analysis report.
    """
    report = {
        "executive_summary": generate_executive_summary(analysis_results),
        "methodology": {
            "statistical_methods": describe_statistical_methods(),
            "evaluation_metrics": describe_evaluation_metrics(),
            "experimental_design": describe_experimental_design()
        },
        "detailed_results": analysis_results,
        "statistical_analysis": perform_comprehensive_statistical_analysis(analysis_results),
        "visualizations": generate_visualization_descriptions(),
        "conclusions": generate_conclusions(analysis_results),
        "future_work": suggest_future_work(analysis_results)
    }

    # Save report
    with open(f"{output_path}/detailed_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Generate markdown report
    markdown_report = convert_to_markdown(report)
    with open(f"{output_path}/analysis_report.md", 'w') as f:
        f.write(markdown_report)

    return report
```

## Advanced Analysis Techniques

### 1. Correlation Analysis

Analyze correlations between different metrics:

```python
def perform_correlation_analysis(results_matrix):
    """
    Perform comprehensive correlation analysis.
    """
    import pandas as pd

    # Create correlation matrix
    df = pd.DataFrame(results_matrix)
    correlation_matrix = df.corr()

    # Find strong correlations
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # Strong correlation threshold
                strong_correlations.append({
                    "metric1": correlation_matrix.columns[i],
                    "metric2": correlation_matrix.columns[j],
                    "correlation": corr_value,
                    "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                })

    return {
        "correlation_matrix": correlation_matrix,
        "strong_correlations": strong_correlations,
        "interpretation": interpret_correlations(strong_correlations)
    }
```

### 2. Clustering Analysis

Identify patterns in experiment results:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering_analysis(results_matrix, n_clusters=3):
    """
    Perform clustering analysis on experiment results.
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(results_matrix)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Analyze clusters
    cluster_analysis = {}
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_data = results_matrix[cluster_mask]

        cluster_analysis[f"cluster_{i}"] = {
            "size": np.sum(cluster_mask),
            "characteristics": analyze_cluster_characteristics(cluster_data),
            "representative_configs": find_representative_configs(cluster_data)
        }

    return {
        "cluster_labels": cluster_labels,
        "cluster_analysis": cluster_analysis,
        "silhouette_score": calculate_silhouette_score(scaled_data, cluster_labels)
    }
```

### 3. Sensitivity Analysis

Analyze parameter sensitivity:

```python
def perform_sensitivity_analysis(parameter_results):
    """
    Perform sensitivity analysis for different parameters.
    """
    sensitivity_results = {}

    for param_name, param_variations in parameter_results.items():
        # Calculate sensitivity metrics
        param_values = list(param_variations.keys())
        metric_values = [param_variations[pv]['rag_score'] for pv in param_values]

        # Calculate sensitivity coefficient
        sensitivity_coeff = calculate_sensitivity_coefficient(param_values, metric_values)

        sensitivity_results[param_name] = {
            "sensitivity_coefficient": sensitivity_coeff,
            "parameter_importance": classify_parameter_importance(sensitivity_coeff),
            "optimal_value": param_values[np.argmax(metric_values)],
            "value_range": [min(param_values), max(param_values)],
            "performance_range": [min(metric_values), max(metric_values)]
        }

    return sensitivity_results

def calculate_sensitivity_coefficient(param_values, metric_values):
    """
    Calculate sensitivity coefficient using normalized derivatives.
    """
    # Normalize values
    param_range = max(param_values) - min(param_values)
    metric_range = max(metric_values) - min(metric_values)

    if param_range == 0 or metric_range == 0:
        return 0

    # Calculate approximate derivative
    derivatives = []
    for i in range(1, len(param_values)):
        dparam = param_values[i] - param_values[i-1]
        dmetric = metric_values[i] - metric_values[i-1]

        if dparam != 0:
            derivative = dmetric / dparam
            derivatives.append(derivative)

    # Normalize by ranges
    avg_derivative = np.mean(derivatives) if derivatives else 0
    sensitivity = abs(avg_derivative) * (param_range / metric_range)

    return sensitivity
```

## Quality Assurance

### 1. Result Validation

Validate analysis results:

```python
def validate_analysis_results(analysis_results):
    """
    Validate analysis results for consistency and accuracy.
    """
    validation_errors = []

    # Check for missing data
    required_metrics = ['rag_score', 'precision_at_5', 'recall_at_5', 'rouge_l']
    for exp_name, results in analysis_results.items():
        for metric in required_metrics:
            if metric not in results:
                validation_errors.append(f"Missing {metric} in {exp_name}")

    # Check metric ranges
    for exp_name, results in analysis_results.items():
        for metric, value in results.items():
            if metric.endswith('_score') or metric.startswith('precision') or metric.startswith('recall'):
                if not (0 <= value <= 1):
                    validation_errors.append(f"Invalid {metric} value in {exp_name}: {value}")

    # Check statistical consistency
    if len(analysis_results) > 1:
        # Check for unrealistic differences
        rag_scores = [r['rag_score'] for r in analysis_results.values()]
        if max(rag_scores) - min(rag_scores) > 0.8:
            validation_errors.append("Unrealistic RAG score differences detected")

    return validation_errors
```

### 2. Reproducibility Checks

Ensure analysis reproducibility:

```python
def check_reproducibility(original_results, reproduced_results, tolerance=0.05):
    """
    Check if analysis results are reproducible.
    """
    reproducibility_report = {
        "reproducible_metrics": [],
        "non_reproducible_metrics": [],
        "overall_reproducible": True
    }

    for metric in original_results:
        if metric in reproduced_results:
            original_value = original_results[metric]
            reproduced_value = reproduced_results[metric]

            if isinstance(original_value, (int, float)) and isinstance(reproduced_value, (int, float)):
                relative_error = abs(original_value - reproduced_value) / abs(original_value)

                if relative_error <= tolerance:
                    reproducibility_report["reproducible_metrics"].append({
                        "metric": metric,
                        "original": original_value,
                        "reproduced": reproduced_value,
                        "relative_error": relative_error
                    })
                else:
                    reproducibility_report["non_reproducible_metrics"].append({
                        "metric": metric,
                        "original": original_value,
                        "reproduced": reproduced_value,
                        "relative_error": relative_error
                    })
                    reproducibility_report["overall_reproducible"] = False

    return reproducibility_report
```

## Conclusion

This comprehensive results analysis framework provides:

1. **Statistical rigor** through proper significance testing and confidence intervals
2. **Visual insights** through comprehensive visualization tools
3. **Actionable recommendations** based on systematic analysis
4. **Quality assurance** through validation and reproducibility checks
5. **Advanced techniques** for deeper understanding of system behavior

By following these guidelines and using the provided tools, researchers can conduct thorough, reliable, and insightful analysis of their RAG system experiments.

For more information, see:
- [Architecture Documentation](architecture.md)
- [API Reference](api_reference.md)
- [Experiment Logs](experiment_logs.md)