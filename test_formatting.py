def print_metrics_table(metrics):
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 50)
    
    # General
    print(f"{'Accuracy':<25} | {metrics['accuracy']:.4f}")
    print("-" * 50)
    
    # Spike
    print(f"{'Precision (Spike)':<25} | {metrics['precision_spike']:.3f}")
    print(f"{'Recall (Spike)':<25} | {metrics['recall_spike']:.3f}")
    print("-" * 50)

    # Season
    print(f"{'Precision (Season)':<25} | {metrics['precision_season']:.3f}")
    print(f"{'Recall (Season)':<25} | {metrics['recall_season']:.3f}")
    print("-" * 50)

    # Outlier
    print(f"{'Precision (Outlier)':<25} | {metrics['precision_outlier']:.3f}")
    print(f"{'Recall (Outlier)':<25} | {metrics['recall_outlier']:.3f}")
    print("="*50 + "\n")

metrics = {
    "accuracy": 0.949,
    "precision_spike": 0.613,
    "recall_spike": 0.741,
    "precision_season": 0.713,
    "recall_season": 0.792,
    "precision_outlier": 0.843,
    "recall_outlier": 0.670,
}

print("🧮 Local evaluation results:")
print_metrics_table(metrics)
