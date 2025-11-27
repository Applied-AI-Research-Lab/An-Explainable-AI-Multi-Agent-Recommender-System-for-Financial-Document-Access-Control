"""
I created this script to compare all the fine-tuned models.

Usage:
    python 05_model_comparison.py --base_path /path/to/project
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import mcnemar
import warnings
warnings.filterwarnings('ignore')

# Setting up the arguments
parser = argparse.ArgumentParser(description='Compare all models')
parser.add_argument('--base_path', type=str, default='/workspace/AgenticFinance')
args = parser.parse_args()

BASE_PATH = args.base_path

print("="*80)
print("MODEL COMPARISON AND ANALYSIS")
print("="*80)
print(f"Base path: {BASE_PATH}")

print("\n" + "="*80)
print("STEP 1: LOAD ALL PREDICTIONS")
print("="*80)

finbert_df = pd.read_csv(f'{BASE_PATH}/results/finbert_predictions.csv')
bert_df = pd.read_csv(f'{BASE_PATH}/results/bert_predictions.csv')
llama3_df = pd.read_csv(f'{BASE_PATH}/results/llama3_predictions.csv')
mistral_df = pd.read_csv(f'{BASE_PATH}/results/mistral_predictions.csv')

print("Loaded predictions from 4 models")
print(f"  - FinBERT: {len(finbert_df)} samples")
print(f"  - BERT: {len(bert_df)} samples")
print(f"  - LLaMA-3: {len(llama3_df)} samples")
print(f"  - Mistral: {len(mistral_df)} samples")

print("\n" + "="*80)
print("STEP 2: CALCULATE METRICS")
print("="*80)

models = {
    'FinBERT': finbert_df,
    'BERT': bert_df,
    'LLaMA-3': llama3_df,
    'Mistral': mistral_df
}

results = []

for model_name, df in models.items():
    y_true = df['true_sensitivity_level'].values
    y_pred = df['predicted_sensitivity_level'].values
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1 (Macro)': f1_macro,
        'F1 (Weighted)': f1_weighted,
        'Precision': precision,
        'Recall': recall
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

results_df.to_csv(f'{BASE_PATH}/results/model_comparison.csv', index=False)
print(f"\nComparison saved to: {BASE_PATH}/results/model_comparison.csv")

print("\n" + "="*80)
print("STEP 3: CONFUSION MATRICES (2x2 GRID)")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
label_names = ['Public', 'Internal', 'Confidential', 'Restricted']

for idx, (model_name, df) in enumerate(models.items()):
    row = idx // 2
    col = idx % 2
    
    cm = confusion_matrix(df['true_sensitivity_level'], df['predicted_sensitivity_level'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=label_names, yticklabels=label_names)
    axes[row, col].set_title(f'{model_name}', fontsize=14, fontweight='bold')
    axes[row, col].set_ylabel('True Label')
    axes[row, col].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Confusion matrices saved")

print("\n" + "="*80)
print("STEP 4: PERFORMANCE BAR CHARTS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Comparing the accuracy of models
axes[0].bar(results_df['Model'], results_df['Accuracy'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(results_df['Accuracy']):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Comparing the F1 macro scores
axes[1].bar(results_df['Model'], results_df['F1 (Macro)'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[1].set_title('F1 Score (Macro) Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('F1 Score (Macro)')
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(results_df['F1 (Macro)']):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/performance_comparison.png', dpi=300, bbox_inches='tight')
print("Performance charts saved")

print("\n" + "="*80)
print("STEP 5: McNEMAR'S TEST")
print("="*80)

print("\nComparing all models against FinBERT (baseline):\n")

baseline = finbert_df['predicted_sensitivity_level'].values
y_true = finbert_df['true_sensitivity_level'].values

for model_name, df in list(models.items())[1:]:
    y_pred = df['predicted_sensitivity_level'].values
    
    # Building the contingency table for the test
    correct_both = np.sum((baseline == y_true) & (y_pred == y_true))
    baseline_only = np.sum((baseline == y_true) & (y_pred != y_true))
    model_only = np.sum((baseline != y_true) & (y_pred == y_true))
    incorrect_both = np.sum((baseline != y_true) & (y_pred != y_true))
    
    # Running McNemar's test
    if baseline_only + model_only > 0:
        statistic = (abs(baseline_only - model_only) - 1)**2 / (baseline_only + model_only)
        p_value = 1 - 0.5 * (1 + np.sign(statistic) * np.sqrt(1 - np.exp(-statistic)))
    else:
        statistic, p_value = 0, 1
    
    print(f"{model_name} vs FinBERT:")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n")

print("\n" + "="*80)
print("STEP 6: PER-CLASS PERFORMANCE")
print("="*80)

per_class_results = []

for model_name, df in models.items():
    y_true = df['true_sensitivity_level'].values
    y_pred = df['predicted_sensitivity_level'].values
    
    for class_idx, class_name in enumerate(label_names):
        mask = y_true == class_idx
        if mask.sum() > 0:
            class_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            per_class_results.append({
                'Model': model_name,
                'Class': class_name,
                'Accuracy': class_accuracy,
                'Support': mask.sum()
            })

per_class_df = pd.DataFrame(per_class_results)
pivot_df = per_class_df.pivot(index='Class', columns='Model', values='Accuracy')

print("\nPer-Class Accuracy:")
print(pivot_df.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
pivot_df.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax.set_title('Per-Class Accuracy Across Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Sensitivity Level')
ax.set_ylim([0, 1])
ax.legend(title='Model', bbox_to_anchor=(1.05, 1))
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/per_class_performance.png', dpi=300, bbox_inches='tight')
print("\nPer-class performance chart saved")

print("\n" + "="*80)
print("STEP 7: CONFIDENCE ANALYSIS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (model_name, df) in enumerate(models.items()):
    row = idx // 2
    col = idx % 2
    
    correct = df['true_sensitivity_level'] == df['predicted_sensitivity_level']
    
    axes[row, col].hist([df[correct]['confidence_score'], df[~correct]['confidence_score']],
                       bins=20, label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    axes[row, col].set_title(f'{model_name} - Confidence Distribution', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('Confidence Score')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()
    axes[row, col].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/confidence_analysis.png', dpi=300, bbox_inches='tight')
print("Confidence analysis saved")

print("\n" + "="*80)
print("MODEL COMPARISON COMPLETE!")
print("="*80)
print("\nKey Findings:")
print(f"  - Best Accuracy: {results_df.loc[results_df['Accuracy'].idxmax(), 'Model']} ({results_df['Accuracy'].max():.4f})")
print(f"  - Best F1 (Macro): {results_df.loc[results_df['F1 (Macro)'].idxmax(), 'Model']} ({results_df['F1 (Macro)'].max():.4f})")
print(f"\nAll results saved to: {BASE_PATH}/results/")
print(f"\nNext step: Run 06_agentic_system.py")
