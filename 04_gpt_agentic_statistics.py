"""
I created this script to extract statistics for GPT predictions from the agentic system results.

Usage:
    python 04_gpt_agentic_statistics.py --base_path /path/to/project
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setting up the arguments
parser = argparse.ArgumentParser(description='Extract GPT statistics from agentic system results')
parser.add_argument('--base_path', type=str, default='/Users/rkonstadinos/PycharmProjects/backup-thunder-compute')
args = parser.parse_args()

BASE_PATH = args.base_path

print("="*80)
print("GPT MODEL STATISTICS FROM AGENTIC SYSTEM RESULTS")
print("="*80)
print(f"Base path: {BASE_PATH}")

print("\n" + "="*80)
print("STEP 1: LOAD AGENTIC SYSTEM RESULTS")
print("="*80)

results_df = pd.read_csv(f'{BASE_PATH}/results/agentic_system_results.csv')
print(f"Total samples: {len(results_df)}")
print(f"\nColumns available: {list(results_df.columns)}")

# Showing the first few rows to understand the data
print("\nFirst few rows:")
print(results_df[['true_label', 'true_label_name', 'gpt_pred', 'gpt_conf']].head())

print("\n" + "="*80)
print("STEP 2: EXTRACT GPT STATISTICS")
print("="*80)

# Pulling out the GPT predictions and true labels
true_labels = results_df['true_label'].values
gpt_predictions = results_df['gpt_pred'].values
gpt_confidence = results_df['gpt_conf'].values

print(f"True labels shape: {true_labels.shape}")
print(f"GPT predictions shape: {gpt_predictions.shape}")
print(f"GPT confidence shape: {gpt_confidence.shape}")

# Mapping the labels to names
label_map = {0: 'Public', 1: 'Internal', 2: 'Confidential', 3: 'Restricted'}

print("\n" + "="*80)
print("STEP 3: CALCULATE METRICS")
print("="*80)

# Computing the metrics
accuracy = accuracy_score(true_labels, gpt_predictions)
f1_macro = f1_score(true_labels, gpt_predictions, average='macro')
f1_weighted = f1_score(true_labels, gpt_predictions, average='weighted')
precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, gpt_predictions, average='macro'
)

print(f"GPT Model Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 (macro): {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")
print(f"  Precision (macro): {precision:.4f}")
print(f"  Recall (macro): {recall:.4f}")

# Calculating metrics for each class
print("\nPer-class metrics:")
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    true_labels, gpt_predictions, average=None
)

for i, label_name in label_map.items():
    if i < len(precision_per_class):
        print(f"  {label_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall: {recall_per_class[i]:.4f}")
        print(f"    F1-score: {f1_per_class[i]:.4f}")
        print(f"    Support: {support_per_class[i]}")

# Analyzing the confidence scores
print("\nConfidence Statistics:")
print(f"  Mean confidence: {gpt_confidence.mean():.4f}")
print(f"  Median confidence: {np.median(gpt_confidence):.4f}")
print(f"  Std confidence: {gpt_confidence.std():.4f}")
print(f"  Min confidence: {gpt_confidence.min():.4f}")
print(f"  Max confidence: {gpt_confidence.max():.4f}")

print("\n" + "="*80)
print("STEP 4: GENERATE CONFUSION MATRIX")
print("="*80)

# Creating the confusion matrix
cm = confusion_matrix(true_labels, gpt_predictions)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Public', 'Internal', 'Confidential', 'Restricted'],
    yticklabels=['Public', 'Internal', 'Confidential', 'Restricted']
)
plt.title('GPT-4.1-mini - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

os.makedirs(f'{BASE_PATH}/results', exist_ok=True)
plt.savefig(f'{BASE_PATH}/results/gpt_agentic_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {BASE_PATH}/results/gpt_agentic_confusion_matrix.png")

print("\n" + "="*80)
print("STEP 5: ANALYZE GPT REASONING")
print("="*80)

# Generating the classification report
report = classification_report(
    true_labels, gpt_predictions,
    target_names=['Public', 'Internal', 'Confidential', 'Restricted'],
    digits=4
)
print("\nClassification Report:")
print(report)

print("\n" + "="*80)
print("STEP 6: ANALYZE GPT REASONING")
print("="*80)

# Showing some examples of GPT reasoning
print("\nSample GPT Reasoning (first 5 examples):")
for idx in range(min(5, len(results_df))):
    print(f"\nExample {idx+1}:")
    print(f"  Text: {results_df.iloc[idx]['text'][:100]}...")
    print(f"  True Label: {results_df.iloc[idx]['true_label_name']}")
    print(f"  GPT Prediction: {label_map[results_df.iloc[idx]['gpt_pred']]}")
    print(f"  GPT Confidence: {results_df.iloc[idx]['gpt_conf']:.4f}")
    print(f"  GPT Reasoning: {results_df.iloc[idx]['gpt_reasoning']}")

# Saving the detailed GPT results to CSV
gpt_results = results_df[['text', 'true_label', 'true_label_name', 'gpt_pred', 'gpt_conf', 'gpt_reasoning']].copy()
gpt_results['gpt_pred_name'] = gpt_results['gpt_pred'].map(label_map)
gpt_results['correct'] = gpt_results['true_label'] == gpt_results['gpt_pred']
gpt_results.to_csv(f'{BASE_PATH}/results/gpt_agentic_detailed_results.csv', index=False)
print(f"\nDetailed GPT results saved to: {BASE_PATH}/results/gpt_agentic_detailed_results.csv")

print("\n" + "="*80)
print("STEP 7: ERROR ANALYSIS")
print("="*80)

# Finding the misclassified samples
errors = results_df[results_df['true_label'] != results_df['gpt_pred']].copy()
print(f"\nTotal errors: {len(errors)} out of {len(results_df)} ({len(errors)/len(results_df)*100:.2f}%)")

if len(errors) > 0:
    print("\nError breakdown by true label:")
    error_by_label = errors.groupby('true_label_name').size()
    for label, count in error_by_label.items():
        total_in_class = len(results_df[results_df['true_label_name'] == label])
        print(f"  {label}: {count}/{total_in_class} ({count/total_in_class*100:.2f}%)")
    
    print("\nSample errors (first 3):")
    for idx in range(min(3, len(errors))):
        err = errors.iloc[idx]
        print(f"\nError {idx+1}:")
        print(f"  Text: {err['text'][:100]}...")
        print(f"  True: {err['true_label_name']}")
        print(f"  GPT Predicted: {label_map[err['gpt_pred']]}")
        print(f"  Confidence: {err['gpt_conf']:.4f}")
        print(f"  Reasoning: {err['gpt_reasoning']}")

print("\n" + "="*80)
print("GPT STATISTICS EXTRACTION COMPLETE!")
print("="*80)
print(f"GPT Model Accuracy: {accuracy:.4f}")
print(f"GPT Model F1 (macro): {f1_macro:.4f}")
print(f"\nOutputs saved to: {BASE_PATH}/results/")
print(f"  - {BASE_PATH}/results/gpt_agentic_confusion_matrix.png")
print(f"  - {BASE_PATH}/results/gpt_agentic_detailed_results.csv")
