"""
I created this script to generate comprehensive analysis for the agentic system results.

Usage:
    python 06_agentic_analysis.py --base_path /home/ubuntu
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# Get the arguments
parser = argparse.ArgumentParser(description='Generate agentic system analysis')
parser.add_argument('--base_path', type=str, default='/home/ubuntu')
args = parser.parse_args()

BASE_PATH = args.base_path

# Set the style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("AGENTIC SYSTEM RESULTS ANALYSIS")
print("="*80)
print(f"Base path: {BASE_PATH}")

# Create output directory
analysis_dir = f'{BASE_PATH}/agentic_analysis'
os.makedirs(analysis_dir, exist_ok=True)

print("\n" + "="*80)
print("STEP 1: LOAD AGENTIC SYSTEM RESULTS")
print("="*80)

# Load agentic predictions
agentic_file = f'{BASE_PATH}/results/agentic_system_results.csv'
if not os.path.exists(agentic_file):
    print(f"ERROR: {agentic_file} not found!")
    print("Please run 06_agentic_system.py first to generate predictions.")
    exit(1)

agentic_df = pd.read_csv(agentic_file)
print(f"Loaded agentic predictions: {len(agentic_df)} samples")
print(f"Columns: {list(agentic_df.columns)}")

# Load individual model predictions for comparison
model_files = {
    'FinBERT-FT': f'{BASE_PATH}/results/finbert_predictions.csv',
    'BERT-FT': f'{BASE_PATH}/results/bert_predictions.csv',
    'GPT-4.1-mini-FT': f'{BASE_PATH}/results/gpt_finetuned_predictions.csv'
}

model_predictions = {}
for model_name, filepath in model_files.items():
    if os.path.exists(filepath):
        model_predictions[model_name] = pd.read_csv(filepath)
        print(f"Loaded {model_name}: {len(model_predictions[model_name])} samples")
    else:
        print(f"Warning: {model_name} predictions not found at {filepath}")

# Load test set for ground truth
test_df = pd.read_csv(f'{BASE_PATH}/data/test.csv')
print(f"Loaded test set: {len(test_df)} samples")

print("\n" + "="*80)
print("STEP 2: EXTRACT AGENTIC SYSTEM STATISTICS")
print("="*80)

# Calculate agreement metrics
def calculate_agent_agreement(row):
    """Calculate how many agents agreed on the final prediction"""
    agents = ['finbert_pred', 'bert_pred', 'gpt_pred']
    final_pred = row['orchestrator_pred']
    agreement_count = sum([row[agent] == final_pred for agent in agents if agent in row])
    return agreement_count

if all(col in agentic_df.columns for col in ['finbert_pred', 'bert_pred', 'gpt_pred']):
    agentic_df['agent_agreement'] = agentic_df.apply(calculate_agent_agreement, axis=1)
    unanimous_decisions = len(agentic_df[agentic_df['agent_agreement'] == 3])
    majority_decisions = len(agentic_df[agentic_df['agent_agreement'] == 2])
    split_decisions = len(agentic_df[agentic_df['agent_agreement'] == 1])
    
    print(f"Unanimous (3/3 agents agree): {unanimous_decisions} ({unanimous_decisions/len(agentic_df)*100:.1f}%)")
    print(f"Majority (2/3 agents agree): {majority_decisions} ({majority_decisions/len(agentic_df)*100:.1f}%)")
    print(f"Split (1/3 agents agree): {split_decisions} ({split_decisions/len(agentic_df)*100:.1f}%)")

# Calculate orchestrator override rate
def orchestrator_overrode_majority(row):
    """Check if orchestrator chose differently from majority vote"""
    if 'agent_agreement' not in row:
        return False
    agents = ['finbert_pred', 'bert_pred', 'gpt_pred']
    agent_preds = [row[agent] for agent in agents if agent in row]
    if len(agent_preds) < 3:
        return False
    majority_vote = Counter(agent_preds).most_common(1)[0][0]
    return row['orchestrator_pred'] != majority_vote

if all(col in agentic_df.columns for col in ['finbert_pred', 'bert_pred', 'gpt_pred', 'orchestrator_pred']):
    agentic_df['orchestrator_override'] = agentic_df.apply(orchestrator_overrode_majority, axis=1)
    override_count = agentic_df['orchestrator_override'].sum()
    print(f"\nOrchestrator overrides (chose against majority): {override_count} ({override_count/len(agentic_df)*100:.1f}%)")

# Calculate accuracy metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

label_mapping = {'Public': 0, 'Internal': 1, 'Confidential': 2, 'Restricted': 3}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Ensure predictions and ground truth are aligned
if 'true_label' in agentic_df.columns and 'orchestrator_pred' in agentic_df.columns:
    y_true = agentic_df['true_label'].values
    y_pred_orch = agentic_df['orchestrator_pred'].values
    
    accuracy_orch = accuracy_score(y_true, y_pred_orch)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_orch, average='macro', zero_division=0)
    
    print(f"\nAgentic System (Orchestrator) Performance:")
    print(f"  Accuracy: {accuracy_orch*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    
    # Compare with individual agents
    individual_accuracies = {}
    for agent in ['finbert_pred', 'bert_pred', 'gpt_pred']:
        if agent in agentic_df.columns:
            agent_name = agent.replace('_pred', '').upper()
            y_pred_agent = agentic_df[agent].values
            acc = accuracy_score(y_true, y_pred_agent)
            individual_accuracies[agent_name] = acc
            print(f"\n{agent_name} (Individual) Performance:")
            print(f"  Accuracy: {acc*100:.2f}%")

print("\n" + "="*80)
print("STEP 3: GENERATE LATEX TABLES")
print("="*80)

# Table 1: Agent Agreement Statistics
latex_agreement = r"""\begin{table}[h]
\centering
\caption{Agent Agreement Statistics in Agentic System}
\begin{tabular}{|l|r|r|}
\hline
\textbf{Agreement Type} & \textbf{Count} & \textbf{Percentage} \\
\hline
\hline
Unanimous (3/3 agents agree) & """ + f"{unanimous_decisions}" + r""" & """ + f"{unanimous_decisions/len(agentic_df)*100:.1f}\%" + r""" \\
Majority (2/3 agents agree) & """ + f"{majority_decisions}" + r""" & """ + f"{majority_decisions/len(agentic_df)*100:.1f}\%" + r""" \\
Split (1/3 agents agree) & """ + f"{split_decisions}" + r""" & """ + f"{split_decisions/len(agentic_df)*100:.1f}\%" + r""" \\
\hline
Orchestrator Overrides & """ + f"{override_count}" + r""" & """ + f"{override_count/len(agentic_df)*100:.1f}\%" + r""" \\
\hline
\textbf{Total Predictions} & """ + f"{len(agentic_df)}" + r""" & \textbf{100.0\%} \\
\hline
\end{tabular}
\end{table}"""

# Table 2: Model Performance Comparison
if 'true_label' in agentic_df.columns:
    # Calculate metrics for all models
    all_metrics = []
    
    # Orchestrator
    y_true = agentic_df['true_label'].values
    y_pred = agentic_df['orchestrator_pred'].values
    acc = accuracy_score(y_true, y_pred) * 100
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    all_metrics.append(['Agentic System (Orchestrator)', acc, p*100, r*100, f1*100])
    
    # Individual agents
    agent_names = {
        'finbert_pred': 'FinBERT-FT',
        'bert_pred': 'BERT-FT',
        'gpt_pred': 'GPT-4.1-mini-FT'
    }
    
    for agent_col, agent_name in agent_names.items():
        if agent_col in agentic_df.columns:
            y_pred = agentic_df[agent_col].values
            acc = accuracy_score(y_true, y_pred) * 100
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            all_metrics.append([agent_name, acc, p*100, r*100, f1*100])
    
    # Sort by accuracy descending
    all_metrics.sort(key=lambda x: x[1], reverse=True)
    
    latex_performance = r"""\begin{table}[h]
\centering
\caption{Model Performance Comparison - Agentic System vs Individual Agents}
\begin{tabular}{|l|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Accuracy (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Score (\%)} \\
\hline
\hline
"""
    
    for metric in all_metrics:
        latex_performance += f"{metric[0]} & {metric[1]:.2f} & {metric[2]:.2f} & {metric[3]:.2f} & {metric[4]:.2f} \\\\\n"
    
    latex_performance += r"""\hline
\end{tabular}
\end{table}"""

# Table 3: Per-Class Performance (Orchestrator)
if 'true_label' in agentic_df.columns and 'orchestrator_pred' in agentic_df.columns:
    y_true = agentic_df['true_label'].values
    y_pred = agentic_df['orchestrator_pred'].values
    
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3])
    
    latex_class_perf = r"""\begin{table}[h]
\centering
\caption{Per-Class Performance - Agentic System Orchestrator}
\begin{tabular}{|l|r|r|r|r|}
\hline
\textbf{Class} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Score (\%)} & \textbf{Support} \\
\hline
\hline
"""
    
    class_names = ['Public', 'Internal', 'Confidential', 'Restricted']
    for i, class_name in enumerate(class_names):
        latex_class_perf += f"{class_name} & {p[i]*100:.2f} & {r[i]*100:.2f} & {f1[i]*100:.2f} & {int(support[i])} \\\\\n"
    
    latex_class_perf += r"""\hline
\textbf{Macro Average} & """ + f"{np.mean(p)*100:.2f}" + r""" & """ + f"{np.mean(r)*100:.2f}" + r""" & """ + f"{np.mean(f1)*100:.2f}" + r""" & """ + f"{int(np.sum(support))}" + r""" \\
\hline
\end{tabular}
\end{table}"""

print("\n" + "="*80)
print("STEP 4: GENERATE VISUALIZATIONS")
print("="*80)

# 1. Agent Agreement Distribution (Pie Chart)
if 'agent_agreement' in agentic_df.columns:
    agreement_counts = agentic_df['agent_agreement'].value_counts().sort_index()
    labels = [f'{count}/3 Agents Agree' for count in agreement_counts.index]
    sizes = agreement_counts.values
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    explode = tuple([0.05 * i for i in range(len(sizes))])  # Dynamic explode based on number of categories
    
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Agent Agreement Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Add count labels
    for i, (label, size) in enumerate(zip(labels, sizes)):
        texts[i].set_text(f'{label}\n({size} samples)')
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/agent_agreement_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Agent agreement pie chart saved")

# 2. Model Performance Comparison (Bar Chart)
if all_metrics:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = [m[0] for m in all_metrics]
    accuracies = [m[1] for m in all_metrics]
    f1_scores = [m[4] for m in all_metrics]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Agentic System vs Individual Agents', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/model_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Model comparison bar chart saved")

# 3. Confusion Matrix - Orchestrator
if 'true_label' in agentic_df.columns and 'orchestrator_pred' in agentic_df.columns:
    y_true = agentic_df['true_label'].values
    y_pred = agentic_df['orchestrator_pred'].values
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    class_names = ['Public', 'Internal', 'Confidential', 'Restricted']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Confusion Matrix - Agentic System (Orchestrator)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/confusion_matrix_orchestrator.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved")

# 4. Per-Class F1-Score Comparison (Grouped Bar Chart)
if 'true_label' in agentic_df.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    class_names = ['Public', 'Internal', 'Confidential', 'Restricted']
    x = np.arange(len(class_names))
    
    # Calculate F1 for each model and class
    model_f1_scores = {}
    
    # Orchestrator
    y_true = agentic_df['true_label'].values
    y_pred = agentic_df['orchestrator_pred'].values
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3])
    model_f1_scores['Orchestrator'] = f1 * 100
    
    # Individual agents
    for agent_col, agent_name in agent_names.items():
        if agent_col in agentic_df.columns:
            y_pred = agentic_df[agent_col].values
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3])
            model_f1_scores[agent_name] = f1 * 100
    
    # Plot grouped bars
    n_models = len(model_f1_scores)
    width = 0.8 / n_models
    colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, (model_name, f1_scores) in enumerate(model_f1_scores.items()):
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=model_name, color=colors_list[i % len(colors_list)], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Sensitivity Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Per-class F1 comparison saved")

# 5. Orchestrator Override Analysis
if 'orchestrator_override' in agentic_df.columns:
    override_df = agentic_df[agentic_df['orchestrator_override'] == True]
    
    if len(override_df) > 0:
        # Check if overrides were correct
        override_df['override_correct'] = override_df['orchestrator_pred'] == override_df['true_label']
        
        correct_overrides = override_df['override_correct'].sum()
        total_overrides = len(override_df)
        
        labels = ['Correct Override', 'Incorrect Override']
        sizes = [correct_overrides, total_overrides - correct_overrides]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        plt.title(f'Orchestrator Override Accuracy\n({total_overrides} total overrides)', fontsize=16, fontweight='bold', pad=20)
        
        # Add count labels
        for i, (label, size) in enumerate(zip(labels, sizes)):
            texts[i].set_text(f'{label}\n({size} samples)')
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{analysis_dir}/orchestrator_override_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Orchestrator override accuracy chart saved")

# 6. Agreement vs Accuracy Analysis
if 'agent_agreement' in agentic_df.columns and 'true_label' in agentic_df.columns:
    agreement_accuracy = []
    for agreement_level in sorted(agentic_df['agent_agreement'].unique()):
        subset = agentic_df[agentic_df['agent_agreement'] == agreement_level]
        if len(subset) > 0:
            acc = accuracy_score(subset['true_label'], subset['orchestrator_pred']) * 100
            agreement_accuracy.append((agreement_level, acc, len(subset)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    levels = [x[0] for x in agreement_accuracy]
    accuracies = [x[1] for x in agreement_accuracy]
    counts = [x[2] for x in agreement_accuracy]
    
    bars = ax.bar(levels, accuracies, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Number of Agents in Agreement', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Agent Agreement Level', fontsize=14, fontweight='bold')
    ax.set_xticks(levels)
    ax.set_xticklabels([f'{level}/3' for level in levels])
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels
    for bar, acc, count in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/agreement_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Agreement vs accuracy chart saved")

# 7. Agent Contribution Heatmap
if all(col in agentic_df.columns for col in ['finbert_pred', 'bert_pred', 'gpt_pred', 'orchestrator_pred', 'true_label']):
    # Calculate how often each agent's prediction matches the final correct answer
    agent_cols = ['finbert_pred', 'bert_pred', 'gpt_pred']
    agent_display_names = ['FinBERT-FT', 'BERT-FT', 'GPT-4.1-mini-FT']
    class_names = ['Public', 'Internal', 'Confidential', 'Restricted']
    
    contribution_matrix = np.zeros((len(agent_cols), 4))
    
    for class_idx, class_name in enumerate(class_names):
        class_subset = agentic_df[agentic_df['true_label'] == class_idx]
        for agent_idx, agent_col in enumerate(agent_cols):
            correct_preds = (class_subset[agent_col] == class_idx).sum()
            contribution_matrix[agent_idx, class_idx] = correct_preds / len(class_subset) * 100 if len(class_subset) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(contribution_matrix, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=class_names,
                yticklabels=agent_display_names, cbar_kws={'label': 'Accuracy (%)'}, ax=ax, vmin=0, vmax=100)
    ax.set_title('Agent Contribution: Per-Class Accuracy Heatmap', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Sensitivity Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agent', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/agent_contribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Agent contribution heatmap saved")

# 8. Prediction Confidence Analysis (if reasoning is available)
if 'reasoning' in agentic_df.columns:
    # Analyze reasoning length as proxy for confidence/complexity
    agentic_df['reasoning_length'] = agentic_df['reasoning'].fillna('').str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot by class
    class_names = ['Public', 'Internal', 'Confidential', 'Restricted']
    reasoning_by_class = [agentic_df[agentic_df['true_label'] == i]['reasoning_length'].values for i in range(4)]
    
    bp = axes[0].boxplot(reasoning_by_class, labels=class_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_xlabel('Sensitivity Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Reasoning Length (characters)', fontsize=12, fontweight='bold')
    axes[0].set_title('Orchestrator Reasoning Length by Class', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Histogram
    axes[1].hist(agentic_df['reasoning_length'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1].axvline(agentic_df['reasoning_length'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {agentic_df["reasoning_length"].mean():.0f}')
    axes[1].axvline(agentic_df['reasoning_length'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {agentic_df["reasoning_length"].median():.0f}')
    axes[1].set_xlabel('Reasoning Length (characters)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Reasoning Length', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/reasoning_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Reasoning analysis chart saved")

print("\n" + "="*80)
print("AGENTIC SYSTEM ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {analysis_dir}/")
print("\nGenerated files:")
print("  - 8 high-resolution visualizations (.png)")
print("\nVisualization files:")
for f in sorted(os.listdir(analysis_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
