"""
I created this script to generate comprehensive visualizations and analysis for the dataset.

Usage:
    python 00_dataset_visualization.py --base_path /home/ubuntu
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setting up the arguments
parser = argparse.ArgumentParser(description='Generate dataset visualizations')
parser.add_argument('--base_path', type=str, default='/home/ubuntu')
args = parser.parse_args()

BASE_PATH = args.base_path

# Configuring the plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("DATASET VISUALIZATION AND ANALYSIS")
print("="*80)
print(f"Base path: {BASE_PATH}")

# Making sure the viz directory exists
viz_dir = f'{BASE_PATH}/visualizations'
os.makedirs(viz_dir, exist_ok=True)

print("\n" + "="*80)
print("STEP 1: LOAD ALL DATASETS")
print("="*80)

# Loading the original data from the file
original_sentences = []
with open(f'{BASE_PATH}/data/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt', 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('@')
        if len(parts) == 2:
            original_sentences.append(parts[0].strip())

print(f"Original Financial PhraseBank: {len(original_sentences)} sentences")

# Loading all the processed CSV files
full_df = pd.read_csv(f'{BASE_PATH}/data/full_dataset.csv')
balanced_df = pd.read_csv(f'{BASE_PATH}/data/balanced_subset.csv')
train_df = pd.read_csv(f'{BASE_PATH}/data/train.csv')
val_df = pd.read_csv(f'{BASE_PATH}/data/val.csv')
test_df = pd.read_csv(f'{BASE_PATH}/data/test.csv')

print(f"Full dataset: {len(full_df)} samples")
print(f"Balanced dataset: {len(balanced_df)} samples")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

print("\n" + "="*80)
print("STEP 2: GENERATE VISUALIZATIONS")
print("="*80)

# Plotting sentiment distributions for original and balanced datasets
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

full_df['sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#95a5a6', '#2ecc71'])
axes[0].set_title('Sentiment Distribution - Original Dataset', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
for i, v in enumerate(full_df['sentiment'].value_counts().values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

balanced_df['sentiment'].value_counts().plot(kind='bar', ax=axes[1], color=['#e74c3c', '#95a5a6', '#2ecc71'])
axes[1].set_title('Sentiment Distribution - Balanced Dataset', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
for i, v in enumerate(balanced_df['sentiment'].value_counts().values):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/sentiment_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Sentiment distribution comparison saved")

# Plotting sensitivity distributions before and after balancing
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

full_df['sensitivity_label'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='#3498db')
axes[0].set_title('Sensitivity Distribution - Original Dataset (Imbalanced)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sensitivity Level')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
for i, v in enumerate(full_df['sensitivity_label'].value_counts().sort_index().values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

balanced_df['sensitivity_label'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='#2ecc71')
axes[1].set_title('Sensitivity Distribution - Balanced Dataset (After Oversampling)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sensitivity Level')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
for i, v in enumerate(balanced_df['sensitivity_label'].value_counts().sort_index().values):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/sensitivity_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Sensitivity distribution comparison saved")

# Creating a heatmap for sensitivity by sentiment
sensitivity_sentiment = pd.crosstab(full_df['sensitivity_label'], full_df['sentiment'])
plt.figure(figsize=(10, 8))
sns.heatmap(sensitivity_sentiment, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('Sensitivity Level by Sentiment - Original Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Sensitivity Level')
plt.tight_layout()
plt.savefig(f'{viz_dir}/sensitivity_by_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Sensitivity by sentiment heatmap saved")

# Plotting the word count distribution
plt.figure(figsize=(12, 6))
plt.hist(full_df['word_count'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
plt.axvline(full_df['word_count'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {full_df["word_count"].mean():.1f}')
plt.axvline(full_df['word_count'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {full_df["word_count"].median():.1f}')
plt.title('Word Count Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/word_count_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Word count distribution saved")

# Visualizing the train/val/test split distribution
split_data = {
    'Train': [len(train_df[train_df['sensitivity_label'] == label]) for label in ['Public', 'Internal', 'Confidential', 'Restricted']],
    'Validation': [len(val_df[val_df['sensitivity_label'] == label]) for label in ['Public', 'Internal', 'Confidential', 'Restricted']],
    'Test': [len(test_df[test_df['sensitivity_label'] == label]) for label in ['Public', 'Internal', 'Confidential', 'Restricted']]
}

x = np.arange(4)
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, split_data['Train'], width, label='Train', color='#3498db')
bars2 = ax.bar(x, split_data['Validation'], width, label='Validation', color='#e67e22')
bars3 = ax.bar(x + width, split_data['Test'], width, label='Test', color='#2ecc71')

ax.set_xlabel('Sensitivity Level', fontsize=12)
ax.set_ylabel('Sample Count', fontsize=12)
ax.set_title('Stratified Train/Validation/Test Split Distribution', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Public', 'Internal', 'Confidential', 'Restricted'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{viz_dir}/train_val_test_split.png', dpi=300, bbox_inches='tight')
plt.close()
print("Train/val/test split visualization saved")

# Creating a pie chart for split proportions
split_sizes = [len(train_df), len(val_df), len(test_df)]
split_labels = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
colors = ['#3498db', '#e67e22', '#2ecc71']
explode = (0.05, 0, 0)

plt.figure(figsize=(10, 8))
plt.pie(split_sizes, explode=explode, labels=split_labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Dataset Split Proportions', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f'{viz_dir}/split_proportions_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("Split proportions pie chart saved")

# Making a stacked bar for sentiment composition by sensitivity level
fig, ax = plt.subplots(figsize=(10, 6))
colors_stack = ['#e74c3c', '#95a5a6', '#2ecc71']

sentiment_by_level = {}
for level in ['Public', 'Internal', 'Confidential', 'Restricted']:
    level_data = full_df[full_df['sensitivity_label'] == level]
    sentiment_by_level[level] = [
        len(level_data[level_data['sentiment'] == 'Negative']),
        len(level_data[level_data['sentiment'] == 'Neutral']),
        len(level_data[level_data['sentiment'] == 'Positive'])
    ]

bottom = np.zeros(4)
for idx, sentiment in enumerate(['Negative', 'Neutral', 'Positive']):
    values = [sentiment_by_level[level][idx] for level in ['Public', 'Internal', 'Confidential', 'Restricted']]
    ax.bar(['Public', 'Internal', 'Confidential', 'Restricted'], values, bottom=bottom, label=sentiment, color=colors_stack[idx])
    bottom += values

ax.set_ylabel('Count', fontsize=12)
ax.set_title('Sentiment Composition by Sensitivity Level', fontsize=16, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/sensitivity_sentiment_stacked.png', dpi=300, bbox_inches='tight')
plt.close()
print("Sentiment composition by sensitivity saved")

# Showing the effect of oversampling on class distribution
fig, ax = plt.subplots(figsize=(12, 6))
labels = ['Public', 'Internal', 'Confidential', 'Restricted']
original_counts = [
    len(full_df[full_df['sensitivity_label'] == 'Public']),
    len(full_df[full_df['sensitivity_label'] == 'Internal']),
    len(full_df[full_df['sensitivity_label'] == 'Confidential']),
    len(full_df[full_df['sensitivity_label'] == 'Restricted'])
]
balanced_counts = [1094, 1094, 1094, 1094]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, original_counts, width, label='Original (Imbalanced)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, balanced_counts, width, label='After Oversampling (Balanced)', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Sensitivity Level', fontsize=12)
ax.set_ylabel('Sample Count', fontsize=12)
ax.set_title('Effect of Oversampling on Class Distribution', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/oversampling_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print("Oversampling effect visualization saved")

print("\n" + "="*80)
print("DATASET VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {viz_dir}/")
print("\nGenerated files:")
print("  8 visualization PNG files")
print("\nVisualization files:")
for f in sorted(os.listdir(viz_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
