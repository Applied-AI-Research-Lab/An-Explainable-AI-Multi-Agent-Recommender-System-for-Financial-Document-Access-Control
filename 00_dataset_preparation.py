"""
Dataset Preparation Script
I wrote this to prepare the Financial PhraseBank dataset with synthetic sensitivity labels.

Usage:
    python 00_dataset_preparation.py --base_path /path/to/project
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setting up command line arguments
parser = argparse.ArgumentParser(description='Prepare Financial PhraseBank dataset')
parser.add_argument('--base_path', type=str, default='/workspace/AgenticFinance',
                    help='Base path for the project (default: /workspace/AgenticFinance)')
parser.add_argument('--samples_per_level', type=int, default=100,
                    help='Number of samples per sensitivity level (default: 100)')
args = parser.parse_args()

BASE_PATH = args.base_path
SAMPLES_PER_LEVEL = args.samples_per_level

# Making sure directories exist
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(f'{BASE_PATH}/data', exist_ok=True)
os.makedirs(f'{BASE_PATH}/FTmodels', exist_ok=True)
os.makedirs(f'{BASE_PATH}/results', exist_ok=True)

print(f"Working directory: {BASE_PATH}")
print("Subdirectories created: data/, FTmodels/, results/")

# Configuring pandas and seaborn display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
sns.set_style('whitegrid')

print("\n" + "="*80)
print("STEP 1: LOADING FINANCIAL PHRASEBANK DATASET")
print("="*80)

# Checking if we already have the full dataset
full_dataset_path = f'{BASE_PATH}/data/full_dataset.csv'
if os.path.exists(full_dataset_path):
    print("Loading from existing full_dataset.csv...")
    df = pd.read_csv(full_dataset_path)
    print(f"\nLoaded {len(df)} sentences from existing full_dataset.csv")
else:
    # Loading the dataset directly from parquet to avoid issues
    print("Loading Financial PhraseBank dataset...")
    dataset = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/takala/financial_phrasebank/resolve/refs%2Fconvert%2Fparquet/sentences_allagree/train-00000-of-00001.parquet"
    )

    # Converting to pandas for easier handling
    df = dataset['train'].to_pandas()

    print(f"\nLoaded {len(df)} sentences from Financial PhraseBank")
    print(f"Columns: {df.columns.tolist()}")

print(f"\nLabel distribution:")
print(df['label'].value_counts().sort_index())
print(f"\nSample sentences:")
print(df.head(3)[['sentence', 'label']])

print("\n" + "="*80)
print("STEP 2: DATA EXPLORATION")
print("="*80)

# Mapping labels to sentiment names
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['sentiment'] = df['label'].map(sentiment_map)

# Calculating some stats on text length
df['text_length'] = df['sentence'].str.len()
df['word_count'] = df['sentence'].str.split().str.len()

print(f"Total samples: {len(df)}")
print(f"\nSentiment distribution:")
print(df['sentiment'].value_counts())
print(f"\nText statistics:")
print(f"  Avg character length: {df['text_length'].mean():.1f}")
print(f"  Avg word count: {df['word_count'].mean():.1f}")
print(f"  Min/Max words: {df['word_count'].min()} / {df['word_count'].max()}")

print("\n" + "="*80)
print("STEP 3: CREATE SENSITIVITY LABELS")
print("="*80)

# Defining keywords for sensitivity levels
sensitivity_keywords = {
    3: ['insider', 'confidential', 'unreleased', 'merger', 'acquisition', 'regulatory', 
        'sec filing', 'material information', 'non-public', 'restricted'],
    2: ['strategic', 'competitive', 'proprietary', 'internal', 'forecast', 'projection',
        'executive', 'board', 'management decision', 'restructuring'],
    1: ['company', 'analysis', 'performance', 'revenue', 'profit', 'loss', 'growth',
        'market share', 'competitor', 'industry'],
    0: ['public', 'announced', 'reported', 'statement', 'press release', 'market',
        'general', 'news', 'sector', 'economic']
}

def assign_sensitivity(text, sentiment):
    """Assign sensitivity level based on keywords and sentiment."""
    text_lower = text.lower()
    
    # Check for highest sensitivity first
    for level in [3, 2, 1, 0]:
        for keyword in sensitivity_keywords[level]:
            if keyword in text_lower:
                # Increase sensitivity if negative news
                if sentiment == 'Negative' and level < 3:
                    return level + 1
                return level
    
    # Default: assign based on sentiment
    if sentiment == 'Negative':
        return np.random.choice([1, 2], p=[0.7, 0.3])
    else:
        return np.random.choice([0, 1], p=[0.6, 0.4])

# Assigning sensitivity levels to each sentence
np.random.seed(42)
df['sensitivity_level'] = df.apply(
    lambda row: assign_sensitivity(row['sentence'], row['sentiment']), axis=1
)

# Converting levels to readable labels
sensitivity_labels = {
    0: 'Public',
    1: 'Internal',
    2: 'Confidential',
    3: 'Restricted'
}
df['sensitivity_label'] = df['sensitivity_level'].map(sensitivity_labels)

print("Sensitivity labels created")
print(f"\nSensitivity distribution:")
print(df['sensitivity_label'].value_counts().sort_index())

print("\n" + "="*80)
print("STEP 4: OVERSAMPLE TO BALANCE CLASSES")
print("="*80)

# Looking at the current class distribution
print("Original sensitivity distribution:")
print(df['sensitivity_label'].value_counts().sort_index())

# Finding the largest class size
max_class_size = df['sensitivity_level'].value_counts().max()
print(f"\nMaximum class size: {max_class_size}")

# Oversampling to balance the classes
balanced_dfs = []
for level in [0, 1, 2, 3]:
    level_data = df[df['sensitivity_level'] == level]
    current_size = len(level_data)
    if current_size < max_class_size:
        # Oversample with replacement
        oversampled = level_data.sample(n=max_class_size, replace=True, random_state=42)
        print(f"Level {level} ({sensitivity_labels[level]}): {current_size} -> {len(oversampled)} (oversampled)")
    else:
        oversampled = level_data
        print(f"Level {level} ({sensitivity_labels[level]}): {current_size} (no oversampling needed)")
    balanced_dfs.append(oversampled)

# Putting all the balanced data together
df_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset created: {len(df_balanced)} total samples")
print(f"\nBalanced sensitivity distribution:")
print(df_balanced['sensitivity_label'].value_counts().sort_index())

print("\n" + "="*80)
print("STEP 5: TRAIN/VALIDATION/TEST SPLIT")
print("="*80)

# Splitting into train/val/test with stratification
train_df, temp_df = train_test_split(
    df_balanced, 
    test_size=0.3, 
    random_state=42, 
    stratify=df_balanced['sensitivity_level']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df['sensitivity_level']
)

print(f"Total balanced size: {len(df_balanced)}")
print(f"\nTrain: {len(train_df)} samples ({len(train_df)/len(df_balanced)*100:.1f}%)")
print(f"Validation: {len(val_df)} samples ({len(val_df)/len(df_balanced)*100:.1f}%)")
print(f"Test: {len(test_df)} samples ({len(test_df)/len(df_balanced)*100:.1f}%)")

print("\nSensitivity distribution in splits:")
print("\nTrain:")
print(train_df['sensitivity_label'].value_counts().sort_index())
print("\nValidation:")
print(val_df['sensitivity_label'].value_counts().sort_index())
print("\nTest:")
print(test_df['sensitivity_label'].value_counts().sort_index())

print("\n" + "="*80)
print("STEP 6: SAVE PROCESSED DATA")
print("="*80)

# Saving the splits to CSV files
train_df.to_csv(f'{BASE_PATH}/data/train.csv', index=False)
val_df.to_csv(f'{BASE_PATH}/data/val.csv', index=False)
test_df.to_csv(f'{BASE_PATH}/data/test.csv', index=False)
df_balanced.to_csv(f'{BASE_PATH}/data/balanced_subset.csv', index=False)
df.to_csv(f'{BASE_PATH}/data/full_dataset.csv', index=False)

print("Data saved to:")
print(f"  - {BASE_PATH}/data/train.csv")
print(f"  - {BASE_PATH}/data/val.csv")
print(f"  - {BASE_PATH}/data/test.csv")
print(f"  - {BASE_PATH}/data/balanced_subset.csv")
print(f"  - {BASE_PATH}/data/full_dataset.csv")

# Creating a template for results
results_template = test_df[['sentence', 'sentiment', 'sensitivity_level', 'sensitivity_label']].copy()
results_template.columns = ['text', 'true_sentiment', 'true_sensitivity_level', 'true_sensitivity_label']
results_template['predicted_sensitivity_level'] = -1
results_template['predicted_sensitivity_label'] = ''
results_template['confidence_score'] = 0.0
results_template['explanation'] = ''

results_template.to_csv(f'{BASE_PATH}/data/results_template.csv', index=False)
print(f"  - {BASE_PATH}/data/results_template.csv")

print("\n" + "="*80)
print("DATASET PREPARATION COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  - Original dataset: {len(df):,} samples")
print(f"  - Balanced dataset: {len(df_balanced):,} samples")
print(f"  - Training: {len(train_df):,} samples")
print(f"  - Validation: {len(val_df):,} samples")
print(f"  - Test: {len(test_df):,} samples")
print(f"\nNext step: Run fine-tuning scripts (01-04)")
