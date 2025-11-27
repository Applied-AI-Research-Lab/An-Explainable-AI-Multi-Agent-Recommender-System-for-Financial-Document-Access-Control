"""
I wrote this script to create training data for GPT fine-tuning.

Usage:
    python create_gpt_training_data.py --base_path /home/ubuntu
"""

import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='Create GPT training data')
parser.add_argument('--base_path', type=str, default='/home/ubuntu')
args = parser.parse_args()

BASE_PATH = args.base_path

# Set up the label mapping
label_map = {0: 'Public', 1: 'Internal', 2: 'Confidential', 3: 'Restricted'}

print("="*80)
print("CREATING GPT-4O-MINI FINE-TUNING DATA")
print("="*80)

# Load the data
train_df = pd.read_csv(f'{BASE_PATH}/data/train.csv')
val_df = pd.read_csv(f'{BASE_PATH}/data/val.csv')

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

def create_training_example(text, sensitivity_level):
    """Create a prompt-completion pair in OpenAI fine-tuning format"""
    system_message = "You are a financial document classifier that categorizes text into sensitivity levels: Public, Internal, Confidential, or Restricted."
    
    user_message = f"Classify the following financial text into one of these sensitivity levels: Public, Internal, Confidential, or Restricted.\n\nText: {text}\n\nSensitivity Level:"
    
    assistant_message = label_map[sensitivity_level]
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }

# Generate training data
print("\nCreating training examples...")
train_examples = []
for _, row in train_df.iterrows():
    example = create_training_example(row['sentence'], row['sensitivity_level'])
    train_examples.append(example)

# Generate validation data
print("Creating validation examples...")
val_examples = []
for _, row in val_df.iterrows():
    example = create_training_example(row['sentence'], row['sensitivity_level'])
    val_examples.append(example)

# Save as JSONL files
train_file = f'{BASE_PATH}/data/train_gpt.jsonl'
val_file = f'{BASE_PATH}/data/val_gpt.jsonl'

print(f"\nSaving to {train_file}...")
with open(train_file, 'w') as f:
    for example in train_examples:
        f.write(json.dumps(example) + '\n')

print(f"Saving to {val_file}...")
with open(val_file, 'w') as f:
    for example in val_examples:
        f.write(json.dumps(example) + '\n')

print("\n" + "="*80)
print("GPT TRAINING DATA CREATED!")
print("="*80)
print(f"Train examples: {len(train_examples)} -> {train_file}")
print(f"Validation examples: {len(val_examples)} -> {val_file}")
print("\nSensitivity distribution in training data:")
print(train_df['sensitivity_label'].value_counts().sort_index())
print("\nNext steps:")
print("1. Upload files to OpenAI: openai.files.create(file=open('train_gpt.jsonl', 'rb'), purpose='fine-tune')")
print("2. Create fine-tune job: openai.fine_tuning.jobs.create(training_file=file_id, model='gpt-4o-mini-2024-07-18')")
print("3. Monitor: openai.fine_tuning.jobs.retrieve(job_id)")

