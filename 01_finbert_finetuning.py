"""
I created this script to fine-tune FinBERT for sensitivity classification.

Usage:
    python 01_finbert_finetuning.py --base_path /path/to/project
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import time
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setting up the arguments
parser = argparse.ArgumentParser(description='Fine-tune FinBERT model')
parser.add_argument('--base_path', type=str, default='/workspace/AgenticFinance',
                    help='Base path for the project')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
args = parser.parse_args()

BASE_PATH = args.base_path
MODEL_NAME = "ProsusAI/finbert"

print("="*80)
print("FINBERT FINE-TUNING")
print("="*80)
print(f"Base path: {BASE_PATH}")
print(f"Model: {MODEL_NAME}")

# Checking what device we're using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n" + "="*80)
print("STEP 1: LOAD DATA")
print("="*80)

train_df = pd.read_csv(f'{BASE_PATH}/data/train.csv')
val_df = pd.read_csv(f'{BASE_PATH}/data/val.csv')
test_df = pd.read_csv(f'{BASE_PATH}/data/test.csv')

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

print("\n" + "="*80)
print("STEP 2: PREPARE DATASETS")
print("="*80)

# Getting the tokenizer ready
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Loaded tokenizer: {MODEL_NAME}")
print(f"Vocab size: {tokenizer.vocab_size}")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

def prepare_dataset(df):
    dataset = Dataset.from_pandas(
        df[['sentence', 'sensitivity_level']].rename(
            columns={'sentence': 'text', 'sensitivity_level': 'label'}
        )
    )
    return dataset.map(tokenize_function, batched=True, remove_columns=['text'])

train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

print("Datasets tokenized")

print("\n" + "="*80)
print("STEP 3: INITIALIZE MODEL")
print("="*80)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model loaded")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("\n" + "="*80)
print("STEP 4: TRAINING CONFIGURATION")
print("="*80)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }

os.makedirs(f'{BASE_PATH}/FTmodels/finbert', exist_ok=True)

training_args = TrainingArguments(
    output_dir=f'{BASE_PATH}/FTmodels/finbert',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    warmup_steps=500,
    eval_strategy='steps',
    eval_steps=100,
    save_strategy='steps',
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    logging_dir=f'{BASE_PATH}/FTmodels/finbert/logs',
    logging_steps=50,
    save_total_limit=2,
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Trainer initialized")

print("\n" + "="*80)
print("STEP 5: TRAIN MODEL")
print("="*80)

print("Starting training...\n")
start_time = time.time()

train_result = trainer.train()

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/3600:.2f} hours")

print("\n" + "="*80)
print("STEP 6: SAVE MODEL")
print("="*80)

trainer.save_model(f'{BASE_PATH}/FTmodels/finbert/best_model')
tokenizer.save_pretrained(f'{BASE_PATH}/FTmodels/finbert/best_model')
print(f"Model saved to: {BASE_PATH}/FTmodels/finbert/best_model")

print("\n" + "="*80)
print("STEP 7: EVALUATE ON TEST SET")
print("="*80)

test_results = trainer.evaluate(test_dataset)
print("Test Results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

print("\n" + "="*80)
print("STEP 8: GENERATE PREDICTIONS")
print("="*80)

predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
confidence_scores = pred_probs.max(axis=1)

label_map = {0: 'Public', 1: 'Internal', 2: 'Confidential', 3: 'Restricted'}

def generate_explanation(text, predicted_level, confidence):
    level_explanations = {
        0: "This document contains publicly available financial information suitable for general distribution.",
        1: "This document contains company-specific analysis that should be restricted to internal staff.",
        2: "This document contains confidential strategic information requiring restricted access.",
        3: "This document contains highly sensitive information with potential regulatory or market impact."
    }
    base_explanation = level_explanations[predicted_level]
    confidence_text = f" Classification confidence: {confidence:.2%}."
    return base_explanation + confidence_text

explanations = [
    generate_explanation(text, pred, conf)
    for text, pred, conf in zip(test_df['sentence'].values, pred_labels, confidence_scores)
]

print(f"Generated predictions for {len(pred_labels)} samples")

print("\n" + "="*80)
print("STEP 9: SAVE PREDICTIONS")
print("="*80)

os.makedirs(f'{BASE_PATH}/results', exist_ok=True)

results_df = pd.DataFrame({
    'text': test_df['sentence'].values,
    'true_sentiment': test_df['sentiment'].values,
    'true_sensitivity_level': test_df['sensitivity_level'].values,
    'true_sensitivity_label': test_df['sensitivity_label'].values,
    'predicted_sensitivity_level': pred_labels,
    'predicted_sensitivity_label': [label_map[l] for l in pred_labels],
    'confidence_score': confidence_scores,
    'explanation': explanations,
    'model': 'FinBERT'
})

results_df.to_csv(f'{BASE_PATH}/results/finbert_predictions.csv', index=False)
print(f"Predictions saved to: {BASE_PATH}/results/finbert_predictions.csv")

print("\n" + "="*80)
print("STEP 10: GENERATE REPORTS")
print("="*80)

# Creating the confusion matrix plot
cm = confusion_matrix(test_df['sensitivity_level'].values, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Public', 'Internal', 'Confidential', 'Restricted'],
    yticklabels=['Public', 'Internal', 'Confidential', 'Restricted']
)
plt.title('FinBERT - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/finbert_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved")

# Generating the classification report
report = classification_report(
    test_df['sensitivity_level'].values, pred_labels,
    target_names=['Public', 'Internal', 'Confidential', 'Restricted'],
    digits=4
)
print("\nClassification Report:")
print(report)

with open(f'{BASE_PATH}/results/finbert_classification_report.txt', 'w') as f:
    f.write(report)
print("Classification report saved")

print("\n" + "="*80)
print("FINBERT FINE-TUNING COMPLETE!")
print("="*80)
print(f"Training time: {training_time/3600:.2f} hours")
print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1 (macro): {test_results['eval_f1_macro']:.4f}")
print(f"\nOutputs saved to: {BASE_PATH}/results/")
print("Next step: Run 02_bert_finetuning.py")
