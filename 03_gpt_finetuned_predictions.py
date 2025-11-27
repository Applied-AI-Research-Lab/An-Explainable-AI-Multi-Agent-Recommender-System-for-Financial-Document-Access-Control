"""
I created this script to generate predictions using fine-tuned GPT-4o-mini.

Usage:
    python 05_gpt_finetuned_predictions.py --base_path /home/ubuntu --api_key YOUR_KEY --model YOUR_MODEL

Run:
python 03_gpt_finetuned_predictions.py \
  --base_path /home/ubuntu \
  --api_key sk... \
  --model gpt-4.1-mini-2025-04-14

"""

import os
import argparse
import pandas as pd
import numpy as np
import time
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setting up the arguments
parser = argparse.ArgumentParser(description='Generate predictions using fine-tuned GPT model')
parser.add_argument('--base_path', type=str, default='/home/ubuntu')
parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--model', type=str, required=True, help='Fine-tuned model ID (e.g., gpt-4.1-mini-2025-04-14)')
args = parser.parse_args()

BASE_PATH = args.base_path
API_KEY = args.api_key
MODEL_ID = args.model

print("="*80)
print("GPT-4O-MINI FINE-TUNED MODEL PREDICTIONS")
print("="*80)
print(f"Base path: {BASE_PATH}")
print(f"Model: {MODEL_ID}")

# Setting up the OpenAI client
client = OpenAI(api_key=API_KEY)

print("\n" + "="*80)
print("STEP 1: LOAD TEST DATA")
print("="*80)

test_df = pd.read_csv(f'{BASE_PATH}/data/test.csv')
print(f"Test samples: {len(test_df)}")

print("\n" + "="*80)
print("STEP 2: GENERATE PREDICTIONS")
print("="*80)

label_map = {0: 'Public', 1: 'Internal', 2: 'Confidential', 3: 'Restricted'}
reverse_label_map = {v: k for k, v in label_map.items()}

def extract_sensitivity_level(response_text):
    """Extract sensitivity level from GPT response"""
    response_lower = response_text.lower().strip()
    if 'restricted' in response_lower:
        return 3
    elif 'confidential' in response_lower:
        return 2
    elif 'internal' in response_lower:
        return 1
    elif 'public' in response_lower:
        return 0
    # If the prediction is unclear, default to the most common level
    return 1

pred_labels = []
confidence_scores = []
explanations = []

# Setting up the CSV file for results
results_csv_path = f'{BASE_PATH}/results/gpt_finetuned_predictions.csv'
os.makedirs(f'{BASE_PATH}/results', exist_ok=True)

# Writing the header row
with open(results_csv_path, 'w') as f:
    f.write('text,true_sentiment,true_sensitivity_level,true_sensitivity_label,predicted_sensitivity_level,predicted_sensitivity_label,confidence_score,explanation,model\n')

print("Generating predictions...")
start_time = time.time()

for idx, row in test_df.iterrows():
    text = row['sentence']
    true_sentiment = row['sentiment']
    true_level = row['sensitivity_level']
    true_label = row['sensitivity_label']
    
    try:
        # Making the API call to the fine-tuned model
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a financial document classifier that categorizes text into sensitivity levels: Public, Internal, Confidential, or Restricted."},
                {"role": "user", "content": f"Classify the following financial text into one of these sensitivity levels: Public, Internal, Confidential, or Restricted.\n\nText: {text}\n\nSensitivity Level:"}
            ],
            temperature=0,
            max_tokens=10
        )
        
        # Pulling out the prediction from the response
        prediction_text = response.choices[0].message.content.strip()
        pred_level = extract_sensitivity_level(prediction_text)
        pred_label = label_map[pred_level]
        
        # Saving the results
        pred_labels.append(pred_level)
        confidence_scores.append(0.90)  # GPT models don't provide confidence scores
        explanations.append(f"GPT-4o-mini fine-tuned classification: {prediction_text}")
        
        # Appending to the CSV right away
        text_escaped = text.replace('"', '""')
        explanation_escaped = f"GPT-4o-mini fine-tuned classification: {prediction_text}".replace('"', '""')
        with open(results_csv_path, 'a') as f:
            f.write(f'"{text_escaped}",{true_sentiment},{true_level},{true_label},{pred_level},{pred_label},0.90,"{explanation_escaped}",GPT-4o-mini-FT\n')
        
        # Showing progress every 10 samples
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(test_df) - idx - 1)
            print(f"Processed {idx + 1}/{len(test_df)} samples... (Est. {remaining/60:.1f} min remaining)")
        
        # Adding a small delay to not hit API rate limits
        time.sleep(0.1)
        
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        # If something goes wrong, use a fallback
        pred_level = 1  # Default to Internal
        pred_label = label_map[pred_level]
        pred_labels.append(pred_level)
        confidence_scores.append(0.50)
        explanations.append(f"Error in prediction: {str(e)}")
        
        text_escaped = text.replace('"', '""')
        with open(results_csv_path, 'a') as f:
            f.write(f'"{text_escaped}",{true_sentiment},{true_level},{true_label},{pred_level},{pred_label},0.50,"Error in prediction",GPT-4o-mini-FT\n')

total_time = time.time() - start_time
pred_labels = np.array(pred_labels)
confidence_scores = np.array(confidence_scores)

print(f"\nGenerated {len(pred_labels)} predictions in {total_time/60:.2f} minutes")

print("\n" + "="*80)
print("STEP 3: CALCULATE METRICS")
print("="*80)

# Computing the evaluation metrics
accuracy = accuracy_score(test_df['sensitivity_level'].values, pred_labels)
f1_macro = f1_score(test_df['sensitivity_level'].values, pred_labels, average='macro')
f1_weighted = f1_score(test_df['sensitivity_level'].values, pred_labels, average='weighted')
precision, recall, f1, _ = precision_recall_fscore_support(
    test_df['sensitivity_level'].values, pred_labels, average='macro'
)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 (macro): {f1_macro:.4f}")
print(f"Test F1 (weighted): {f1_weighted:.4f}")
print(f"Test Precision (macro): {precision:.4f}")
print(f"Test Recall (macro): {recall:.4f}")

print("\n" + "="*80)
print("STEP 4: GENERATE REPORTS")
print("="*80)

# Creating the confusion matrix plot
cm = confusion_matrix(test_df['sensitivity_level'].values, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Public', 'Internal', 'Confidential', 'Restricted'],
    yticklabels=['Public', 'Internal', 'Confidential', 'Restricted']
)
plt.title('GPT-4o-mini Fine-tuned - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/gpt_finetuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved")

# Generating the classification report
report = classification_report(
    test_df['sensitivity_level'].values, pred_labels,
    target_names=['Public', 'Internal', 'Confidential', 'Restricted'],
    digits=4
)
print("\nClassification Report:")
print(report)

with open(f'{BASE_PATH}/results/gpt_finetuned_classification_report.txt', 'w') as f:
    f.write(report)
print("Classification report saved")

print("\n" + "="*80)
print("GPT-4O-MINI FINE-TUNED PREDICTIONS COMPLETE!")
print("="*80)
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test F1 (macro): {f1_macro:.4f}")
print(f"\nOutputs saved to: {BASE_PATH}/results/")
print(f"  - {results_csv_path}")
print(f"  - {BASE_PATH}/results/gpt_finetuned_confusion_matrix.png")
print(f"  - {BASE_PATH}/results/gpt_finetuned_classification_report.txt")
