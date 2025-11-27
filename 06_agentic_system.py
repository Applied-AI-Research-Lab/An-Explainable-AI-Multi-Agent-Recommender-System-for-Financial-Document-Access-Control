"""
I built this script to run the agentic AI system with multiple agents and an orchestrator.

Usage:
    python 06_agentic_system.py --base_path /path/to/project --api_key YOUR_KEY --ft_model YOUR_FT_MODEL

python 06_agentic_system.py \
  --base_path /home/ubuntu \
  --api_key sk... \
  --ft_model ft:gpt-4.1-mini-2025-04-14:personal:fin:Cf0EKDYy
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Get the command line arguments
parser = argparse.ArgumentParser(description='Run agentic AI system')
parser.add_argument('--base_path', type=str, default='/home/ubuntu')
parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--ft_model', type=str, required=True, help='Fine-tuned GPT model ID')
parser.add_argument('--orchestrator_model', type=str, default='gpt-5.1-2025-11-13', help='Orchestrator model (reasoning model)')
args = parser.parse_args()

BASE_PATH = args.base_path
API_KEY = args.api_key
FT_MODEL = args.ft_model
ORCHESTRATOR_MODEL = args.orchestrator_model

print("="*80)
print("AGENTIC AI SYSTEM WITH GPT-5.1 ORCHESTRATOR")
print("="*80)
print(f"Base path: {BASE_PATH}")
print(f"Fine-tuned Agent Model: {FT_MODEL}")
print(f"Orchestrator Model: {ORCHESTRATOR_MODEL}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up the OpenAI client
client = OpenAI(api_key=API_KEY)

print("\n" + "="*80)
print("STEP 1: LOAD TEST DATA")
print("="*80)

test_df = pd.read_csv(f'{BASE_PATH}/data/test.csv')
print(f"Loaded {len(test_df)} test samples")

label_map = {0: 'Public', 1: 'Internal', 2: 'Confidential', 3: 'Restricted'}
reverse_label_map = {v: k for k, v in label_map.items()}

print("\n" + "="*80)
print("STEP 2: LOAD FINE-TUNED MODELS")
print("="*80)

# Load the FinBERT model
print("Loading FinBERT...")
finbert_tokenizer = AutoTokenizer.from_pretrained(f'{BASE_PATH}/FTmodels/finbert/best_model')
finbert_model = AutoModelForSequenceClassification.from_pretrained(f'{BASE_PATH}/FTmodels/finbert/best_model').to(device)
finbert_model.eval()

# Load the BERT model
print("Loading BERT...")
bert_tokenizer = AutoTokenizer.from_pretrained(f'{BASE_PATH}/FTmodels/bert-base/best_model')
bert_model = AutoModelForSequenceClassification.from_pretrained(f'{BASE_PATH}/FTmodels/bert-base/best_model').to(device)
bert_model.eval()

print("All local models loaded")
print(f"GPT Fine-tuned Agent: {FT_MODEL} (API-based)")
print(f"Orchestrator: {ORCHESTRATOR_MODEL} (API-based)")

print("\n" + "="*80)
print("STEP 3: DEFINE AGENT CLASSES")
print("="*80)

class ClassifierAgent:
    """Agent for BERT-based classifiers (FinBERT, BERT)"""
    def __init__(self, name, model, tokenizer, device):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        reasoning = f"{self.name} classified as '{label_map[pred_label]}' with {confidence:.2%} confidence based on financial domain patterns."
        return pred_label, confidence, reasoning

class GPTAgent:
    """Agent for GPT fine-tuned model"""
    def __init__(self, name, model_id, client):
        self.name = name
        self.model_id = model_id
        self.client = client
    
    def predict(self, text):
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a financial document classifier that categorizes text into sensitivity levels: Public, Internal, Confidential, or Restricted."},
                    {"role": "user", "content": f"Classify the following financial text into one of these sensitivity levels: Public, Internal, Confidential, or Restricted.\n\nText: {text}\n\nSensitivity Level:"}
                ],
                temperature=0,
                max_tokens=10
            )
            
            prediction_text = response.choices[0].message.content.strip()
            
            # Map the response to a label
            response_lower = prediction_text.lower()
            if 'restricted' in response_lower:
                pred_label = 3
            elif 'confidential' in response_lower:
                pred_label = 2
            elif 'internal' in response_lower:
                pred_label = 1
            else:
                pred_label = 0
            
            confidence = 0.90  # GPT doesn't give confidence scores
            reasoning = f"{self.name} classified as '{label_map[pred_label]}' based on fine-tuned understanding of financial sensitivity patterns."
            return pred_label, confidence, reasoning
            
        except Exception as e:
            print(f"  Error in {self.name}: {e}")
            return 1, 0.5, f"{self.name} error: {str(e)}"

# Set up the agents
finbert_agent = ClassifierAgent("FinBERT-FT", finbert_model, finbert_tokenizer, device)
bert_agent = ClassifierAgent("BERT-FT", bert_model, bert_tokenizer, device)
gpt_agent = GPTAgent("GPT-4.1-mini-FT", FT_MODEL, client)

agents = [finbert_agent, bert_agent, gpt_agent]

print("Agents initialized")

print("\n" + "="*80)
print("STEP 4: DEFINE ORCHESTRATOR")
print("="*80)

class Orchestrator:
    """GPT-5.1 based orchestrator with reasoning for final decision"""
    def __init__(self, model_id, client):
        self.model_id = model_id
        self.client = client
    
    def make_decision(self, text, agent_proposals):
        # Prepare the agent proposals for the prompt
        proposals_text = "\n".join([
            f"- {name}: {label_map[pred]} (confidence: {conf:.2%})\n  Reasoning: {reasoning}"
            for name, pred, conf, reasoning in agent_proposals
        ])
        
        prompt = f"""You are an expert orchestrator that makes final classification decisions based on multiple specialized AI agent predictions.

IMPORTANT: The three agents below are fine-tuned models that were specifically trained on this exact financial sensitivity classification task with 83-85% accuracy. They are domain experts that have learned patterns from thousands of labeled examples. Your role is to synthesize their expert predictions, NOT to override them with general reasoning.

Financial Text:
"{text}"

Agent Predictions (from fine-tuned specialist models):
{proposals_text}

DECISION RULES:
1. If ALL THREE agents agree → You MUST follow their consensus (they are specialized experts with 91% accuracy when unanimous)
2. If TWO agents agree → Strongly favor the majority prediction unless you have compelling domain-specific financial sensitivity evidence
3. Only override if you can identify a clear error in their domain-specific reasoning based on financial sensitivity criteria

Classification Criteria:
- Public: Publicly available information suitable for general distribution (press releases, public statements, announced data)
- Internal: Company-specific information for internal staff only (performance data, operational details, revenue figures)
- Confidential: Strategic confidential information requiring restricted access (undisclosed strategies, competitive intelligence, forecasts)
- Restricted: Highly sensitive information with potential regulatory or market impact (M&A deals, insider information, material non-public information)

Provide:
1. Your final classification (Public/Internal/Confidential/Restricted)
2. A clear recommendation explaining your decision and what actions should be taken

Format your response as:
Classification: [Your classification]
Recommendation: [Your detailed recommendation]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_completion_tokens=500
            )
            
            output_text = response.choices[0].message.content.strip()
            
            # Parse the classification from the response
            lines = output_text.split('\n')
            classification_line = [l for l in lines if l.startswith('Classification:')]
            recommendation_lines = [l for l in lines if l.startswith('Recommendation:')]
            
            if classification_line:
                classification_text = classification_line[0].replace('Classification:', '').strip().lower()
            else:
                classification_text = output_text.lower()
            
            # Determine the final prediction
            if 'restricted' in classification_text:
                final_pred = 3
            elif 'confidential' in classification_text:
                final_pred = 2
            elif 'internal' in classification_text:
                final_pred = 1
            else:
                final_pred = 0
            
            # Get the recommendation
            if recommendation_lines:
                recommendation = recommendation_lines[0].replace('Recommendation:', '').strip()
                # If multi-line, get the rest
                rec_start_idx = lines.index(recommendation_lines[0])
                recommendation = '\n'.join(lines[rec_start_idx:]).replace('Recommendation:', '').strip()
            else:
                recommendation = output_text
            
            return final_pred, label_map[final_pred], recommendation, output_text
            
        except Exception as e:
            print(f"  Error in Orchestrator: {e}")
            # Use majority vote as fallback
            agent_preds = [pred for _, pred, _, _ in agent_proposals]
            final_pred = max(set(agent_preds), key=agent_preds.count)
            recommendation = f"Fallback to majority vote due to error: {str(e)}"
            return final_pred, label_map[final_pred], recommendation, ""

orchestrator = Orchestrator(ORCHESTRATOR_MODEL, client)
print("Orchestrator initialized")

print("\n" + "="*80)
print("STEP 5: RUN AGENTIC SYSTEM")
print("="*80)

results = []
start_time = time.time()

# Set up the results CSV
results_csv_path = f'{BASE_PATH}/results/agentic_system_results.csv'
os.makedirs(f'{BASE_PATH}/results', exist_ok=True)

with open(results_csv_path, 'w') as f:
    f.write('text,true_label,true_label_name,finbert_pred,finbert_conf,finbert_reasoning,bert_pred,bert_conf,bert_reasoning,gpt_pred,gpt_conf,gpt_reasoning,orchestrator_pred,orchestrator_pred_name,orchestrator_recommendation,orchestrator_full_response,correct\n')

for idx, row in test_df.iterrows():
    text = row['sentence']
    true_label = row['sensitivity_level']
    
    if (idx + 1) % 10 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (len(test_df) - idx - 1)
        print(f"Processing sample {idx+1}/{len(test_df)}... (Est. {remaining/60:.1f} min remaining)")
    
    # Get predictions from each agent
    agent_proposals = []
    for agent in agents:
        pred, conf, reasoning = agent.predict(text)
        agent_proposals.append((agent.name, pred, conf, reasoning))
    
    # Let the orchestrator decide
    final_pred, final_label, recommendation, full_response = orchestrator.make_decision(text, agent_proposals)
    
    is_correct = final_pred == true_label
    
    # Write to CSV right away
    text_escaped = text.replace('"', '""')
    finbert_reasoning_escaped = agent_proposals[0][3].replace('"', '""')
    bert_reasoning_escaped = agent_proposals[1][3].replace('"', '""')
    gpt_reasoning_escaped = agent_proposals[2][3].replace('"', '""')
    recommendation_escaped = recommendation.replace('"', '""')
    full_response_escaped = full_response.replace('"', '""')
    
    with open(results_csv_path, 'a') as f:
        f.write(f'"{text_escaped}",{true_label},{label_map[true_label]},{agent_proposals[0][1]},{agent_proposals[0][2]:.4f},"{finbert_reasoning_escaped}",{agent_proposals[1][1]},{agent_proposals[1][2]:.4f},"{bert_reasoning_escaped}",{agent_proposals[2][1]},{agent_proposals[2][2]:.4f},"{gpt_reasoning_escaped}",{final_pred},{final_label},"{recommendation_escaped}","{full_response_escaped}",{is_correct}\n')
    
    results.append({
        'text': text,
        'true_label': true_label,
        'true_label_name': label_map[true_label],
        'finbert_pred': agent_proposals[0][1],
        'bert_pred': agent_proposals[1][1],
        'gpt_pred': agent_proposals[2][1],
        'orchestrator_pred': final_pred,
        'orchestrator_pred_name': final_label,
        'correct': is_correct
    })
    
    # Brief pause to avoid hitting rate limits
    time.sleep(0.2)

total_time = time.time() - start_time

print("\n" + "="*80)
print("STEP 6: CALCULATE METRICS & GENERATE REPORTS")
print("="*80)

results_df = pd.DataFrame(results)

# Compute the metrics
accuracy = accuracy_score(results_df['true_label'], results_df['orchestrator_pred'])
f1_macro = f1_score(results_df['true_label'], results_df['orchestrator_pred'], average='macro')
f1_weighted = f1_score(results_df['true_label'], results_df['orchestrator_pred'], average='weighted')

print(f"\nAgentic System Performance:")
print(f"  Samples processed: {len(results_df)}")
print(f"  Total time: {total_time/60:.2f} minutes")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 (macro): {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")

# Check how often agents agree
finbert_bert_agree = (results_df['finbert_pred'] == results_df['bert_pred']).mean()
finbert_gpt_agree = (results_df['finbert_pred'] == results_df['gpt_pred']).mean()
bert_gpt_agree = (results_df['bert_pred'] == results_df['gpt_pred']).mean()
full_consensus = ((results_df['finbert_pred'] == results_df['bert_pred']) & 
                  (results_df['bert_pred'] == results_df['gpt_pred'])).mean()

print(f"\nAgent Agreement Rates:")
print(f"  FinBERT-FT vs BERT-FT: {finbert_bert_agree:.2%}")
print(f"  FinBERT-FT vs GPT-4.1-mini-FT: {finbert_gpt_agree:.2%}")
print(f"  BERT-FT vs GPT-4.1-mini-FT: {bert_gpt_agree:.2%}")
print(f"  Full Consensus: {full_consensus:.2%}")

# Create confusion matrix plot
cm = confusion_matrix(results_df['true_label'], results_df['orchestrator_pred'])
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=list(label_map.values()),
    yticklabels=list(label_map.values())
)
plt.title('Agentic System - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/results/agentic_system_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved")

# Generate classification report
report = classification_report(
    results_df['true_label'], results_df['orchestrator_pred'],
    target_names=list(label_map.values()),
    digits=4
)
print("\nClassification Report:")
print(report)

with open(f'{BASE_PATH}/results/agentic_system_classification_report.txt', 'w') as f:
    f.write(report)
print(f"Classification report saved")

print("\n" + "="*80)
print("AGENTIC SYSTEM COMPLETE!")
print("="*80)
print(f"\nResults saved to:")
print(f"  - {results_csv_path}")
print(f"  - {BASE_PATH}/results/agentic_system_confusion_matrix.png")
print(f"  - {BASE_PATH}/results/agentic_system_classification_report.txt")
print("\nThe multi-agent system successfully demonstrates:")
print("  Three specialized agents (FinBERT-FT, BERT-FT, GPT-4.1-mini-FT)")
print(f"  {ORCHESTRATOR_MODEL} orchestrator with advanced reasoning capabilities")
print("  Collaborative decision-making with detailed recommendations")
print(f"  Final accuracy: {accuracy:.2%}")
