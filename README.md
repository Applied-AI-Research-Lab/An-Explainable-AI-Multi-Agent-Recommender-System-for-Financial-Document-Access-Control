# An Explainable AI Multi-Agent Recommender System for Financial Document Access Control

## Article
* **Journal**: 
* **Title**: An Explainable AI Multi-Agent Recommender System for Financial Document Access Control
* **DOI**:

## Authors
* **Prof. Kanellos Toudas**
* **Prof. Dimitrios K. Nasiopoulos**
* **Dr. Konstantinos I. Roumeliotis**
* **Prof. George Georgakopoulos**
  
## Abstract
Financial institutions require robust document access control mechanisms that balance security with transparency and explainability. Traditional classification systems often operate as black boxes, failing to provide justifications for access-control decisions. This work presents a novel explainable AI multi-agent recommender system for financial document sensitivity classification that addresses critical ethical concerns in AI-powered decision-making. We fine-tuned three state-of-the-art models—FinBERT, BERT-base-uncased, and GPT-4.1-mini—on a custom-labeled Financial PhraseBank dataset with four sensitivity levels: Public, Internal, Confidential, and Restricted. These fine-tuned models serve as specialized AI agents within a multi-agent architecture orchestrated by GPT-5.1, a large reasoning model operating in zero-shot mode. The orchestrator synthesizes agent predictions and generates natural language recommendations that justify classification decisions. Our agentic AI multi-agent recommender system achieves 83.71\% overall accuracy, comparable to individual models (82.80\%-84.93\%), while providing interpretable explanations for each decision. Critically, agent agreement analysis reveals that unanimous decisions (3/3 agents agree, 78.8\% of cases) achieve 92.28\% accuracy—significantly outperforming any individual model—validating the collaborative decision-making approach. The system demonstrates that multi-agent architectures can provide both high-confidence predictions and natural language explainability, creating transparent, accountable AI systems for financial document access control. All code and methodologies are released as open-source on our GitHub to support reproducibility and further research in explainable AI for finance.

## Keywords
Document Access Control, Recommender Systems, Multi-Agent System, Explainable AI, Decision Support Systems, Business Intelligence
