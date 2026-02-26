# 🏛 Legal AI Copilot for Indian Law  
### Authority-Aware Multi-Hop RAG System with Hybrid Retrieval & Evaluation

---

## 🚀 Overview

Legal AI Copilot is an end-to-end Retrieval-Augmented Generation (RAG) system designed for Indian Law. It performs hybrid search, multi-hop reasoning, citation-grounded answer generation, and rigorous evaluation to reduce hallucination in legal responses. The system processes the Indian Constitution and multiple Indian Acts to provide structured, citation-backed legal answers.

---

## 🧠 System Architecture

User Query  
↓  
Hybrid Retrieval (BM25 + Dense FAISS Embeddings)  
↓  
Cross-Encoder Re-ranking  
↓  
Multi-Hop Retrieval Loop  
↓  
Citation-Grounded LLM Generation  
↓  
Hallucination Detection  
↓  
Evaluation (Recall@K, MRR, Faithfulness)

---

## 🔍 Key Features

- Hybrid Retrieval (BM25 + Dense Embeddings)
- Cross-Encoder Re-ranking (bge-reranker-large)
- Multi-Hop Reasoning Pipeline
- Citation-Aware Answer Generation
- Authority-Weighted Metadata Support
- Hallucination Detection
- Evaluation Framework (Recall@K, MRR, Faithfulness)
- Streamlit Evaluation Dashboard

---

## 📊 Evaluation Metrics

- Recall@K
- Mean Reciprocal Rank (MRR)
- Faithfulness (RAGAS)
- Answer Relevancy
- Latency Tracking
- Citation Validation

---

## 📂 Dataset Used

- Indian Constitution (PDF)
- Indian Contract Act 1872
- Arbitration & Conciliation Act 1996
- Arbitration Act 1940
- Indian Evidence Act 1872
- CrPC Dataset
- Copyright Amendment Act 2012

Documents are automatically parsed and structured section-wise.

---

## 🛠 Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence-Transformers (Cross-Encoder)
- OpenAI GPT
- RAGAS
- Streamlit

---

## ⚙️ Installation

Run:

pip install -r requirements.txt

---

## ▶️ Run Model

python legal_ai_full_pipeline.py

---

## 📊 Run Evaluation Dashboard

streamlit run evaluation_dashboard.py

---

## 🎯 Why This Project Matters

Legal AI systems require high retrieval precision, citation grounding, reduced hallucination, authority-aware reasoning, and multi-hop legal interpretation. This project demonstrates advanced RAG system design beyond basic chatbot implementations and focuses on reliability and evaluation.

---

## 📌 Future Improvements

- Authority-weighted ranking (Supreme Court > Act > Amendment)
- Amendment-aware temporal filtering
- Legal knowledge graph integration
- Self-verification chain
- Production API deployment

---

## 👨‍💻 Author

Final Year Engineering Student specializing in Generative AI and Retrieval Systems.
