# 🌿 AI Crop Disease Diagnostic System

An end-to-end AI-powered agricultural decision support system that combines **computer vision, RAG (Retrieval-Augmented Generation), and LLM reasoning** to diagnose plant diseases and provide actionable insights.

---

## 🚀 Live Demo
🔗 https://plant-disease-analyzer-bot.streamlit.app/

---

## 🧠 Key Features

### 🔍 AI-Based Disease Diagnosis
- Image-based classification using **MobileNetV2 (Hugging Face)**
- Supports multi-crop disease detection
- Confidence scoring (High / Moderate / Low)

### 🔥 Explainable AI (Grad-CAM)
- Visual heatmaps highlighting affected regions
- Improves model transparency and trust

### 📚 RAG-Based Symptom Analysis
- Semantic search using **vector embeddings (sentence-transformers)**
- Matches user symptoms with disease database (1000+ records)

### 🤖 LLM Reasoning Layer
- Integrated **OpenAI (GPT-4o-mini)** via LangChain
- Generates short, human-readable explanations

### 🔄 Agentic Workflow (LangGraph)
- Multi-stage AI pipeline:
  - Image prediction
  - Symptom retrieval
  - LLM explanation
- Orchestrated using **LangGraph**

### 🌦 Weather-Based Risk Analysis *(Optional / Extendable)*
- Supports integration with weather APIs
- Predicts disease spread likelihood

---

## 🛠 Tech Stack

**Languages**
- Python

**AI / ML**
- Hugging Face Transformers
- PyTorch
- Scikit-learn

**GenAI & RAG**
- OpenAI API
- LangChain
- LangGraph
- Sentence Transformers

**Explainability**
- Grad-CAM
- OpenCV

**Frontend & Deployment**
- Streamlit
- GitHub
- Streamlit Cloud

---

## 🏗 Architecture

## ▶️ How to Run Locally

```bash
git clone https://github.com/keerthu2908-hash/plant-disease-ai.git
cd plant-disease-ai
pip install -r requirements.txt
streamlit run app.py
