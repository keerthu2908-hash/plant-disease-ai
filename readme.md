# 🌱 Plant Disease AI Assistant

An AI-powered web application that helps identify plant diseases using semantic search.  
Users can search by disease name, crop, or symptoms — even with spelling mistakes.

🔗 **Live App:** https://plant-disease-analyzer-bot.streamlit.app/  
💻 **GitHub Repo:** https://github.com/keerthu2908-hash/plant-disease-ai

---

## 🚀 Features

- 🔍 Semantic search (understands meaning, not just keywords)
- ✨ Works even with typos and spelling mistakes
- 🌿 Covers multiple crops and plant diseases
- 📊 Displays:
  - Disease name
  - Crop
  - Symptoms
  - Diagnosis
  - Management
  - Causal organism
- ⚡ Fast and user-friendly interface

---

## 🧠 How It Works (Simple Explanation)

1. Plant disease data is stored in JSON format  
2. Each disease is converted into vector embeddings  
3. Stored in Pinecone (vector database)  
4. User query → converted to embedding  
5. Similar diseases are retrieved using semantic search  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Sentence Transformers  
- Pinecone  

---

## 📂 Project Structure

plant-disease-ai/  
├── app.py  
├── embedder.py  
├── retriever.py  
├── pinecone_db.py  
├── disease_data.json  
├── requirements.txt  
└── README.md  

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/keerthu2908-hash/plant-disease-ai.git
cd plant-disease-ai
pip install -r requirements.txt
streamlit run app.py
