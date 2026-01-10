# Chat with PDF (Local RAG App)

A local, privacy-friendly **PDF Question Answering application** built using  
**Ollama**, **FAISS**, and **Streamlit**.

Upload a PDF and ask questions — answers are generated **only from the document**
with source citations.


## Features

- Fully local (no cloud APIs)
- Upload any PDF
- Ask natural language questions
- Retrieval-Augmented Generation (RAG)
- FAISS vector database with persistent embeddings
- Page-level citations
- Chat-style UI (Streamlit)


## Tech Stack

- **Python**
- **Ollama** (LLMs & embeddings)
- **FAISS** (vector search)
- **Streamlit** (UI)
- **PyPDF** (PDF parsing)
- **NumPy**


## Project Structure

pdf-qa-local/
├── app.py # Streamlit UI
├── backend.py # Core RAG logic
├── requirements.txt
├── .gitignore
└── screenshots/


## How to Run Locally

### 1. Clone the repository

-bash

git clone https://github.com/Sarth17/pdf-qa-local.git
cd pdf-qa-local


### 2. Create and activate virtual env

python -m venv venv
venv\Scripts\activate


### 3. install requirements

pip install -r requirements.txt


### 4. get the models 

Download and run Ollama from: https://ollama.com

ollama pull llama3.1:8b
ollama pull nomic-embed-text


## Run the app

streamlit run app.py
