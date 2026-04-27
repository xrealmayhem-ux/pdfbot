---
title: pdf bot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.13.0
app_file: app.py
pinned: false
---

# pdf bot

A Retrieval-Augmented Generation (RAG) Question Answering Bot built with LangChain, ChromaDB, and Gradio. This project uses 100% free and open-source models via Hugging Face.

## Features
- **PDF Processing:** Upload PDF files to extract and analyze text.
- **On-Server Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2`, which runs directly on the hosting server's CPU. This means no external API calls or extra costs for vector generation.
- **Vector Database:** Uses ChromaDB to store and retrieve document chunks efficiently.
- **LLM:** Powered by `Mistral-7B-Instruct-v0.3` via the free Hugging Face Inference API.
- **Web Interface:** Built with Gradio for a clean, user-friendly experience.

## Prerequisites
- Python 3.10 or 3.11
- A [Hugging Face account](https://huggingface.co/) and an Access Token.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd pdfbot
   ```

2. **Create a virtual environment (Conda is recommended):**
   ```bash
   conda create -n qabot_env python=3.10 -y
   conda activate qabot_env
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the Environment Variables:**
   Open the `.env` file and replace the placeholder with your actual Hugging Face token:
   ```env
   HF_TOKEN="your_huggingface_token_here"
   ```

## Usage
Run the application using Python:
```bash
python app.py
```
This will start a local Gradio server. Upload a PDF, wait a few seconds for it to process, and start asking questions!
