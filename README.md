# Simple RAG implementation example

This is an example of RAG implementation using local LLMs with Ollama and FAISS vector database.

# Requirements

- Create a `.env` file according to the `.env.example` file
- Ollama (https://ollama.com/) installed in your local machine serving the models configured in the `.env` file

# Installation

1. Create a python virtual environment and activate it.
```
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements
```
pip install -r requirements.txt
```

# Data ingestion

Use the following command to ingest the PDFs into the vector database.

```
python3 ingest.py
```

# Start the Bot UI

This is the command to start the user interface:

```
streamlit run main.py
```