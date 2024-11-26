import os
import argparse
import shutil

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

PDF_PATH="pdf"
DB_PATH="ollama-local"
EMBEDDING_MODEL="hf.co/tensorblock/gte-Qwen2-1.5B-instruct-GGUF:Q5_K_M"

def load_documents():
    loader = PyPDFDirectoryLoader(PDF_PATH)
    return loader.load()

def add_to_vector_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    # load the model
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # create vector database
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    
    return db

def clean_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean the database")
    args = parser.parse_args()

    if args.clean:
        print("Cleaning the database...")
        clean_db()
    else:
        print("Loading documents...")
        docs = load_documents()
        print("Adding documents to the database...")
        add_to_vector_db(docs)
