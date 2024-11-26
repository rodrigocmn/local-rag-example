import os

from typing import List, Dict, Any

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from embeddings.ollama_local import get_embeddings_function

PDF_PATH="pdf"
DB_PATH="ollama-local"
EMBEDDING_MODEL="hf.co/tensorblock/gte-Qwen2-1.5B-instruct-GGUF:Q5_K_M"
CHAT_MODEL="incept5/llama3.1-claude"

def run_llm(query: str, chat_history: List[Dict[str,Any]]):
    # load the chat model
    chat = ChatOllama(model=CHAT_MODEL)
    
    # Get the embeddings model
    embeddings = get_embeddings_function(EMBEDDING_MODEL)
    
    # load the vector database
    db = FAISS.load_local(DB_PATH, embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
    
    # Question and Answer prompt based solely on the context
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Rephrase the prompt with chat history
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    
    # history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, 
        retriever=db.as_retriever(), 
        prompt=rephrase_prompt
    )
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        combine_docs_chain=create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    )
    
    # invoke the chain
    return retrieval_chain.invoke({"input": query, "chat_history": chat_history})