from langchain_ollama import OllamaEmbeddings

def get_embeddings_function(embedding_model: str):
    embeddings = OllamaEmbeddings(model=embedding_model)
    return embeddings