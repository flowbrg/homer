from src.env import OLLAMA_CLIENT

######################################## Embedding model ########################################

from langchain.embeddings.base import Embeddings

def load_embedding_model(
        model: str,
        host: str = None   #"http://localhost:11434"
    ) -> Embeddings:   
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model = model, base_url = host)

######################################## Chat model ########################################

from langchain.chat_models.base import BaseChatModel

def load_chat_model(
        model: str,
        host: str = None   #"http://localhost:11434"
    )-> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(model = model, base_url= host)