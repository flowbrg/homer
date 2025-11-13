from utils.logging import get_logger

modelLogger = get_logger(__name__)

######################################## Embedding model ########################################

from langchain.embeddings.base import Embeddings

def load_embedding_model(
    model: str,
    host: str = None   #"http://localhost:11434"
  ) -> Embeddings:   
  from langchain_ollama import OllamaEmbeddings
  modelLogger.info(f"Loading embedding model: {model} with context window size 4096 and base URL {host if host else 'default'}")
  return OllamaEmbeddings(model = model,
              num_ctx=4096, # Context window size, default 2048 seems to trigger "decode: cannot decode batches with this context (use llama_encode() instead)" in Ollama
              base_url = host,)

######################################## Chat model ########################################

from langchain.chat_models.base import BaseChatModel

def load_chat_model(
    model: str,
    host: str = None #"http://localhost:11434"
  )-> BaseChatModel:
  from langchain_ollama import ChatOllama
  modelLogger.info(f"Loading chat model: {model}")
  return ChatOllama(model = model,
            temperature=0,
            num_ctx= 8192, #16384,
            base_url = host)