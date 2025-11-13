import streamlit as st
import ollama

from tqdm import tqdm
from pages.utils import is_connected, is_ollama_client_available
from core.configuration import load_config
from constant import OLLAMA_CLIENT, OLLAMA_LOCALHOST


############################## Initialization ##############################


st.set_page_config(
  page_title="Model",
  layout="centered",
)

if "baseConfig" not in st.session_state:
  st.session_state.baseConfig = load_config()
if "ollama_host" not in st.session_state:
  st.session_state.ollama_host = OLLAMA_CLIENT
  

############################## Sidebar ##############################


# Server connection toggle
connectionButton = st.sidebar.toggle(
  label="Server execution",
  value=is_connected(st.session_state),
  key="modelsConnectionButton",
)

# Configure server host based on connection preference
if connectionButton:
  conn = is_ollama_client_available(st.session_state.ollama_host)
  if conn:
    st.session_state.baseConfig.ollama_host = st.session_state.ollama_host
  else:
    st.sidebar.warning(f"Could not connect to {st.session_state.ollama_host}")
    st.session_state.baseConfig.ollama_host = OLLAMA_LOCALHOST
else:
  st.session_state.baseConfig.ollama_host = OLLAMA_LOCALHOST

st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")
host = st.session_state.baseConfig.ollama_host
# Create Ollama client connecting to custom host/port
client = ollama.Client(
  host = host
)


############################## Page ##############################


st.info("**INFO**\n\n This page allows you to pull a model from Ollama, either locally or from a host. This depends on the server execution state \n\n \n\n Ollama must be installed where you want to run the model (locally or/and on the cluster) \n\n The model name must be written exactly as listed on the [Ollama model hub](https://ollama.com/library) ", icon="ℹ️")

model = st.text_input("Enter the model you want to pull:")
pullButton = st.button("Pull")
if pullButton and model:
  try:
    
    # Check if model already exists locally
    client.show(model)
    st.success(f"Model {model} is already available on host: {host}")
  except ollama.ResponseError as e:
    # If model doesn't exist (404 error), download it
    if e.status_code == 404:
      # Initialize tracking variables for progress bars
      current_digest, bars = "", {}
      
      try:
        with st.spinner(f"Pulling model {model}"):
          # Stream the model download process to get real-time updates
          for progress in client.pull(model, stream=True):
            # Get the digest (unique ID) for current file chunk being downloaded
            digest = progress.get("digest", "")
            
            # If we've moved to a new chunk, close the previous progress bar
            if digest != current_digest and current_digest in bars:
              bars[current_digest].close()

            # Create new progress bar for this chunk if it doesn't exist and has size info
            if digest not in bars and (total := progress.get("total")):
              bars[digest] = tqdm(
                total=total,              # Total bytes to download for this chunk
                desc=f"pulling {digest[7:19]}",     # Show first 12 chars of digest as description
                unit="B",               # Display units in bytes
                unit_scale=True,            # Auto-scale to KB/MB/GB
              )

            # Update progress bar with newly downloaded bytes
            if completed := progress.get("completed"):
              # Update only the difference between current completed and last position
              bars[digest].update(completed - bars[digest].n)

            # Track current digest for next iteration
            current_digest = digest
          st.success(f"Model {model} pulled successfully on {host}")
      except ollama.ResponseError as e:
        if e.status_code == 500:
          st.warning(f"Model {model} does not exist, visit https://ollama.com/ to find a compatible model.")


# Already available models
availableModels = f"Available models at {host}\n"
models = client.list().models
for m in models:
  availableModels += f"- {m.model}\n"
st.info(availableModels)