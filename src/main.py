import subprocess

from constant import UPLOAD_DIR, VECTORSTORE_DIR
from utils.utils import ensure_path


# Ensure necessary directories exist
ensure_path(path_str = UPLOAD_DIR)
ensure_path(path_str = VECTORSTORE_DIR)

# Disable Chromadb telemetry
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Run the Streamlit app
try:
    subprocess.run(["streamlit", "run", "src/streamlit_app.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Streamlit exited with error: {e}")
