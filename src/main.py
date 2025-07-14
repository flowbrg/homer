import subprocess

from constant import *
from utils.utils import ensure_path

def run_streamlit_app():
    try:
        subprocess.run(["streamlit", "run", "src/streamlit_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error: {e}")

if __name__ == "__main__":

    ensure_path(path_str = UPLOAD_DIR)
    ensure_path(path_str = VECTORSTORE_DIR)
    run_streamlit_app()