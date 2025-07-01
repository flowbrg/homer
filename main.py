from src.utils.logging import setup_logging
setup_logging("INFO")  # or "DEBUG" for more detailed logs

import subprocess

from src.env import *
from src.utils.utils import ensure_path

def run_streamlit_app():
    try:
        #subprocess.run(["streamlit", "run", "homer_persistent.py"], check=True)
        subprocess.run(["streamlit", "run", "homer_inmem.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error: {e}")

if __name__ == "__main__":

    ensure_path(path_str = UPLOAD_DIR)
    ensure_path(path_str = VECTORSTORE_DIR)
    #ensure_path(path_str = PERSISTENT_DIR)    #Only for homer_persistent
    #db.initialize_database()
    run_streamlit_app()