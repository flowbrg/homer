from src.utils.logging import setup_logging
setup_logging("INFO")  # or "DEBUG" for more detailed logs

import subprocess

from src.core import database as db
from src.env import *


def run_streamlit_app():
    try:
        #subprocess.run(["streamlit", "run", "homer_persistent.py"], check=True)
        subprocess.run(["streamlit", "run", "homer_inmem.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error: {e}")

def ensure_path(path_str: str):
    """Crée le répertoire parent si le chemin est un fichier, ou le répertoire lui-même."""
    from pathlib import Path
    path = Path(path_str)
    
    # Si le chemin se termine par '/' ou n'a pas d'extension, c'est un dossier
    if path_str.endswith('/') or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
    else:
        # C'est un fichier, créer le répertoire parent
        path.parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    ensure_path(path_str = UPLOAD_DIR)
    ensure_path(path_str = VECTORSTORE_DIR)
    ensure_path(path_str = PERSISTENT_DIR)    #Only for homer_persistent
    db.initialize_database()
    run_streamlit_app()