import subprocess

from src.core import database as db

def run_streamlit_app():
    try:
        subprocess.run(["streamlit", "run", "webui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error: {e}")

if __name__ == "__main__":
    db.initialize_database()
    run_streamlit_app()