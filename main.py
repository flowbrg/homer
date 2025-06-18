import subprocess

def run_streamlit_app():
    try:
        subprocess.run(["streamlit", "run", "webui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error: {e}")

if __name__ == "__main__":
    run_streamlit_app()