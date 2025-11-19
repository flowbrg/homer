# Homer

Homer is a small academic project of a RAG application using Ollama and Langgraph as well as streamlit for the UI.

This project is designed to be executed on relatively low-performance laptops; therefore, computationally intensive tasks are delegated to a server (specifically, an Ollama instance operating on a non-default IP address).
## Environment

In PowerShell:
```
git pull https://github.com/flowbrg/homer.git
cd ./homer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ./requirements.txt
```

Bash:
```
git pull https://github.com/flowbrg/homer.git
cd ./homer
python -m venv .venv
source .\.venv\bin\activate
pip install -r ./requirements.txt
```

You can use [uv](https://github.com/astral-sh/uv)
```
u
## Dependencies

- [Ollama](https://ollama.com) 
cd ./homer
uv venv
source .\.venv\bin\activate # or .\.venv\Scripts\Activate.ps1
uv sync
uv run .\src\main.py
```

## TODO

- Refactor the configuration (persistent configuration using json)
- Improve the RAG-as-a-tool agent
- Explore microsoft [markitdown](https://github.com/microsoft/markitdown) and google's [langextract](https://github.com/google/langextract)
- Migrate from streamlit probably