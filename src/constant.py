OLLAMA_CLIENT = "http://127.0.0.1:6780/" # Change this value for distant server
OLLAMA_LOCALHOST = "http://127.0.0.1:11434/" # DO NOT CHANGE
UPLOAD_DIR = "./user_data/temp"
OUTPUT_DIR = "./user_data/outputs"
VECTORSTORE_DIR = "./user_data/vectorstore/"
LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL


# Compatible models
VISION_MODELS = ("qwen2.5vl",
                 "llava",
                 "minicpm-v",
                 "llama3.2-vision",
                 "moondream",
                 "mistral-small3.1",
                )
REASONING_MODELS = ("qwen3",
                    "deepseek-r1"
                   )