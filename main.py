from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.core.application import Application
from src.resources.utils import load_config

from dotenv import load_dotenv

import sys

def main():

    load_dotenv()  # take environment variables
    baseConfig = load_config()
    
    print("[INFO] Configuration loaded successfully.")
    print(f"[INFO] Configuration: {baseConfig}")
    backend = Application(config=baseConfig)
    app = QApplication(sys.argv)
    window = MainWindow(backend=backend)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# TODO: 
# -memory threads and database refactor
# -streaming the rag graph
# -report graph
# -indexing graph
# -create a parser API
