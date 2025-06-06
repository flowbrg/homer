from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from PIL import Image
from transformers import AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm

class ImageLoader(BaseLoader):
    """A loader that turn images into a caption"""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def load(self) -> Document:
        with Image.open(self.file_path).convert('RGB') as i:
            m=AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2",
                                                   revision="2025-04-14",
                                                   trust_remote_code=True,
                                                   device_map={"": "cuda"})
            caption = m.caption(i, length="short")["caption"]
            return Document(
                page_content=caption,
                metadata={"type": "","page": "","source": ""},
            )
    

def parse_image_metadata(image_path: Path) -> dict:
    """
    Extrait les métadonnées à partir du nom du fichier image, supposé être du type :
    _page_100_Figure_3.jpeg ou _page_106_Picture_1.jpeg
    """
    name = image_path.stem  # Sans extension
    parts = name.split('_')

    try:
        page_index = parts.index('page') + 1
        page_number = int(parts[page_index])
    except (ValueError, IndexError):
        page_number = None

    figure_type = None
    figure_number = None
    for token in parts:
        if token.startswith("Figure") or token.startswith("Picture"):
            figure_type = token.split('_')[0]
            figure_number = token.split('_')[1] if '_' in token else parts[-1]
            break
        elif "Figure" in token or "Picture" in token:
            for fig_type in ["Figure", "Picture"]:
                if fig_type in token:
                    figure_type = fig_type
                    figure_number = token.split(fig_type)[-1]
                    break

    return {
        "type": f"{figure_type} {figure_number}" if figure_type and figure_number else None,
        "source": image_path.parents[0].name  # dossier = nom du fichier PDF sans extension
    }

    #"page": page_number,

def iterate_image_caption(path: str) -> list[Document]:
    documents = []
    path_obj = Path(path)
    image_files = list(path_obj.rglob("*.jpeg"))

    for image_path in tqdm(image_files, "loading images"):
        try:
            loader = ImageLoader(str(image_path))
            document = loader.load()
            document.metadata = parse_image_metadata(image_path)
            documents.append(document)
        except Exception as e:
            print(f"Error while processing {image_path} : {e}")
    print("Images loading complete")
    return documents
