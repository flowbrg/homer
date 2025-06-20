import os
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import save_output

def _single_pdf_parser(path: str, file_name: str, method: str = "markdown") -> str:
    assert method in ["json", "html", "markdown"], "Invalid method. Choose from 'json', 'html', or 'markdown'."
    config = {
        "output_format": method,
        "ADDITIONAL_KEY": "VALUE"
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    full_path = os.path.join(path, file_name)
    renderer = converter(full_path)
    return renderer

def pdf_parser(path: str, temp_path: str = None):

    #list every pdf file in the directory
    files = os.listdir(path)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    print(f'processing {len(pdf_files)} pdf files')
    for file in pdf_files:
        try:
            renderer = _single_pdf_parser(path=path, file_name=file, method="markdown")
            file_stem = Path(file).stem
            output_dir = os.path.join(temp_path, file_stem)
            os.makedirs(output_dir, exist_ok=True)
            save_output(renderer, output_dir, file_stem)
        except Exception as e:
            print(f"Error while processing {file} : {e}")
    print("pdf files processing complete.")