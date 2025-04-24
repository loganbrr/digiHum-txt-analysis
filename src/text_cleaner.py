import re
import logging
from pathlib import Path
from typing import Optional
import unicodedata

# / logging #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self):
        self.cleaned_data_dir = Path("data/cleaned")
        self.cleaned_data_dir.mkdir(parents=True, exist_ok=True)

    # / clean text #
    def clean_text(self, text: str) -> str:
        try:
            #/ convert to lowercase #
            text = text.lower()
            
            # / remove special characters and extra whitespace #
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # / normalize unicode characters #
            text = unicodedata.normalize('NFKD', text)
            
            # / remove numbers #
            text = re.sub(r'\d+', '', text)
            
            # / remove extra whitespace #
            text = text.strip()
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text

    def process_file(self, input_path: Path) -> Optional[str]:
        # / use cleanfile method and save file into txt #
        try:
            # / read the input file #
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # / clean the text #
            cleaned_text = self.clean_text(text)
            
            # / generate output path #
            output_path = self.cleaned_data_dir / f"{input_path.stem}_cleaned.txt"
            
            # / save cleaned text #
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            logger.info(f"Successfully cleaned and saved {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {str(e)}")
            return None

    def process_directory(self, input_dir: Path) -> None:
        # / process all text files in the input directory #
        for file_path in input_dir.glob("*_processed.txt"):
            self.process_file(file_path)

if __name__ == "__main__":
    # / initialize the cleaner #
    cleaner = TextCleaner()
    
    # / process the input directory #
    input_dir = Path("data/raw")
    cleaner.process_directory(input_dir) 