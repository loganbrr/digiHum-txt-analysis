import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextVectorizer:
    def __init__(self, method: str = "tfidf"):
        self.method = method
        self.vectors_dir = Path("data/vectors")
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        
        if method == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.model.eval()
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

    def vectorize_text(self, text: str) -> np.ndarray:
        """Convert text to vector representation."""
        try:
            if self.method == "bert":
                # Tokenize and get BERT embeddings
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Use [CLS] token embedding as document representation
                return outputs.last_hidden_state[:, 0, :].numpy()
            else:
                # TF-IDF vectorization
                return self.vectorizer.fit_transform([text]).toarray()
        except Exception as e:
            logger.error(f"Error vectorizing text: {str(e)}")
            return np.array([])

    def process_file(self, input_path: Path) -> Optional[str]:
        """Process a single file and save its vector representation."""
        try:
            # Read the input file
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Vectorize the text
            vector = self.vectorize_text(text)
            
            # Generate output path
            output_path = self.vectors_dir / f"{input_path.stem}_vector.npy"
            
            # Save vector
            np.save(output_path, vector)
            
            # Save metadata
            metadata = {
                "method": self.method,
                "shape": vector.shape,
                "file": str(input_path)
            }
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Successfully vectorized and saved {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {str(e)}")
            return None

    def process_directory(self, input_dir: Path) -> None:
        """Process all cleaned text files in the input directory."""
        for file_path in input_dir.glob("*_cleaned.txt"):
            self.process_file(file_path)

if __name__ == "__main__":
    vectorizer = TextVectorizer(method="tfidf")  # or "bert"
    input_dir = Path("data/cleaned")
    vectorizer.process_directory(input_dir) 