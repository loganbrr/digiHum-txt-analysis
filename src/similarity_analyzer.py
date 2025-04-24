import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    def __init__(self):
        self.vectors_dir = Path("data/vectors")
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_vectors(self) -> List[Tuple[str, np.ndarray]]:
        """Load all vector files and their metadata."""
        vectors = []
        for vector_path in sorted(self.vectors_dir.glob("*_vector.npy")):
            try:
                # Load vector
                vector = np.load(vector_path)
                
                # Load metadata
                with open(vector_path.with_suffix('.json'), 'r') as f:
                    metadata = json.load(f)
                
                vectors.append((str(vector_path), vector))
            except Exception as e:
                logger.error(f"Error loading vector {vector_path}: {str(e)}")
        
        return vectors

    def compute_similarities(self, vectors: List[Tuple[str, np.ndarray]]) -> pd.DataFrame:
        """Compute cosine similarity between consecutive vectors."""
        similarities = []
        dates = []
        
        for i in range(len(vectors) - 1):
            try:
                current_path, current_vector = vectors[i]
                next_path, next_vector = vectors[i + 1]
                
                # Compute cosine similarity
                similarity = cosine_similarity(current_vector, next_vector)[0][0]
                
                # Extract dates from filenames
                current_date = Path(current_path).stem.split('_')[2]
                next_date = Path(next_path).stem.split('_')[2]
                
                similarities.append(similarity)
                dates.append((current_date, next_date))
            except Exception as e:
                logger.error(f"Error computing similarity: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'current_date': [d[0] for d in dates],
            'next_date': [d[1] for d in dates],
            'similarity': similarities
        })
        
        return df

    def visualize_similarities(self, df: pd.DataFrame) -> None:
        """Create visualizations of the similarity scores."""
        try:
            # Create line plot
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='current_date', y='similarity')
            plt.title('FOMC Minutes Similarity Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cosine Similarity')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'similarity_trend.png')
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            similarity_matrix = np.zeros((len(df), len(df)))
            for i in range(len(df)):
                similarity_matrix[i, i] = 1.0
                if i < len(df) - 1:
                    similarity_matrix[i, i+1] = df.iloc[i]['similarity']
                    similarity_matrix[i+1, i] = df.iloc[i]['similarity']
            
            sns.heatmap(similarity_matrix, 
                       xticklabels=df['current_date'],
                       yticklabels=df['current_date'],
                       cmap='YlOrRd')
            plt.title('FOMC Minutes Similarity Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'similarity_matrix.png')
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    def save_results(self, df: pd.DataFrame) -> None:
        """Save similarity results to CSV."""
        try:
            output_path = self.results_dir / 'similarity_scores.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def run(self):
        """Main execution method."""
        vectors = self.load_vectors()
        if vectors:
            df = self.compute_similarities(vectors)
            self.visualize_similarities(df)
            self.save_results(df)
        else:
            logger.error("No vectors found to analyze")

if __name__ == "__main__":
    analyzer = SimilarityAnalyzer()
    analyzer.run() 