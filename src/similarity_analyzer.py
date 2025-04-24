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
    # / initialize the analyzer #
    def __init__(self):
        self.vectors_dir = Path("data/vectors")
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_vectors(self) -> List[Tuple[str, np.ndarray]]:
        vectors = []
        for vector_path in sorted(self.vectors_dir.glob("*_vector.npy")):
            try:
                vector = np.load(vector_path)
                if vector.ndim == 1:
                    vector = vector.reshape(1, -1)  # Reshape to 2D array
                
                with open(vector_path.with_suffix('.json'), 'r') as f:
                    metadata = json.load(f)
                
                vectors.append((str(vector_path), vector))
                logger.info(f"Successfully loaded vector from {vector_path}")
            except Exception as e:
                logger.error(f"Error loading vector {vector_path}: {str(e)}")
                continue
        
        if not vectors:
            logger.warning("No vectors were successfully loaded")
        return vectors

    def compute_similarities(self, vectors: List[Tuple[str, np.ndarray]]) -> pd.DataFrame:
        similarities = []
        dates = []
        
        # Find minimum dimension across all vectors
        min_dim = min(vector[1].shape[1] for vector in vectors)
        logger.info(f"Using minimum dimension of {min_dim} for all vectors")
        
        for i in range(len(vectors) - 1):
            try:
                current_path, current_vector = vectors[i]
                next_path, next_vector = vectors[i + 1]
                
                # Ensure vectors are 2D arrays
                if current_vector.ndim == 1:
                    current_vector = current_vector.reshape(1, -1)
                if next_vector.ndim == 1:
                    next_vector = next_vector.reshape(1, -1)
                
                # Truncate vectors to minimum dimension
                current_vector = current_vector[:, :min_dim]
                next_vector = next_vector[:, :min_dim]
                
                # Compute cosine similarity
                similarity = cosine_similarity(current_vector, next_vector)[0][0]
                
                # Extract dates from filenames
                current_date = Path(current_path).stem.split('_')[2]
                next_date = Path(next_path).stem.split('_')[2]
                
                similarities.append(similarity)
                dates.append((current_date, next_date))
                logger.debug(f"Computed similarity between {current_date} and {next_date}: {similarity}")
            except Exception as e:
                logger.error(f"Error computing similarity between {current_path} and {next_path}: {str(e)}")
                continue
        
        if not similarities:
            logger.error("No similarities were computed")
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'current_date': [d[0] for d in dates],
            'next_date': [d[1] for d in dates],
            'similarity': similarities
        })
        
        return df

    def visualize_similarities(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.error("Cannot create visualizations: DataFrame is empty")
            return
            
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
            plt.close()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            n = len(df)
            similarity_matrix = np.zeros((n, n))
            
            # Fill diagonal with 1.0
            np.fill_diagonal(similarity_matrix, 1.0)
            
            # Fill off-diagonal elements with similarities
            for i in range(n-1):
                similarity_matrix[i, i+1] = df.iloc[i]['similarity']
                similarity_matrix[i+1, i] = df.iloc[i]['similarity']
            
            sns.heatmap(similarity_matrix, 
                       xticklabels=df['current_date'],
                       yticklabels=df['current_date'],
                       cmap='YlOrRd',
                       vmin=0,
                       vmax=1)
            plt.title('FOMC Minutes Similarity Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'similarity_matrix.png')
            plt.close()
            
            logger.info("Successfully created visualizations")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    def save_results(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.error("Cannot save results: DataFrame is empty")
            return
            
        try:
            output_path = self.results_dir / 'similarity_scores.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def run(self):
        vectors = self.load_vectors()
        if not vectors:
            logger.error("No vectors found to analyze")
            return
            
        df = self.compute_similarities(vectors)
        if not df.empty:
            self.visualize_similarities(df)
            self.save_results(df)
        else:
            logger.error("No similarities were computed")

if __name__ == "__main__":
    analyzer = SimilarityAnalyzer()
    analyzer.run()