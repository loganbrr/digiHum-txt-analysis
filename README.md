# FOMC Minutes Analysis

This project analyzes Federal Reserve Open Market Committee (FOMC) meeting minutes:

1. Download and process FOMC meeting minutes
2. Clean and normalize the text
3. Convert text to vector representations
4. Compute similarity between consecutive releases

## Project Structure

```
fed-data/
├── data/
│   ├── raw/           # Original PDF files
│   ├── cleaned/       # Cleaned .txt files
│   ├── vectors/       # Vectorized representations
│   └── results/       # Analysis results and visualizations
├── src/
│   ├── data_ingestion.py
│   ├── text_cleaner.py
│   ├── vectorizer.py
│   └── similarity_analyzer.py
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Features

- **Data Ingestion**: Downloads FOMC meeting minutes from the Federal Reserve website
- **Text Cleaning**: Normalizes and cleans the text for analysis
- **Vectorization**: Converts text to vector representations using either TF-IDF or BERT
- **Similarity Analysis**: Computes cosine similarity between consecutive releases
- **Visualization**: Generates plots and heatmaps of similarity scores

## Usage

The project can be run in sequence using the following commands:

1. Download and process FOMC minutes (will not re-process old files)

```bash
python src/data_ingestion.py
```

2. Clean the text:

```bash
python src/text_cleaner.py
```

3. Vectorize the text:

```bash
python src/vectorizer.py
```

4. Analyze similarities:

```bash
python src/similarity_analyzer.py
```

## Output

The analysis generates:

- Cleaned text files in `data/cleaned/`
- Vector representations in `data/vectors/`
- Similarity scores in `data/results/similarity_scores.csv`
- Visualizations in `data/results/`:
  - `similarity_trend.png`: Line plot of similarity over time
  - `similarity_matrix.png`: Heatmap of similarity between releases

## Dependencies

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## License

MIT License
