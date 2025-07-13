# Information Retrieval (IR) Project

A comprehensive Information Retrieval system featuring multiple search algorithms, hybrid search strategies, and a modern web interface. This project implements TF-IDF, BM25, BERT embeddings, and hybrid search approaches with advanced features like RAG (Retrieval-Augmented Generation).

## 🚀 Features

### Core Search Algorithms

- **TF-IDF**: Traditional vector space model with cosine similarity
- **BM25**: Probabilistic ranking function for document retrieval
- **BERT**: Dense embeddings using sentence transformers
- **Hybrid Search**: Weighted combination of multiple algorithms

### Advanced Features

- **Query Refinement**: Automatic query expansion and improvement
- **Real-time Search**: Fast search with pre-computed embeddings
- **Model Management**: Automatic loading and caching of models
- **Evaluation Framework**: Comprehensive model evaluation with standard IR metrics

### Web Interface

- **Modern UI**: Responsive web interface with real-time search
- **Interactive Controls**: Adjustable weights for hybrid search
- **Visual Feedback**: Real-time search results and model status
- **API Documentation**: Auto-generated API docs with FastAPI

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL database
- CUDA-compatible GPU (optional, for BERT acceleration)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ir-project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

4. **Initialize the database**
   ```bash
   python scripts/ingest_to_db.py --dataset antique
   python scripts/ingest_to_db.py --dataset quora
   ```

## 🚀 Quick Start

1. **Train the models**

   ```bash
   # Train all models for antique dataset
   python scripts/train.py --dataset antique --model both

   # Train all models for quora dataset
   python scripts/train.py --dataset quora --model both
   ```

2. **Start the server**

   ```bash
   python app/main.py
   ```

3. **Access the web interface**
   - Open http://localhost:8000 in your browser
   - Use the interactive search interface

## 📚 Usage

### Web Interface

The web interface provides:

- **Search Bar**: Enter queries and see real-time results
- **Model Selection**: Choose between TF-IDF, BM25, BERT, or Hybrid
- **Hybrid Controls**: Adjust weights for TF-IDF, BERT, and BM25
- **Results Display**: View ranked documents with scores
- **Model Status**: Check which models are loaded and ready

### API Endpoints

#### Core Search

```bash
# Basic search
POST /api/search
{
  "query": "your search query",
  "dataset_name": "antique",
  "model": "hybrid_parallel",
  "top_k": 10,
  "weights": [0.4, 0.4, 0.2]  # [tfidf, bert, bm25]
}


```

#### Model Management

```bash
# Check model status
GET /api/models/status
```

#### Evaluation

```bash
# Evaluate models
python scripts/evaluate_models.py --model hybrid_parallel --dataset antique
```

### Command Line Tools

#### Training Models

```bash
# Train specific models
python scripts/train.py --dataset antique --model tfidf
python scripts/train.py --dataset antique --model bert
python scripts/train.py --dataset antique --model bm25

# Train with document limit
python scripts/train.py --dataset antique --limit 1000
```

#### Evaluation

```bash
# Evaluate specific model
python scripts/evaluate_models.py --model bm25 --dataset antique

# Evaluate with custom weights
python scripts/evaluate_models.py --model hybrid_parallel --dataset antique --weights 0.5 0.3 0.2

# Evaluate all models
python scripts/evaluate_models.py --model all --dataset antique
```

## 🏗️ Architecture

### Core Components

```
app/
├── api/                    # FastAPI endpoints
│   ├── search.py          # Core search API
│   ├── advanced_features.py # RAG and advanced features
│   ├── ranking.py         # Document ranking
│   └── ...
├── services/              # Business logic
│   ├── search_strategies/ # Search algorithm implementations
│   │   ├── tfidf_strategy.py
│   │   ├── bm25_strategy.py
│   │   ├── bert_strategy.py
│   │   └── hybrid_strategy.py
│   ├── model_manager.py   # Model loading and caching
│   └── ...
├── models/                # Data models and enums
├── static/                # Web interface
└── main.py               # FastAPI application
```

### Search Strategies

The system uses a strategy pattern for different search algorithms:

- **TFIDFSearchStrategy**: Vector space model with cosine similarity
- **BM25SearchStrategy**: Probabilistic ranking with proper tokenization
- **BERTSearchStrategy**: Dense embeddings with sentence transformers
- **HybridSearchStrategy**: Weighted combination of multiple strategies

### Model Management

- **Automatic Loading**: Models are loaded on startup and cached
- **On-Demand Loading**: Models can be loaded when needed
- **Memory Management**: Efficient memory usage with model sharing
- **Status Monitoring**: Real-time model status tracking

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/ir_db

# Model Settings
MODEL_CACHE_SIZE=1000
BERT_BATCH_SIZE=32

# Server Settings
HOST=0.0.0.0
PORT=8000
```

### Model Weights

For hybrid search, you can configure weights for different models:

```python
# Example weights
weights = [0.4, 0.4, 0.2]  # [TF-IDF, BERT, BM25]
weights = [0.5, 0.5, 0.0]  # TF-IDF + BERT only
weights = [1.0, 0.0, 0.0]  # TF-IDF only
```

## 📊 Evaluation

The system supports comprehensive evaluation using standard IR metrics:

- **MAP (Mean Average Precision)**
- **MRR (Mean Reciprocal Rank)**
- **Precision@k**
- **Recall@k**

### Running Evaluation

```bash
# Evaluate specific model
python scripts/evaluate_models.py --model bm25 --dataset antique

# Compare multiple models
python scripts/evaluate_models.py --model all --dataset antique

# Custom evaluation
python scripts/evaluate_models.py --model hybrid_parallel --dataset antique --weights 0.4 0.4 0.2
```

## 🔍 Advanced Features

### Query Refinement

Automatic query improvement and expansion:

```bash
POST /api/query-refinement/refine
{
  "query": "original query",
  "method": "expansion"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**

   ```bash
   # Check model status
   GET /api/models/status

   # Re-train models
   python scripts/train.py --dataset antique --model both
   ```

2. **Database Connection Issues**

   ```bash
   # Check database connection
   python scripts/ingest_to_db.py --dataset antique
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in .env
   BERT_BATCH_SIZE=16
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python app/main.py
```

## 📈 Performance

### Optimization Features

- **Model Caching**: Pre-loaded models for fast response
- **Batch Processing**: Efficient BERT embedding computation
- **Connection Pooling**: Optimized database connections

### Benchmarks

Typical performance metrics:

- **Search Response Time**: < 100ms for cached models
- **Model Loading Time**: 2-5 seconds per model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ir_datasets**: For providing evaluation datasets
- **sentence-transformers**: For BERT embeddings
- **FastAPI**: For the web framework
- **Transformers**: For language models

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the evaluation notebook

---

**Happy Searching! 🔍**
