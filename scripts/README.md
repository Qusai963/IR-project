# Training Scripts

This directory contains scripts for training different types of models for the IR project.

## Scripts Overview

### 1. `train.py` - Combined Training Script

The main training script that can train both TF-IDF and BERT models.

**Usage:**

```bash
# Train both TF-IDF and BERT models for antique dataset
python scripts/train.py --dataset antique

# Train only TF-IDF model
python scripts/train.py --dataset antique --model tfidf

# Train only BERT model
python scripts/train.py --dataset antique --model bert

# Train with document limit
python scripts/train.py --dataset antique --limit 1000
```

**Arguments:**

- `--dataset`: Dataset name to train on (default: "antique")
- `--model`: Model type to train - "tfidf", "bert", or "both" (default: "both")
- `--limit`: Optional limit on number of documents to process

### 2. `train_tfidf.py` - TF-IDF Only Training

Dedicated script for training TF-IDF models only.

**Usage:**

```bash
python scripts/train_tfidf.py
```

This script will train TF-IDF models for both "antique" and "quora" datasets.

### 3. `train_bert.py` - BERT Only Training

Dedicated script for training BERT models only.

**Usage:**

```bash
python scripts/train_bert.py
```

This script will train BERT models for both "antique" and "quora" datasets.

## Architecture

### Ranking Services

The project now uses separate ranking services for different model types:

- **`TfidfRankingService`**: Optimized for TF-IDF sparse vectors
- **`BertRankingService`**: Optimized for BERT dense embeddings
- **`create_ranking_service()`**: Factory method to create appropriate service

This separation provides:

- Better type safety
- Optimized performance for each model type
- Easier extension for new ranking algorithms
- Clearer code organization

## Output Structure

Models are saved in the following directory structure:

```
app/saved_models/
├── antique/
│   ├── tfidf/
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── tfidf_vectors.joblib
│   │   └── doc_ids.json
│   └── bert/
│       ├── bert_model.joblib
│       ├── bert_embeddings.joblib
│       └── doc_ids.json
└── quora/
    ├── tfidf/
    └── bert/
```

## Prerequisites

1. Make sure you have ingested data to the database using `ingest_to_db.py`
2. Install all required dependencies from `requirements.txt`
3. Ensure the database service is properly configured

## Notes

- Each model type is saved in its own subdirectory for better organization
- The scripts automatically create the necessary directories
- Processed data is exported to JSONL format for reference
- Document IDs are preserved to maintain mapping between original and processed documents
