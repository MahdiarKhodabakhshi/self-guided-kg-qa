# Self-Guided Few-Shot Refinement for Knowledge Graph Question Answering

A self-guided framework for knowledge-graph question answering that maps natural-language questions (NLQ) to executable SPARQL. The pipeline retrieves and ranks grounding triples, generates an initial “hypothesis” query, and then refines it via Hybrid Example Search (HES), dynamic few-shot selection that combines semantic similarity (question) and structural similarity (SPARQL pattern), optionally augmented with concise chain-of-thought rationales to guide controlled revision. The repository includes modular components for preprocessing, entity linking, triple retrieval, ColBERT ranking, model fine-tuning/inference, and evaluation on standard KGQA benchmarks (QALD-9, LC-QuAD-1, VQuAnDa) over DBpedia endpoints.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Models Supported](#models-supported)
- [Datasets Supported](#datasets-supported)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Pipeline Overview

The pipeline consists of 8 main stages:

1. **Preprocessing**: Normalize and filter datasets
2. **SPARQL Parsing**: Extract triples from SPARQL queries
3. **Entity Extraction**: Extract entities from natural language questions
4. **Triple Retrieval**: Retrieve relevant triples from knowledge base
5. **Ranking**: Rank triples using ColBERT
6. **Finetuning**: Finetune language models on SPARQL generation task
7. **Initial SPARQL Generation**: Generate initial queries using finetuned models
8. **Dynamic Few-Shot Selection**: Select similar examples based on semantic and structural similarity
9. **Chain of Thought Generation**: Generate reasoning chains for few-shot examples
10. **Final SPARQL Generation**: Generate final queries with dynamic few-shot + CoT

## Repository Structure

```
sparql_generation_pipeline/
├── src/
│   ├── preprocessing/      # Dataset preprocessing modules
│   ├── retrieval/         # Entity extraction and triple retrieval
│   ├── ranking/           # ColBERT ranking modules
│   ├── finetuning/        # Model finetuning scripts
│   ├── generation/        # SPARQL generation scripts
│   ├── evaluation/        # Evaluation utilities
│   └── utils/             # Shared utilities
├── configs/               # Configuration files
├── scripts/               # Main pipeline scripts
├── data/                  # Data directories
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── models/               # Model checkpoints
├── outputs/              # Output files
└── notebooks/            # Jupyter notebooks for analysis
```

## Installation

### Quick Setup (Recommended)

The repository includes an automated setup script that downloads all necessary datasets and verifies model access:

```bash
# Clone the repository
git clone git@github.com:MahdiarKhodabakhshi/self-guided-kg-qa.git
cd self-guided-kg-qa

# Install dependencies
pip install -r requirements.txt

# Run setup script to download datasets and verify models
python scripts/setup.py --datasets all --splits train test --models all
```

This will:
- Download QALD-9+, LC-QuAD, and VQuanda datasets (train and test splits)
- Verify access to all Hugging Face models (they'll auto-download when first used)
- Create necessary directory structure

### Manual Installation

If you prefer manual setup:

```bash
# Clone or navigate to the repository
cd sparql_generation_pipeline

# Install dependencies
pip install -r requirements.txt

# Install ColBERT (if not already installed)
pip install colbert-ai

# Manually download datasets to data/raw/ directory
# - QALD-9+: https://github.com/ag-sc/QALD
# - LC-QuAD: https://github.com/AskNowQA/LC-QuAD2.0
# - VQuanda: https://github.com/AskNowQA/VQuanda
```

### Setup Script Options

```bash
# Download only QALD dataset
python scripts/setup.py --datasets qald

# Download specific datasets
python scripts/setup.py --datasets qald lcquad --splits train

# Verify only specific models
python scripts/setup.py --skip-datasets --models mistral mixtral

# Skip model verification (models will download automatically when used)
python scripts/setup.py --datasets all --skip-models
```

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for model finetuning and inference)
- OpenAI API key (for GPT-3.5 finetuning and CoT generation)
- Access to SPARQL endpoints (DBpedia, Wikidata)

## Quick Start

### Initial Setup

First, run the setup script to download all necessary resources:

```bash
# Download all datasets (QALD, LC-QuAD, VQuanda) for train and test splits
python scripts/setup.py --datasets all --splits train test

# Or download only specific datasets
python scripts/setup.py --datasets qald --splits train test
```

This will:
- Download datasets to `data/raw/` directory
- Verify Hugging Face model access (models auto-download when first used)
- Create necessary directory structure

### 1. Preprocessing

```bash
python scripts/run_pipeline.py \
    --config configs/pipeline_config.json \
    --stages preprocessing sparql_parsing
```

### 2. Entity Extraction and Retrieval

```bash
python scripts/run_pipeline.py \
    --config configs/pipeline_config.json \
    --stages entity_extraction triple_retrieval
```

### 3. Ranking

```bash
python scripts/run_pipeline.py \
    --config configs/pipeline_config.json \
    --stages ranking
```

### 4. Finetuning

```bash
# Mistral
python scripts/finetune.py \
    --config configs/pipeline_config.json \
    --model mistral

# Mixtral
python scripts/finetune.py \
    --config configs/pipeline_config.json \
    --model mixtral

# CodeLlama
python scripts/finetune.py \
    --config configs/pipeline_config.json \
    --model codellama
```

### 5. Generation

```bash
python scripts/generate_sparql.py \
    --config configs/pipeline_config.json \
    --model mistral \
    --input data/processed/test_ranked.json \
    --output outputs/test_predictions.json
```

### 6. Evaluation

```bash
python scripts/evaluate.py \
    --results_file outputs/test_predictions.json \
    --endpoint_url https://dbpedia.org/sparql
```

## Detailed Usage

### Running the Full Pipeline

```bash
python scripts/run_pipeline.py --config configs/pipeline_config.json
```

### Running Individual Stages

```bash
# Preprocessing only
python scripts/run_pipeline.py --config configs/pipeline_config.json --stages preprocessing

# Multiple stages
python scripts/run_pipeline.py --config configs/pipeline_config.json --stages preprocessing entity_extraction triple_retrieval
```

### Dynamic Few-Shot Selection

```python
from src.generation.dy_few_shot import attach_dynamic_pairs_to_dataset
import json

# Load test data with predictions
with open("outputs/test_predictions.json") as f:
    test_data = json.load(f)

# Load training data
with open("data/processed/train_ranked.json") as f:
    train_data = json.load(f)

# Attach dynamic pairs
augmented = attach_dynamic_pairs_to_dataset(
    dataset=test_data,
    trainset=train_data,
    top_k=5,
    lambda_semantic=0.4
)

# Save
with open("outputs/test_few_shot.json", "w") as f:
    json.dump(augmented, f, indent=2)
```

### Chain of Thought Generation

See `notebooks/dy_cot_adder.ipynb` for CoT generation using GPT-4.

### Final Generation with Few-Shot

```bash
python scripts/generate_sparql.py \
    --config configs/pipeline_config.json \
    --model mistral \
    --input outputs/test_few_shot.json \
    --output outputs/test_final.json
```

## Pipeline Stages

### Stage 1: Preprocessing

**Purpose**: Normalize datasets to a common format and filter by language.

**Input**: Raw dataset files (QALD, LC-QuAD, VQuanda)  
**Output**: Normalized JSON files with consistent schema

**Key Components**:
- `QALDPreprocessor`: Handles QALD-9+ datasets
- `LCQAPreprocessor`: Handles LC-QuAD datasets
- `VQuandaPreprocessor`: Handles VQuanda datasets

**Output Schema**:
```json
{
  "id": "question_id",
  "question": "Natural language question",
  "formated_query": "SPARQL query with shortened URIs",
  "answers": []
}
```

### Stage 2: SPARQL Parsing

**Purpose**: Extract triples from SPARQL queries.

**Input**: Preprocessed data with `formated_query` field  
**Output**: Data with `triples` field containing extracted triples

**Key Components**:
- `SparqlParser`: Extracts triples from WHERE clauses

### Stage 3: Entity Extraction

**Purpose**: Extract entities from natural language questions.

**Input**: Preprocessed data with questions  
**Output**: Data with `entities` field containing extracted entity URIs

**Key Components**:
- `RefinedEntityExtractor`: Uses Refined model for entity linking
- `FalconEntityExtractor`: Alternative using FALCON API

**Entity Format**: DBpedia resource names (e.g., "Skype", "United_States")

### Stage 4: Triple Retrieval

**Purpose**: Retrieve relevant triples from knowledge base for each entity.

**Input**: Data with extracted entities  
**Output**: Data with `retrieved_triples` field

**Key Components**:
- `DBpediaRetriever`: Retrieves triples from DBpedia SPARQL endpoint
- `WikidataRetriever`: Retrieves triples from Wikidata SPARQL endpoint

**Triple Format**: List of (subject, predicate, object) tuples

### Stage 5: Ranking

**Purpose**: Rank retrieved triples by relevance to the question.

**Input**: Data with retrieved triples  
**Output**: Data with `retrived_triples_ranked` field

**Key Components**:
- `colbert_ranker.py`: Uses ColBERT v2 for semantic ranking
- `parallel_ranker.py`: Parallel processing for large datasets
- `bm25_ranker.py`: BM25-based ranking (alternative)

**Ranking Method**: ColBERT v2 with cosine similarity between question and triple embeddings

### Stage 6: Finetuning

**Purpose**: Finetune language models on SPARQL generation task.

**Input**: Training data with ranked triples and gold SPARQL queries  
**Output**: Finetuned model checkpoints

**Supported Models**:
- **Mistral-7B**: LoRA finetuning with 4-bit quantization
- **Mixtral-8x7B**: LoRA finetuning with 4-bit quantization
- **CodeLlama-34B**: LoRA finetuning with 4-bit quantization
- **GPT-3.5 Turbo**: OpenAI API finetuning

**Training Format**: 
- System prompt with task description
- User message with question and top-k triples
- Assistant message with gold SPARQL query

### Stage 7: Initial SPARQL Generation

**Purpose**: Generate initial SPARQL queries using finetuned models.

**Input**: Test data with ranked triples  
**Output**: Data with `pred_query` field containing generated SPARQL

**Key Components**:
- Model-specific inference scripts
- Prompt formatting utilities
- SPARQL extraction and validation

### Stage 8: Dynamic Few-Shot Selection

**Purpose**: Select similar training examples for few-shot learning.

**Input**: Test data with initial predictions, training data  
**Output**: Data with `dynamic_pairs` field containing selected examples

**Selection Method**:
- Semantic similarity: Uses sentence transformers (e5-large) to find similar questions
- Structural similarity: Compares SPARQL query structure (decorators, motifs)
- Combined score: `lambda_semantic * semantic_score + (1 - lambda_semantic) * structural_score`

**Key Components**:
- `dy_few_shot.py`: Main few-shot selection module
- FAISS index for efficient semantic search

### Stage 9: Chain of Thought Generation

**Purpose**: Generate reasoning chains explaining how SPARQL queries are derived.

**Input**: Few-shot examples with questions, triples, and SPARQL  
**Output**: Data with `cot` field for each few-shot example

**Generation Method**: Uses GPT-4 to generate concise explanations (2-4 sentences)

**Key Components**:
- `dy_cot_adder.ipynb`: Notebook for CoT generation
- OpenAI batch API for efficient processing

### Stage 10: Final SPARQL Generation

**Purpose**: Generate final SPARQL queries with dynamic few-shot + CoT.

**Input**: Test data with dynamic pairs and CoT  
**Output**: Final SPARQL predictions

**Prompt Structure**:
- System message with task description
- Few-shot examples (with CoT if available)
- User message with question and triples

## Configuration

All configuration is done via `configs/pipeline_config.json`:

```json
{
  "dataset": "qald",
  "split": "train",
  "paths": {
    "raw_data": "data/raw/qald_9_plus_train_dbpedia.json",
    "preprocessed_data": "data/processed/qald_train_preprocessed.json",
    "parsed_data": "data/processed/qald_train_parsed.json",
    "entities_data": "data/processed/qald_train_entities.json",
    "retrieved_data": "data/processed/qald_train_retrieved.json",
    "ranked_data": "data/processed/qald_train_ranked.json",
    "collection_tsv": "data/processed/qald_collection.tsv",
    "colbert_index": "data/processed/qald_colbert_index"
  },
  "retrieval": {
    "endpoint": "https://dbpedia.org/sparql",
    "timeout": 3000,
    "max_retries": 5,
    "retry_sleep": 10,
    "checkpoint_file": "checkpoints/retrieval_checkpoint.json"
  },
  "ranking": {
    "top_k": 200,
    "nbits": 2
  },
  "finetuning": {
    "model": "mistral",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "output_dir": "models/finetuned"
  },
  "generation": {
    "top_k_triples": 10,
    "temperature": 0.0,
    "max_tokens": 512
  },
  "few_shot": {
    "top_k": 5,
    "lambda_semantic": 0.4,
    "model_name": "intfloat/e5-large"
  }
}
```

### Configuration Options

- **Dataset paths**: Configure input/output paths for each stage
- **Model parameters**: Set model-specific hyperparameters
- **Retrieval settings**: Configure SPARQL endpoint, timeouts, retries
- **Ranking parameters**: Set ColBERT ranking parameters
- **Finetuning hyperparameters**: Epochs, batch size, learning rate
- **Generation settings**: Temperature, max tokens, top-k triples
- **Few-shot settings**: Number of examples, semantic/structural weights

## Models Supported

### Mistral-7B
- **Finetuning**: LoRA with 4-bit quantization
- **Inference**: PEFT adapters with dynamic prompt fitting
- **Script**: `src/finetuning/train_mistral.py`, `src/generation/infer_mistral.py`

### Mixtral-8x7B
- **Finetuning**: LoRA with 4-bit quantization
- **Inference**: PEFT adapters with dynamic prompt fitting
- **Script**: `src/finetuning/train_mixtral.py`, `src/generation/infer_mixtral.py`

### CodeLlama-34B
- **Finetuning**: LoRA with 4-bit quantization
- **Inference**: PEFT adapters with dynamic prompt fitting
- **Script**: `src/finetuning/train_codellama.py`, `src/generation/infer_codellama.py`

### GPT-3.5 Turbo
- **Finetuning**: OpenAI API finetuning
- **Inference**: OpenAI API with batch processing
- **Script**: `src/finetuning/train_gpt.py`, `src/generation/infer_gpt.py`

#### GPT-3.5 Turbo Finetuning Guide

**Prerequisites**:
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Ranked JSON data with triples and gold SPARQL queries

**Quick Start**:

1. **Prepare Training Data**:
```bash
python -m src.finetuning.gpt_data_prep \
    --input data/processed/qald_train_ranked_triples.json \
    --output data/processed/qald_train_finetuning.jsonl \
    --triples_limit 10 \
    --kb DBpedia
```

2. **Finetune Model**:
```bash
python scripts/finetune.py \
    --model gpt \
    --config configs/gpt_finetuning_config.json
```

Or use the direct script:
```bash
python -m src.finetuning.train_gpt \
    --input data/processed/qald_train_ranked_triples.json \
    --output data/processed/qald_train_finetuning.jsonl \
    --model gpt-3.5-turbo \
    --n_epochs 3 \
    --kb DBpedia
```

3. **Inference**:
```bash
# Prepare inference data
python -m src.generation.infer_gpt prepare \
    --input data/processed/qald_test_ranked_triples.json \
    --output data/processed/qald_test_inference.jsonl \
    --model ft:gpt-3.5-turbo:personal::xxx \
    --triples_limit 10 \
    --num_demos 1

# Run batch inference
python -m src.generation.infer_gpt batch \
    --input data/processed/qald_test_inference_batch_input.jsonl \
    --output data/processed/qald_test_batch_output.jsonl \
    --poll_interval 60

# Merge predictions
python -m src.generation.infer_gpt merge \
    --gold data/processed/qald_test_ranked_triples.json \
    --predictions data/processed/qald_test_batch_output.jsonl \
    --output data/processed/qald_test_with_predictions.json \
    --key pred_query
```

**Advanced Options**:
- Skip steps: `--skip_prep`, `--skip_upload`, `--skip_job`
- Validation split: `--validation data/processed/qald_val_ranked_triples.json`
- Monitor existing job: `--job_id ftjob-xxx`

## Datasets Supported

- **QALD-9+**: Question Answering over Linked Data
- **LC-QuAD**: Large-scale Complex Question Answering Dataset
- **VQuanda**: Verbalized Queries Dataset

## Evaluation

The evaluation module computes multiple metrics:

- **BLEU Score**: Token-level similarity between predicted and gold SPARQL queries
- **Exact Match**: String-level exact match accuracy
- **Execution Metrics** (if endpoint provided):
  - **Precision**: |pred ∩ gold| / |pred|
  - **Recall**: |pred ∩ gold| / |gold|
  - **F1**: 2 * (precision * recall) / (precision + recall)
  - **Exact Match**: pred == gold (set comparison)

**Usage**:
```bash
python scripts/evaluate.py \
    --results_file outputs/predictions.json \
    --endpoint_url https://dbpedia.org/sparql
```

The evaluation script generates a CSV report with detailed metrics for each example.

## Troubleshooting

### ColBERT Index Not Found

If ColBERT index doesn't exist, the ranking stage will create it automatically. This may take time for large datasets. The index is saved and can be reused across runs.

### Out of Memory

For large models (CodeLlama-34B, Mixtral), use:
- 4-bit quantization (already enabled in finetuning scripts)
- Gradient checkpointing
- Smaller batch sizes (adjust in config)

### Entity Extraction Slow

Refined model loading takes time. Consider:
- Pre-loading the model once
- Using checkpoints for large datasets
- Parallel processing

### GPT Finetuning Issues

**File Upload Fails**:
- Check file size (must be < 512MB)
- Ensure file is valid JSONL format
- Verify OpenAI API key is set correctly

**Job Fails**:
- Check job error message: `client.fine_tuning.jobs.retrieve(job_id).error`
- Verify training data format is correct
- Ensure sufficient OpenAI credits

**Model Not Available**:
- Wait for job to complete (check status)
- Verify job succeeded: `job.status == "succeeded"`
- Check fine-tuned model ID: `job.fine_tuned_model`

### Pipeline Checkpointing

The pipeline supports checkpointing for long-running stages:
- Retrieval: Checkpoints saved to `checkpoints/retrieval_checkpoint.json`
- Ranking: ColBERT indices can be reused
- Generation: Can resume from last processed example

## Output Files

The pipeline generates the following output files:

1. `data/processed/{dataset}_{split}_preprocessed.json` - Preprocessed data
2. `data/processed/{dataset}_{split}_parsed.json` - Parsed SPARQL
3. `data/processed/{dataset}_{split}_entities.json` - Extracted entities
4. `data/processed/{dataset}_{split}_retrieved.json` - Retrieved triples
5. `data/processed/{dataset}_{split}_ranked.json` - Ranked triples
6. `models/finetuned/{model}_{dataset}/` - Finetuned model checkpoints
7. `outputs/{dataset}_{split}_initial.json` - Initial SPARQL predictions
8. `outputs/{dataset}_{split}_few_shot.json` - Data with dynamic pairs
9. `outputs/{dataset}_{split}_cot.json` - Data with CoT
10. `outputs/{dataset}_{split}_final.json` - Final SPARQL predictions

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Specify your license] -->
