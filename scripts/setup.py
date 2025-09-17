#!/usr/bin/env python3
"""
Setup script to download all necessary resources for the SPARQL generation pipeline.

This script downloads:
1. Datasets (QALD-9+, LC-QuAD, VQuanda)
2. Verifies Hugging Face models are accessible
3. Creates necessary directory structure
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error
from tqdm import tqdm

# Dataset URLs and information
DATASETS = {
    "qald": {
        "train": {
            "url": "https://github.com/ag-sc/QALD/raw/master/9/data/qald_9_plus_train_dbpedia.json",
            "filename": "qald_9_plus_train_dbpedia.json",
            "description": "QALD-9+ Training Set (DBpedia)"
        },
        "test": {
            "url": "https://github.com/ag-sc/QALD/raw/master/9/data/qald_9_plus_test_dbpedia.json",
            "filename": "qald_9_plus_test_dbpedia.json",
            "description": "QALD-9+ Test Set (DBpedia)"
        }
    },
    "lcquad": {
        "train": {
            "url": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/train.json",
            "filename": "lcquad_train.json",
            "description": "LC-QuAD 2.0 Training Set"
        },
        "test": {
            "url": "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/test.json",
            "filename": "lcquad_test.json",
            "description": "LC-QuAD 2.0 Test Set"
        }
    },
    "vquanda": {
        "train": {
            "url": "https://github.com/AskNowQA/VQuanda/raw/main/dataset/train.json",
            "filename": "vquanda_train.json",
            "description": "VQuanda Training Set"
        },
        "test": {
            "url": "https://github.com/AskNowQA/VQuanda/raw/main/dataset/test.json",
            "filename": "vquanda_test.json",
            "description": "VQuanda Test Set"
        }
    }
}

# Models to verify (Hugging Face model IDs)
MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "codellama": "codellama/CodeLlama-34b-Instruct-hf",
    "e5-large": "intfloat/e5-large",
    "colbert": "colbert-ir/colbertv2.0"
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress bar."""
    try:
        print(f"Downloading {description or output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file size for progress bar
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        with urllib.request.urlopen(url) as response, \
             open(output_path, 'wb') as out_file, \
             tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
        
        print(f"✓ Downloaded {output_path.name}")
        return True
    except urllib.error.URLError as e:
        print(f"✗ Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def verify_huggingface_model(model_id: str) -> bool:
    """Verify that a Hugging Face model is accessible."""
    try:
        from transformers import AutoTokenizer
        print(f"Verifying {model_id}...")
        # Try to load just the tokenizer (lightweight check)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        print(f"✓ {model_id} is accessible")
        return True
    except Exception as e:
        print(f"✗ {model_id} verification failed: {e}")
        print(f"  Note: Model will be downloaded automatically when first used")
        return False


def create_directory_structure(base_path: Path):
    """Create necessary directory structure."""
    directories = [
        base_path / "data" / "raw",
        base_path / "data" / "processed",
        base_path / "models" / "finetuned",
        base_path / "outputs",
        base_path / "checkpoints",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def download_datasets(datasets: List[str], splits: List[str], data_dir: Path):
    """Download specified datasets."""
    print("\n" + "=" * 60)
    print("Downloading Datasets")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            print(f"✗ Unknown dataset: {dataset_name}")
            continue
        
        for split in splits:
            if split not in DATASETS[dataset_name]:
                print(f"✗ Unknown split '{split}' for dataset {dataset_name}")
                continue
            
            dataset_info = DATASETS[dataset_name][split]
            output_path = data_dir / "raw" / dataset_info["filename"]
            
            # Skip if already exists
            if output_path.exists():
                print(f"⊘ {dataset_info['filename']} already exists, skipping...")
                continue
            
            total_count += 1
            if download_file(
                dataset_info["url"],
                output_path,
                dataset_info["description"]
            ):
                success_count += 1
    
    print(f"\nDownloaded {success_count}/{total_count} datasets")
    return success_count == total_count


def verify_models(models: List[str]):
    """Verify Hugging Face models are accessible."""
    print("\n" + "=" * 60)
    print("Verifying Hugging Face Models")
    print("=" * 60)
    
    models_to_check = []
    if "all" in models:
        models_to_check = list(MODELS.values())
    else:
        for model_name in models:
            if model_name in MODELS:
                models_to_check.append(MODELS[model_name])
            else:
                print(f"✗ Unknown model: {model_name}")
    
    success_count = 0
    for model_id in models_to_check:
        if verify_huggingface_model(model_id):
            success_count += 1
    
    print(f"\nVerified {success_count}/{len(models_to_check)} models")
    print("Note: Models will be automatically downloaded from Hugging Face when first used")


def main():
    parser = argparse.ArgumentParser(
        description="Setup script to download datasets and verify models for SPARQL generation pipeline"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["qald"],
        choices=["qald", "lcquad", "vquanda", "all"],
        help="Datasets to download (default: qald)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to download (default: train test)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["mistral", "mixtral", "codellama", "e5-large", "colbert", "all"],
        help="Models to verify (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for datasets (default: data)"
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset downloads"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model verification"
    )
    
    args = parser.parse_args()
    
    # Expand "all" for datasets
    if "all" in args.datasets:
        args.datasets = ["qald", "lcquad", "vquanda"]
    
    base_path = Path(__file__).parent.parent
    data_dir = base_path / args.data_dir
    
    print("=" * 60)
    print("SPARQL Generation Pipeline - Setup Script")
    print("=" * 60)
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure(base_path)
    
    # Download datasets
    if not args.skip_datasets:
        download_datasets(args.datasets, args.splits, data_dir)
    else:
        print("\n⊘ Skipping dataset downloads")
    
    # Verify models
    if not args.skip_models:
        verify_models(args.models)
    else:
        print("\n⊘ Skipping model verification")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review and update configs/pipeline_config.json with your paths")
    print("2. Run the pipeline: python scripts/run_pipeline.py --config configs/pipeline_config.json")
    print("\nNote: Models will be automatically downloaded from Hugging Face when first used.")


if __name__ == "__main__":
    main()
