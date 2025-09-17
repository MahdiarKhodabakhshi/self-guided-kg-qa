#!/usr/bin/env python3
"""
Finetuning script for SPARQL generation models.

Supports:
- Mistral-7B
- Mixtral-8x7B
- CodeLlama-34B
- GPT-3.5 Turbo (via OpenAI API)
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def finetune_mistral(config: dict):
    """Finetune Mistral-7B model."""
    from finetuning.train_mistral import train_mistral
    train_mistral(config)


def finetune_mixtral(config: dict):
    """Finetune Mixtral-8x7B model."""
    from finetuning.train_mixtral import train_mixtral
    train_mixtral(config)


def finetune_codellama(config: dict):
    """Finetune CodeLlama-34B model."""
    from finetuning.train_codellama import train_codellama
    train_codellama(config)


def finetune_gpt(config: dict):
    """Finetune GPT-3.5 Turbo model via OpenAI API."""
    from finetuning.train_gpt import finetune_gpt as gpt_finetune
    return gpt_finetune(**config)


def main():
    parser = argparse.ArgumentParser(description="Finetune models for SPARQL generation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mistral", "mixtral", "codellama", "gpt"],
        help="Model to finetune"
    )
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    if args.model == "mistral":
        finetune_mistral(config)
    elif args.model == "mixtral":
        finetune_mixtral(config)
    elif args.model == "codellama":
        finetune_codellama(config)
    elif args.model == "gpt":
        results = finetune_gpt(config)
        print("\nFinetuning completed!")
        print(f"Fine-tuned model: {results.get('fine_tuned_model', 'N/A')}")
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
