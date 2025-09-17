#!/usr/bin/env python3
"""
SPARQL generation script using finetuned models.

Supports:
- Mistral-7B
- Mixtral-8x7B
- CodeLlama-34B
- GPT-3.5 Turbo
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_with_mistral(config: dict, input_path: str, output_path: str):
    """Generate SPARQL using finetuned Mistral model."""
    from generation.infer_mistral import generate_sparql
    generate_sparql(config, input_path, output_path)


def generate_with_mixtral(config: dict, input_path: str, output_path: str):
    """Generate SPARQL using finetuned Mixtral model."""
    from generation.infer_mixtral import generate_sparql
    generate_sparql(config, input_path, output_path)


def generate_with_codellama(config: dict, input_path: str, output_path: str):
    """Generate SPARQL using finetuned CodeLlama model."""
    from generation.infer_codellama import generate_sparql
    generate_sparql(config, input_path, output_path)


def generate_with_gpt35(config: dict, input_path: str, output_path: str):
    """Generate SPARQL using finetuned GPT-3.5 Turbo."""
    from generation.infer_gpt35 import generate_sparql
    generate_sparql(config, input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate SPARQL queries")
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
        choices=["mistral", "mixtral", "codellama", "gpt35"],
        help="Model to use for generation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for predictions"
    )
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    if args.model == "mistral":
        generate_with_mistral(config, args.input, args.output)
    elif args.model == "mixtral":
        generate_with_mixtral(config, args.input, args.output)
    elif args.model == "codellama":
        generate_with_codellama(config, args.input, args.output)
    elif args.model == "gpt35":
        generate_with_gpt35(config, args.input, args.output)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
