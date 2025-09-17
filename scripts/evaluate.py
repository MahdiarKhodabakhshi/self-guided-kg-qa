#!/usr/bin/env python3
"""
Evaluation script for SPARQL generation results.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import generate_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPARQL generation results")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to JSON file with evaluation results (must have 'formated_query' or 'gold_query' and 'pred_query' or 'predicted_query')"
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default=None,
        help="SPARQL endpoint URL for execution-based evaluation (default: DBpedia)"
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="evaluation_report.csv",
        help="Path to save the CSV evaluation report"
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default=None,
        help="Test name for metrics tracking"
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=None,
        help="Number of few-shot examples used"
    )
    parser.add_argument(
        "--metrics_append_csv",
        type=str,
        default=None,
        help="CSV file to append aggregated metrics"
    )
    
    args = parser.parse_args()
    
    # Use DBpedia endpoint if not specified
    endpoint_url = args.endpoint_url or "https://dbpedia.org/sparql"
    
    generate_report(
        args.results_file,
        endpoint_url,
        args.report_csv,
        args.test_name,
        args.num_demos,
        args.metrics_append_csv
    )


if __name__ == "__main__":
    main()
