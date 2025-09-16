"""
SPARQL evaluation utilities.

This module provides evaluation metrics for SPARQL query generation:
- BLEU score
- Exact match
- Execution-based metrics (Precision, Recall, F1)
"""

import json
import argparse
import os
import time
from typing import List, Dict, Set, Tuple, Any
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

try:
    from SPARQLWrapper import SPARQLWrapper, JSON as SPARQLJSON
except ImportError:
    SPARQLWrapper = None

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# DBpedia endpoint configuration
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
DBPEDIA_PREFIXES = """\
PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:     <http://www.w3.org/2002/07/owl#>
PREFIX xsd:     <http://www.w3.org/2001/XMLSchema#>
PREFIX foaf:    <http://xmlns.com/foaf/0.1/>
PREFIX dc:      <http://purl.org/dc/elements/1.1/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX skos:    <http://www.w3.org/2004/02/skos/core#>
PREFIX dbo:     <http://dbpedia.org/ontology/>
PREFIX dbp:     <http://dbpedia.org/property/>
PREFIX res:     <http://dbpedia.org/resource/>
PREFIX yago:    <http://dbpedia.org/class/yago/>
PREFIX geo:     <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX georss:  <http://www.georss.org/georss/>
PREFIX dbpedia: <http://dbpedia.org/>
PREFIX gold:    <http://purl.org/linguistics/gold/>
"""

MAX_RETRIES = 3
SLEEP_BETWEEN = 0.3


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation results from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def compute_bleu(gold: str, predicted: str) -> float:
    """
    Compute the BLEU score between a gold and predicted SPARQL query.
    """
    gold_tokens = word_tokenize(gold.lower())
    predicted_tokens = word_tokenize(predicted.lower())
    smoothing_fn = SmoothingFunction().method1
    score = sentence_bleu([gold_tokens], predicted_tokens, smoothing_function=smoothing_fn)
    return score


def evaluate_results(results: List[Dict[str, Any]]) -> Tuple[List[float], List[int], float, float]:
    """
    Compute BLEU scores and exact match accuracy for each result entry.
    
    Returns:
        Tuple of (bleu_scores, exact_matches, avg_bleu, exact_match_rate)
    """
    bleu_scores = []
    exact_matches = []
    
    for entry in results:
        gold_query = entry.get('formated_query', '') or entry.get('gold_query', '')
        predicted_query = entry.get('pred_query', '') or entry.get('predicted_query', '')
        
        gold_query = gold_query.strip()
        predicted_query = predicted_query.strip()
        
        bleu = compute_bleu(gold_query, predicted_query)
        bleu_scores.append(bleu)
        exact_matches.append(int(gold_query == predicted_query))
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    exact_match_rate = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    
    return bleu_scores, exact_matches, avg_bleu, exact_match_rate


def execute_sparql_query(query: str, endpoint_url: str = DBPEDIA_ENDPOINT, var: str = "?uri") -> Set[str]:
    """
    Executes a SPARQL query on the provided endpoint and returns the result.
    """
    if SPARQLWrapper is None:
        raise ImportError("SPARQLWrapper is not installed. Please install it to use execution-based evaluation.")
    
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(DBPEDIA_PREFIXES + "\n" + (query or "").lstrip())
    sparql.setReturnFormat(SPARQLJSON)
    sparql.setTimeout(30)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            
            if not bindings:
                return set()
            
            # Determine the variable key
            key = (
                next(iter(bindings[0].keys()))
                if var in ("?first", "?uri") and bindings
                else var.lstrip("?")
            )
            
            return {row[key]["value"] for row in bindings if key in row}
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"Query failed after {MAX_RETRIES} attempts: {str(e)[:120]}...")
                return set()
            time.sleep(1.5 * attempt)
    
    return set()


def compare_query_results(gold_results: List[Dict], predicted_results: List[Dict]) -> Tuple[float, float, float]:
    """
    Compare two sets of query results and compute precision, recall, and F1 score.
    
    For simplicity, this function converts the list of result dictionaries into sets of tuples.
    """
    gold_set = {tuple(sorted(item.items())) for item in gold_results} if gold_results else set()
    predicted_set = {tuple(sorted(item.items())) for item in predicted_results} if predicted_results else set()
    
    true_positives = len(gold_set.intersection(predicted_set))
    precision = true_positives / len(predicted_set) if predicted_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def prf(pred: Set[str], gold: Set[str]) -> Tuple[int, float, float, float]:
    """
    Compute precision, recall, F1, and exact match from result sets.
    
    Returns:
        Tuple of (exact_match, precision, recall, f1)
    """
    if not gold and not pred:
        return 1, 1.0, 1.0, 1.0
    if not pred:
        return 0, 0.0, 0.0, 0.0
    
    common = len(pred & gold)
    prec = common / len(pred)
    rec = common / len(gold) if gold else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    em = int(pred == gold)
    
    return em, prec, rec, f1


def evaluate_execution(
    results: List[Dict[str, Any]],
    endpoint_url: str = DBPEDIA_ENDPOINT,
    sleep_between: float = SLEEP_BETWEEN
) -> Tuple[List[float], List[float], List[float], float, float, float]:
    """
    Execute both the gold and predicted queries on the given SPARQL endpoint,
    and compute precision, recall, and F1 score for each test case.
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for entry in tqdm(results, desc="Executing queries"):
        gold_query = entry.get('formated_query', '') or entry.get('gold_query', '')
        predicted_query = entry.get('pred_query', '') or entry.get('predicted_query', '')
        
        gold_results = execute_sparql_query(gold_query, endpoint_url)
        time.sleep(sleep_between)
        predicted_results = execute_sparql_query(predicted_query, endpoint_url)
        time.sleep(sleep_between)
        
        if gold_results is None or predicted_results is None:
            continue  # Skip this entry if execution failed
        
        em, precision, recall, f1 = prf(predicted_results, gold_results)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return precision_scores, recall_scores, f1_scores, avg_precision, avg_recall, avg_f1


def safe_convert(val: Any) -> Any:
    """
    Recursively convert numpy arrays and numpy scalars to native Python types.
    """
    if isinstance(val, np.ndarray):
        return safe_convert(val.tolist())
    elif isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    elif isinstance(val, (np.float64, np.float32, np.float16)):
        return float(val)
    elif isinstance(val, list):
        return [safe_convert(item) for item in val]
    elif isinstance(val, dict):
        return {key: safe_convert(value) for key, value in val.items()}
    else:
        return val


def generate_report(
    results_filepath: str,
    endpoint_url: Optional[str] = None,
    report_csv: str = 'evaluation_report.csv',
    test_name: Optional[str] = None,
    num_demos: Optional[int] = None,
    metrics_append_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates an evaluation report:
     - Computes BLEU scores and exact match accuracy.
     - If an endpoint is provided, performs execution-based evaluation.
     - Saves a CSV report with individual metrics.
    """
    results = load_results(results_filepath)
    
    if not results:
        raise ValueError(f"No results found in {results_filepath}")
    
    bleu_scores, exact_matches, avg_bleu, exact_match_rate = evaluate_results(results)
    
    report_data = []
    for idx, entry in enumerate(results):
        report_entry = safe_convert({
            'id': entry.get('id', ''),
            'question': entry.get('question', ''),
            'gold_query': entry.get('formated_query', '') or entry.get('gold_query', ''),
            'predicted_query': entry.get('pred_query', '') or entry.get('predicted_query', ''),
            'BLEU_score': bleu_scores[idx],
            'Exact_match': exact_matches[idx]
        })
        report_data.append(report_entry)
    
    df = pd.DataFrame(report_data)
    df.to_csv(report_csv, index=False)
    
    print("----- Automatic Evaluation -----")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Exact Match Rate: {exact_match_rate*100:.2f}%")
    
    # Execution-based evaluation
    if endpoint_url:
        print("\n----- Execution-based Evaluation -----")
        precision_scores, recall_scores, f1_scores, avg_precision, avg_recall, avg_f1 = evaluate_execution(
            results, endpoint_url
        )
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        
        # Add execution metrics to dataframe
        df['Precision'] = precision_scores[:len(df)]
        df['Recall'] = recall_scores[:len(df)]
        df['F1'] = f1_scores[:len(df)]
        df.to_csv(report_csv, index=False)
        
        # Append to metrics CSV if provided
        if metrics_append_csv and test_name is not None and num_demos is not None:
            row = {
                "timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "test_name": test_name,
                "num_demos": num_demos,
                "examples": len(results),
                "exact_match": exact_match_rate,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "data_file": str(results_filepath),
            }
            append_csv_row(metrics_append_csv, row)
    else:
        print("\nNo SPARQL endpoint provided; skipping execution-based evaluation.")
    
    return df


def append_csv_row(csv_path: str, row: dict):
    """Append a row to a CSV metrics file."""
    header = ["timestamp_iso", "test_name", "num_demos", "examples", "exact_match", "precision", "recall", "f1", "data_file"]
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        vals = [
            row["timestamp_iso"],
            row["test_name"],
            str(row["num_demos"]),
            str(row["examples"]),
            f'{row["exact_match"]:.6f}',
            f'{row["precision"]:.6f}',
            f'{row["recall"]:.6f}',
            f'{row["f1"]:.6f}',
            row["data_file"],
        ]
        f.write(",".join(vals) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated SPARQL queries.")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to JSON file with evaluation results."
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default=None,
        help="SPARQL endpoint URL for execution-based evaluation (optional)."
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="evaluation_report.csv",
        help="Path to save the CSV evaluation report."
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default=None,
        help="Test name for metrics tracking."
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=None,
        help="Number of few-shot examples used."
    )
    parser.add_argument(
        "--metrics_append_csv",
        type=str,
        default=None,
        help="CSV file to append aggregated metrics."
    )
    
    args = parser.parse_args()
    
    generate_report(
        args.results_file,
        args.endpoint_url,
        args.report_csv,
        args.test_name,
        args.num_demos,
        args.metrics_append_csv
    )


if __name__ == "__main__":
    main()
