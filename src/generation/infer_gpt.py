"""
GPT inference utilities for SPARQL generation.
Handles batch inference and result extraction.
"""

import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from ..finetuning.gpt_data_prep import (
        lists_to_numbered_string,
        _escape_json_string,
        _normalize_triple_entry,
        _first_available_triples
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from finetuning.gpt_data_prep import (
        lists_to_numbered_string,
        _escape_json_string,
        _normalize_triple_entry,
        _first_available_triples
    )


GENERIC_INSTR = (
    'Given a specific question and up to ten potentially relevant triples, '
    'generate the corresponding SPARQL query for DBpedia. '
    'Return your answer after <Answer>, in JSON with key "sparql" and the query as its string value.'
)


def build_system_msg(sample: Dict[str, Any], num_demos: int = 1) -> Dict[str, str]:
    """
    Build system message with dynamic few-shot examples.
    
    Args:
        sample: Sample data with dynamic_pairs
        num_demos: Number of demo examples to include
    
    Returns:
        System message dict
    """
    demo_list = sample.get("dynamic_pairs") or sample.get("dynamic_paris") or []
    if not demo_list:
        return {"role": "system", "content": GENERIC_INSTR}
    
    blocks = []
    for i, demo in enumerate(demo_list[:num_demos], start=1):
        demo = demo or {}
        demo_q: str = str(demo.get("question", "")).strip()
        demo_sparql: str = str(demo.get("sparql", "")).strip()
        demo_triples = demo.get("retrieved_triples_top10", [])[:10]
        demo_triples_str = lists_to_numbered_string(demo_triples) if demo_triples else "(none)"
        
        if not demo_q or not demo_sparql:
            continue
        
        demo_answer = (
            "<Answer>\n"
            f"{{\"sparql\": \"{_escape_json_string(demo_sparql)}\"}}"
        )
        
        block = (
            f"Example {i} INPUT (exactly what you will receive for every task)\n\n"
            f"Question:\n{demo_q}\n\n"
            f"Candidate Triples (numbered, max 10):\n{demo_triples_str}\n\n"
            f"Example {i} OUTPUT (your response must follow **this exact shape**)\n\n"
            f"{demo_answer}\n"
        )
        blocks.append(block)
    
    if not blocks:
        return {"role": "system", "content": GENERIC_INSTR}
    
    header = (
        "Given a specific question and up to ten potentially relevant triples, generate the\n"
        "corresponding SPARQL query for DBpedia. Return your answer after <Answer>, in JSON\n"
        'with key "sparql" and the query as its string value.\n\n'
    )
    content = header + "\n".join(blocks)
    return {"role": "system", "content": content}


def prepare_inference_jsonl(
    input_path: str,
    output_path: str,
    model: str,
    triples_limit: int = 10,
    num_demos: int = 1
) -> int:
    """
    Prepare inference JSONL file from ranked JSON data.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSONL file
        model: Model name for batch inference
        triples_limit: Maximum triples per example
        num_demos: Number of dynamic few-shot examples
    
    Returns:
        Number of inference examples created
    """
    with open(input_path, encoding="utf-8") as f:
        dataset = json.load(f)
    
    jsonl_rows = []
    for sample in dataset:
        question = sample.get("question", "").strip()
        triples = _first_available_triples(sample, triples_limit)
        triples_str = lists_to_numbered_string(triples) if triples else "(none)"
        
        user_msg = {
            "role": "user",
            "content": f"Question:\n{question}\n\nCandidate Triples (max 10, numbered):\n{triples_str}"
        }
        system_msg = build_system_msg(sample, num_demos=num_demos)
        jsonl_rows.append({"messages": [system_msg, user_msg]})
    
    # Write inference JSONL
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for rec in jsonl_rows:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"[1/2] Wrote {len(jsonl_rows)} inference records to {output_path}")
    if jsonl_rows:
        preview = json.dumps(jsonl_rows[0], indent=2, ensure_ascii=False)
        print(f"Preview of first record:\n{preview[:900]}...")
    
    # Convert to OpenAI Batch JSONL
    batch_path = str(output_path_obj.parent / f"{output_path_obj.stem}_batch_input.jsonl")
    count = 0
    with open(output_path, "r", encoding="utf-8") as fin, \
         open(batch_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            messages = json.loads(line)["messages"]
            batch_row = {
                "custom_id": f"example_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": 0
                }
            }
            fout.write(json.dumps(batch_row) + "\n")
            count += 1
    
    print(f"[2/2] Wrote {count} batch lines to {batch_path}")
    return len(jsonl_rows)


def run_batch_inference(
    batch_input_path: str,
    output_path: str,
    completion_window: str = "24h",
    poll_interval: int = 60,
    api_key: Optional[str] = None
) -> str:
    """
    Run batch inference using OpenAI Batch API.
    
    Args:
        batch_input_path: Path to batch input JSONL file
        output_path: Path to save batch output
        completion_window: Completion window for batch
        poll_interval: Seconds between status checks
        api_key: OpenAI API key
    
    Returns:
        Batch ID
    """
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI()
    
    # Upload batch file
    print(f"Uploading batch file: {batch_input_path}")
    with open(batch_input_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    
    input_file_id = upload.id
    print(f"Uploaded file: {input_file_id}")
    
    # Create batch
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"job": "SPARQL generation inference"}
    )
    print(f"Batch ID: {batch.id}")
    
    # Monitor batch
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"Status: {batch.status}")
        if batch.status in {"failed", "completed"}:
            break
        time.sleep(poll_interval)
    
    if batch.status == "failed":
        print(f"Batch failed! Full batch object:")
        print(batch)
        raise RuntimeError("Batch inference failed")
    
    # Download results
    result_file_id = batch.output_file_id
    result_response = client.files.content(result_file_id)
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result_response.text)
    
    print(f"Saved outputs to {output_path}")
    return batch.id


def extract_sparql_from_response(content: str) -> str:
    """
    Extract SPARQL query from model response.
    
    Args:
        content: Model response content
    
    Returns:
        Extracted SPARQL query
    """
    # Try to find <Answer> tag with JSON
    answer_re = re.compile(r'<Answer>\s*(\{.*\})', re.DOTALL)
    m = answer_re.search(content)
    if m:
        try:
            return json.loads(m.group(1)).get("sparql", "")
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON directly
    json_re = re.compile(r'\{[^{}]*"sparql"[^{}]*\}', re.DOTALL)
    m = json_re.search(content)
    if m:
        try:
            return json.loads(m.group(0)).get("sparql", "")
        except json.JSONDecodeError:
            pass
    
    return ""


def merge_predictions(
    gold_path: str,
    predictions_path: str,
    output_path: str,
    prediction_key: str = "pred_query"
) -> int:
    """
    Merge predictions with gold data.
    
    Args:
        gold_path: Path to gold JSON file
        predictions_path: Path to batch output JSONL
        output_path: Path to output merged JSON
        prediction_key: Key to store predictions in output
    
    Returns:
        Number of records processed
    """
    # Load gold data
    with open(gold_path, encoding="utf-8") as f:
        gold_records = json.load(f)
    
    # Load predictions
    pred_lookup = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["custom_id"]
            content = rec["response"]["body"]["choices"][0]["message"]["content"]
            pred_lookup[cid] = extract_sparql_from_response(content)
    
    # Merge
    for idx, rec in enumerate(gold_records):
        cid = f"example_{idx}"
        rec[prediction_key] = pred_lookup.get(cid, "")
    
    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_records, f, ensure_ascii=False, indent=2)
    
    print(f"Enriched file written → {output_path}. Total records: {len(gold_records)}")
    return len(gold_records)


def main():
    parser = argparse.ArgumentParser(description="GPT inference for SPARQL generation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prepare inference data
    prep_parser = subparsers.add_parser("prepare", help="Prepare inference JSONL")
    prep_parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    prep_parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    prep_parser.add_argument("--model", type=str, required=True, help="Model name")
    prep_parser.add_argument("--triples_limit", type=int, default=10, help="Max triples")
    prep_parser.add_argument("--num_demos", type=int, default=1, help="Number of few-shot examples")
    
    # Run batch inference
    batch_parser = subparsers.add_parser("batch", help="Run batch inference")
    batch_parser.add_argument("--input", type=str, required=True, help="Batch input JSONL")
    batch_parser.add_argument("--output", type=str, required=True, help="Batch output JSONL")
    batch_parser.add_argument("--poll_interval", type=int, default=60, help="Poll interval (seconds)")
    batch_parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    
    # Merge predictions
    merge_parser = subparsers.add_parser("merge", help="Merge predictions with gold data")
    merge_parser.add_argument("--gold", type=str, required=True, help="Gold JSON file")
    merge_parser.add_argument("--predictions", type=str, required=True, help="Predictions JSONL")
    merge_parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    merge_parser.add_argument("--key", type=str, default="pred_query", help="Prediction key name")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_inference_jsonl(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            triples_limit=args.triples_limit,
            num_demos=args.num_demos
        )
    elif args.command == "batch":
        run_batch_inference(
            batch_input_path=args.input,
            output_path=args.output,
            poll_interval=args.poll_interval,
            api_key=args.api_key
        )
    elif args.command == "merge":
        merge_predictions(
            gold_path=args.gold,
            predictions_path=args.predictions,
            output_path=args.output,
            prediction_key=args.key
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
