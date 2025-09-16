"""
Utility functions for preparing GPT finetuning data.
Converts ranked JSON data to OpenAI JSONL format for finetuning.
"""

import json
from typing import List, Dict, Any, Iterable, Optional
from pathlib import Path


def lists_to_numbered_string(triples: List[Any]) -> str:
    """Convert a list of triples to a numbered string format."""
    return "\n".join(
        f"{i}. {' '.join(map(str, t)) if isinstance(t, (list, tuple)) else str(t)}"
        for i, t in enumerate(triples, 1)
    )


def _escape_json_string(s: str) -> str:
    """Escape special characters in a JSON string."""
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", "\\r")
    )


def _normalize_triple_entry(entry: Any) -> Any:
    """Normalize a triple entry to extract the actual triple."""
    if isinstance(entry, dict) and "triple" in entry:
        return entry["triple"]
    return entry


def _first_available_triples(sample: Dict[str, Any], limit: int) -> List[Any]:
    """Extract triples from a sample, checking multiple possible keys."""
    candidate_keys: Iterable[str] = ("retrived_triples_ranked", "retrieved_triples_ranked")
    for k in candidate_keys:
        if k in sample and sample[k]:
            seq = sample[k]
            out = [_normalize_triple_entry(x) for x in seq[:limit]]
            return out
    return []


def sparql_formatter(raw_query: str) -> str:
    """Format SPARQL query into standard <Answer> + JSON string format."""
    one_line = ' '.join(raw_query.strip().split())
    return "<Answer>\n" + json.dumps({"sparql": one_line}, ensure_ascii=False)


def prepare_training_jsonl(
    input_path: str,
    output_path: str,
    triples_limit: int = 10,
    start_index: int = 0,
    system_prompt: Optional[str] = None,
    knowledge_base: str = "DBpedia"
) -> int:
    """
    Prepare training JSONL file from ranked JSON data.
    
    Args:
        input_path: Path to input JSON file with ranked triples
        output_path: Path to output JSONL file for OpenAI finetuning
        triples_limit: Maximum number of triples to include per example
        start_index: Index to start from (for skipping initial examples)
        system_prompt: Custom system prompt (if None, uses default)
        knowledge_base: Knowledge base name (DBpedia or Wikidata)
    
    Returns:
        Number of training examples created
    """
    # Default system prompts
    if system_prompt is None:
        if knowledge_base == "Wikidata":
            system_prompt = (
                "You are an assistant that converts natural language questions into SPARQL queries for Wikidata. "
                "Given the user question, output only valid JSON in this format:\n\n"
                "{\n  \"sparql\": \"SPARQL QUERY HERE\"\n}"
            )
        else:  # DBpedia
            system_prompt = (
                "Given a specific question and up to ten potentially relevant triples, "
                "generate the corresponding SPARQL query for DBpedia. "
                "Return your answer after <Answer>, in JSON with key \"sparql\" and the query as its string value."
            )
    
    system_msg = {"role": "system", "content": system_prompt}
    
    # Load input data
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Prepare training examples
    training_examples = []
    
    for idx, sample in enumerate(dataset[start_index:], start=start_index):
        question = sample.get('question', '').strip()
        if not question:
            continue
        
        # Extract triples
        raw_hits = sample.get('retrived_triples_ranked', []) or sample.get('retrieved_triples_ranked', [])
        triples = []
        if raw_hits:
            triples = [_normalize_triple_entry(hit) for hit in raw_hits[:triples_limit]]
        triples_str = lists_to_numbered_string(triples) if triples else "(none)"
        
        # Create user message
        user_msg = {
            "role": "user",
            "content": f"Question:\n{question}\n\nCandidate Triples (max {triples_limit}, numbered):\n{triples_str}"
        }
        
        # Get gold query
        gold_query = sample.get('formated_query', '').strip()
        if not gold_query:
            continue
        
        # Create assistant message
        assistant_msg = {
            "role": "assistant",
            "content": sparql_formatter(gold_query)
        }
        
        # Create training example
        training_examples.append({
            "messages": [system_msg, user_msg, assistant_msg]
        })
    
    # Write JSONL file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for record in training_examples:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(training_examples)} training records to {output_path}")
    if training_examples:
        preview = json.dumps(training_examples[0], indent=2, ensure_ascii=False)
        print(f"Preview of first record:\n{preview[:700]}...")
    
    return len(training_examples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GPT training data from ranked JSON")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--triples_limit", type=int, default=10, help="Maximum triples per example")
    parser.add_argument("--start_index", type=int, default=0, help="Start index (skip initial examples)")
    parser.add_argument("--kb", type=str, default="DBpedia", choices=["DBpedia", "Wikidata"], 
                       help="Knowledge base type")
    
    args = parser.parse_args()
    
    prepare_training_jsonl(
        input_path=args.input,
        output_path=args.output,
        triples_limit=args.triples_limit,
        start_index=args.start_index,
        knowledge_base=args.kb
    )
