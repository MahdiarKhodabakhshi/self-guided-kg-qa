"""
Generation modules for SPARQL query generation.
"""

from .dy_few_shot import (
    attach_dynamic_pairs_to_dataset,
    semantic_topk,
    structural_score,
)

__all__ = [
    "attach_dynamic_pairs_to_dataset",
    "semantic_topk",
    "structural_score",
]

# GPT inference utilities
try:
    from .infer_gpt import (
        prepare_inference_jsonl,
        run_batch_inference,
        merge_predictions,
        extract_sparql_from_response
    )
    __all__.extend([
        "prepare_inference_jsonl",
        "run_batch_inference",
        "merge_predictions",
        "extract_sparql_from_response"
    ])
except ImportError:
    pass
