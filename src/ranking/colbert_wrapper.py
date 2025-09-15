"""
ColBERT ranking wrapper for triple ranking.

This module provides a clean interface for ranking triples using ColBERT v2.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher


def build_collection_tsv(
    data: List[Dict[str, Any]],
    collection_path: str,
    triple_field: str = "retrieved_triples"
) -> tuple:
    """
    Build a TSV collection file for ColBERT indexing.
    
    Args:
        data: List of data entries with questions and triples
        collection_path: Path to output TSV file
        triple_field: Field name containing retrieved triples
        
    Returns:
        Tuple of (doc_id_to_triple, doc_id_to_question_id) dictionaries
    """
    os.makedirs(os.path.dirname(collection_path) or ".", exist_ok=True)
    
    doc_id = 0
    doc_id_to_triple = {}
    doc_id_to_question_id = {}
    
    with open(collection_path, "w", encoding="utf-8") as fout:
        for entry in data:
            q_id = entry["id"]
            retrieved_lists = entry.get(triple_field, [])
            
            if not retrieved_lists:
                continue
            
            # Flatten triple lists
            all_triples_for_this_question = []
            for triple_chunk in retrieved_lists:
                if (isinstance(triple_chunk, list)
                    and len(triple_chunk) > 0
                    and isinstance(triple_chunk[0], list)):
                    all_triples_for_this_question.extend(triple_chunk)
                else:
                    all_triples_for_this_question.extend(retrieved_lists)
                    break
            
            # Write each triple as a document
            for triple in all_triples_for_this_question:
                triple_str = " ".join(str(x) for x in triple)
                doc_id_str = str(doc_id)
                fout.write(f"{doc_id_str}\t{triple_str}\n")
                
                doc_id_to_triple[doc_id_str] = triple
                doc_id_to_question_id[doc_id_str] = q_id
                doc_id += 1
    
    return doc_id_to_triple, doc_id_to_question_id


def rank_triples(
    data: List[Dict[str, Any]],
    collection_path: str,
    index_path: str,
    top_k: int = 200,
    experiment_name: str = "colbert_ranking",
    root_dir: str = "./experiments",
    checkpoint: str = "colbertv2.0",
    nbits: int = 2,
    overwrite_index: bool = False,
    triple_field: str = "retrieved_triples",
    output_field: str = "retrived_triples_ranked"
) -> List[Dict[str, Any]]:
    """
    Rank triples using ColBERT v2.
    
    Args:
        data: List of data entries with questions and triples
        collection_path: Path to TSV collection file
        index_path: Path to ColBERT index (will be created if doesn't exist)
        top_k: Number of top triples to retrieve per question
        experiment_name: ColBERT experiment name
        root_dir: Root directory for ColBERT experiments
        checkpoint: ColBERT checkpoint name
        nbits: Number of bits for quantization
        overwrite_index: Whether to overwrite existing index
        triple_field: Field name containing retrieved triples
        output_field: Field name for ranked triples output
        
    Returns:
        Data with ranked triples added
    """
    # Build collection TSV
    doc_id_to_triple, doc_id_to_question_id = build_collection_tsv(
        data, collection_path, triple_field
    )
    
    # ColBERT configuration
    config = ColBERTConfig(
        nbits=nbits,
        root=root_dir,
    )
    
    # Index collection
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        indexer = Indexer(
            checkpoint=checkpoint,
            config=config
        )
        
        index_name = os.path.basename(index_path)
        overwrite = "force_silent_overwrite" if overwrite_index else False
        
        indexer.index(
            name=index_name,
            collection=collection_path,
            overwrite=overwrite
        )
        
        # Create searcher
        full_index_path = os.path.join(config.root, experiment_name, "indexes", index_name)
        searcher = Searcher(
            index=full_index_path,
            collection=collection_path,
            config=config
        )
        
        # Rank triples for each question
        for entry in data:
            q_text = entry["question"]
            q_id = entry["id"]
            
            # Search with ColBERT
            doc_ids, ranks, scores = searcher.search(q_text, k=top_k)
            
            # Filter to keep only triples for this question
            relevant_results = []
            for (did_int, rank_int, score_float) in zip(doc_ids, ranks, scores):
                did_str = str(did_int)
                if doc_id_to_question_id.get(did_str) == q_id:
                    relevant_results.append((did_str, rank_int, score_float))
            
            # Sort by score descending
            relevant_results.sort(key=lambda x: x[2], reverse=True)
            
            # Build ranked triples list
            ranked_triples = []
            for final_rank, (did, original_rank, score) in enumerate(relevant_results, start=1):
                triple = doc_id_to_triple[did]
                ranked_triples.append({
                    "triple": triple,
                    "score": float(score),
                    "original_colbert_rank": int(original_rank),
                    "final_rank": int(final_rank)
                })
            
            # Attach ranked triples to entry
            entry[output_field] = ranked_triples
    
    return data
