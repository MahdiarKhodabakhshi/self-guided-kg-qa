import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "64"
os.environ["TORCH_NUM_THREADS"] = "64"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "2"

import torch
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "96")))  # DO NOT set interop here.

import json
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

# ---------------- Configuration ----------------
INPUT_JSON      = "/home/m2khoda/dual_retriever/lcquad2_train_retrieved_triples_first_half_filtered_ready_to_rank_without_statements.json"
OUTPUT_JSON     = "/home/m2khoda/dual_retriever/datasets/lcquad_2/lcquad2_train_first_half_subsampled_400_ranked.json"
COLLECTION_TSV  = "/home/m2khoda/dual_retriever/qald_test_ranker_collection.tsv"
EXPERIMENT      = "lcquad2_experiment_triples_formated"
INDEX_NAME      = "lcquad2_triples.nbits=2"
ROOT_DIR        = "/home/m2khoda/dual_retriever/experiments"
CHECKPOINT      = "colbertv2.0"

NPROCS          = 8         # worker processes
PER_THREADS     = 7           # BLAS/FAISS threads per worker
TOP_K           = 100
DOC_MAXLEN      = None
BATCH_SIZE      = 32          # number of queries per worker task
OVERWRITE_INDEX = True       # set True to rebuild even if index exists

# ---------------- Helpers ----------------
def _clean_tok(x: Any) -> str:
    return str(x).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()

def as_text(equiv: List[Any]) -> str:
    s = " ".join(_clean_tok(x) for x in equiv)
    return " ".join(s.split())

def _equiv_key(equiv: List[Any]) -> Tuple[str, str, str]:
    # normalized 3-tuple key for global dedup
    t = tuple(_clean_tok(x) for x in equiv)
    if len(t) != 3:
        # pad or trim defensively; we only index well-formed triples
        t = (t + ("", "", ""))[:3]
    return t  # (s, p, o)

def build_global_collection(
    data: List[Dict[str, Any]],
    collection_tsv: str
):
    os.makedirs(os.path.dirname(collection_tsv), exist_ok=True)

    # Maps
    equiv2docid: Dict[Tuple[str, str, str], int] = {}
    docid2text: Dict[int, str] = {}
    docid2qids: Dict[int, set] = defaultdict(set)
    # For each doc_id, we may need different triples per question that referenced it
    docid2payload_by_qid: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    did = 0
    written = 0

    with open(collection_tsv, "w", encoding="utf-8") as fout:
        for entry in data:
            q_id = int(entry.get("id"))
            for item in entry.get("retrieved_triples", []):
                if not isinstance(item, dict):
                    continue
                if "equivalent" not in item or "triple" not in item:
                    continue

                equiv = item["equivalent"]
                triple = item["triple"]

                if not isinstance(equiv, (list, tuple)) or len(equiv) != 3:
                    continue

                key = _equiv_key(list(equiv))
                if key not in equiv2docid:
                    # New global document
                    equiv2docid[key] = did
                    text = as_text(list(equiv))
                    fout.write(f"{did}\t{text}\n")

                    docid2text[did] = text
                    did += 1
                    written += 1

                doc_id = equiv2docid[key]
                # Track which questions reference this doc
                docid2qids[doc_id].add(q_id)
                # Store the exact payload to reproduce the right triple/equivalent for that question
                # (in case the same equivalent text appears with different triples in different entries)
                docid2payload_by_qid[doc_id][q_id] = {
                    "triple": triple,
                    "equivalent": list(equiv),
                }

    return equiv2docid, docid2text, docid2qids, docid2payload_by_qid, did, written

def index_collection(overwrite: bool = False):
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT)):
        cfg_kwargs = dict(nbits=2, root=ROOT_DIR, amp=False)
        if DOC_MAXLEN is not None:
            cfg_kwargs["doc_maxlen"] = DOC_MAXLEN
        config = ColBERTConfig(**cfg_kwargs)

        indexer = Indexer(checkpoint=CHECKPOINT, config=config)
        indexer.index(
            name=INDEX_NAME,
            collection=COLLECTION_TSV,
            overwrite=("force_silent_overwrite" if overwrite else False),
        )
        index_path = os.path.join(config.root, EXPERIMENT, "indexes", INDEX_NAME)
    return index_path, config

# ---------------- Batched search (multiprocess workers) ----------------
_SEARCHER = None

def _init_worker(index_path: str, per_threads: int, cfg_kwargs: Dict[str, Any], collection_tsv: str):
    # Set env **before** imports in the worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(per_threads)
    os.environ["MKL_NUM_THREADS"] = str(per_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(per_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(per_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TORCH_NUM_THREADS"] = str(per_threads)
    os.environ["TORCH_NUM_INTEROP_THREADS"] = "2"

    import torch as _torch
    _torch.set_num_threads(per_threads)  # OK; don't set interop here.

    import faiss
    faiss.omp_set_num_threads(per_threads)

    from colbert.infra import ColBERTConfig
    from colbert import Searcher

    config = ColBERTConfig(**cfg_kwargs, amp=False)

    global _SEARCHER
    _SEARCHER = Searcher(index=index_path, collection=collection_tsv, config=config)

def _run_batch(tasks: List[Tuple[int, str]]) -> List[Tuple[int, List[int], List[int], List[float]]]:
    queries = [q for (_, q) in tasks]
    qids    = [int(qid) for (qid, _) in tasks]

    results: List[Tuple[int, List[int], List[int], List[float]]] = []

    try:
        # Try vectorized/batched call if available
        if hasattr(_SEARCHER, "search_all"):
            res = _SEARCHER.search_all(queries, k=TOP_K)
            # Expected shapes: either a tuple of lists or a list of tuples
            if isinstance(res, tuple) and len(res) == 3:
                all_doc_ids, all_ranks, all_scores = res
                for qid, dids, rnk, sc in zip(qids, all_doc_ids, all_ranks, all_scores):
                    results.append((qid, list(map(int, dids)), list(map(int, rnk)), list(map(float, sc))))
            else:
                # list of per-query tuples
                for qid, (dids, rnk, sc) in zip(qids, res):
                    results.append((qid, list(map(int, dids)), list(map(int, rnk)), list(map(float, sc))))
            return results
    except Exception:
        # Fall back to per-query loop
        pass

    # Fallback: per-query search
    for qid, q in zip(qids, queries):
        doc_ids, ranks, scores = _SEARCHER.search(q, k=TOP_K)
        results.append((qid, list(map(int, doc_ids)), list(map(int, ranks)), list(map(float, scores))))
    return results

# ---------------- Main ----------------
def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)[:400]

    # 1) Build a GLOBAL collection with dedup across the whole dataset
    equiv2docid, docid2text, docid2qids, docid2payload_by_qid, total_docs, written = build_global_collection(
        data, COLLECTION_TSV
    )
    print(f"[collection] wrote {written} unique docs (global dedup) to {COLLECTION_TSV}")

    # 2) Build or reuse the ColBERT index
    index_path, base_config = index_collection(overwrite=OVERWRITE_INDEX)
    print(f"[index] ready at {index_path}")

    # 3) Prepare batched tasks
    cfg_kwargs = dict(nbits=2, root=ROOT_DIR)
    if DOC_MAXLEN is not None:
        cfg_kwargs["doc_maxlen"] = DOC_MAXLEN

    tasks_all: List[Tuple[int, str]] = [(int(e["id"]), e["question"]) for e in data]
    # chunk into batches
    batches: List[List[Tuple[int, str]]] = [
        tasks_all[i:i + BATCH_SIZE] for i in range(0, len(tasks_all), BATCH_SIZE)
    ]

    # 4) Run batched search in parallel workers
    raw_by_qid: Dict[int, Tuple[List[int], List[int], List[float]]] = {}

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=NPROCS,
        initializer=_init_worker,
        initargs=(index_path, PER_THREADS, cfg_kwargs, COLLECTION_TSV),
    ) as pool:
        for batch_res in pool.imap_unordered(_run_batch, batches, chunksize=1):
            # batch_res is a list of (qid, doc_ids, ranks, scores)
            for qid, dids, ranks, scores in batch_res:
                raw_by_qid[int(qid)] = (dids, ranks, scores)

    # 5) Write per-entry ranked results (same structure as your reference)
    for entry in data:
        q_id = int(entry["id"])
        doc_ids, ranks, scores = raw_by_qid.get(q_id, ([], [], []))

        # Keep only docs that belong to THIS question (global dedup means each doc may belong to many qids)
        rel = []
        for did, rnk, sc in zip(doc_ids, ranks, scores):
            if q_id in docid2qids.get(did, ()):
                rel.append((did, rnk, sc))

        # Sort by ColBERT score desc
        rel.sort(key=lambda x: x[2], reverse=True)

        ranked = []
        for final_rank, (did, orig_rank, score) in enumerate(rel, 1):
            payload = docid2payload_by_qid[did][q_id]  # picks the exact (triple, equivalent) for this q
            ranked.append({
                "triple": payload["triple"],
                "equivalent": payload["equivalent"],
                "score": float(score),
                "original_colbert_rank": int(orig_rank),
                "final_rank": int(final_rank),
            })
        entry["retrieved_triples_ranked"] = ranked

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote ranked triples to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()