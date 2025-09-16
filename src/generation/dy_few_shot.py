import os
import re
import json
import argparse
import hashlib
from typing import List, Dict, Tuple, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DEC_CHECKS = [
    ("FILTER",   r"\bFILTER\b"),
    ("OPTIONAL", r"\bOPTIONAL\b"),
    ("UNION",    r"\bUNION\b"),
    ("ORDER BY", r"\bORDER\s+BY\b"),
    ("LIMIT",    r"\bLIMIT\b"),
    ("GROUP BY", r"\bGROUP\s+BY\b"),
    ("COUNT",    r"\bCOUNT\s*\("),
    ("DISTINCT", r"\bSELECT\s+DISTINCT\b"),
    ("OFFSET",   r"\bOFFSET\b"),
    ("HAVING",   r"\bHAVING\b"),
]

def extract_decorators(sparql: str) -> List[str]:
    return sorted(label for label, pat in DEC_CHECKS if re.search(pat, sparql, re.IGNORECASE))


TRIPLE_BLOCK_RE = re.compile(r"\{(.*)\}", re.DOTALL)

def extract_triples(sparql: str) -> List[Tuple[str, str, str]]:
    m = TRIPLE_BLOCK_RE.search(sparql)
    if not m:
        return []
    body = m.group(1)
    triples = []
    for line in body.split("."):
        toks = line.strip().split()
        if len(toks) >= 3:
            s, p, o = toks[0], toks[1], toks[2]
            triples.append((s, p, o))
    return triples

def infer_motif(triples: List[Tuple[str, str, str]]) -> str:
    if not triples:
        return "composite"
    if len(triples) == 1:
        return "atomic"
    subs = [s for s, _, _ in triples]
    if len(set(subs)) == 1:
        return "star"

    links = 0
    for s1, _, o1 in triples:
        for s2, _, _ in triples:
            if o1 == s2:
                links += 1
    if links >= len(triples):
        return "cycle"
    if links >= len(triples) - 1:
        return "chain"
    return "composite"



def build_faiss_index(questions: List[str], model_name: str = "intfloat/e5-large"):
    model = SentenceTransformer(model_name)
    q_inputs = [f"query: {q}" for q in questions]
    embeddings = model.encode(q_inputs, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return model, index

def semantic_topk(
    input_question: str,
    model: SentenceTransformer,
    index: faiss.Index,
    train_questions: List[str],
    top_k: int = 200
):
    query_vec = model.encode([f"query: {input_question}"], normalize_embeddings=True, show_progress_bar=False)
    query_vec = query_vec.astype(np.float32)
    k = min(top_k, len(train_questions))
    scores, idxs = index.search(query_vec, k)
    return {train_questions[i]: float(s) for i, s in zip(idxs[0], scores[0])}



def structural_score(query_sparql: str, cand_sparql: str) -> float:
    q_decs = set(extract_decorators(query_sparql))
    c_decs = set(extract_decorators(cand_sparql))

    q_tr = extract_triples(query_sparql)
    c_tr = extract_triples(cand_sparql)

    q_motif = infer_motif(q_tr)
    c_motif = infer_motif(c_tr)

    motif_match = 1.0 if q_motif == c_motif else 0.0
    union = len(q_decs | c_decs)
    jacc = (len(q_decs & c_decs) / union) if union else 1.0
    return motif_match + jacc



def _normalize(d: Dict[Any, float]) -> Dict[Any, float]:
    if not d:
        return d
    m = max(d.values())
    if m <= 0:
        return {k: 0.0 for k in d}
    return {k: v / m for k, v in d.items()}

def _prepare_train_semantic_index(
    trainset: List[Dict[str, Any]],
    model_name: str = "intfloat/e5-large"
):
    train_questions = [d["question"] for d in trainset]
    model, index = build_faiss_index(train_questions, model_name=model_name)
    return model, index, train_questions

def _top10_triples_from_ranked(train_item: Dict[str, Any]) -> List[List[str]]:
    # ranked = train_item.get("retrived_triples_ranked") or []
    ranked = train_item.get("retrieved_triples_ranked") or train_item.get("retrived_triples_ranked") or [] ##For Vquanda
    if not isinstance(ranked, list):
        return []

    def _key(x: Dict[str, Any]):
        fr = x.get("final_rank", float("inf"))
        sc = x.get("score", float("-inf"))
        return (
            fr if isinstance(fr, (int, float)) else float("inf"),
            -(sc if isinstance(sc, (int, float)) else float("-inf")),
        )

    ranked_sorted = sorted(ranked, key=_key)

    out: List[List[str]] = []
    for entry in ranked_sorted[:10]:
        tri = entry.get("triple")
        if isinstance(tri, list) and len(tri) >= 3:
            out.append([str(tri[0]), str(tri[1]), str(tri[2])])
    return out

def _select_topk_for_pair(
    input_question: str,
    input_sparql: str,
    trainset: List[Dict[str, Any]],
    model: SentenceTransformer,
    faiss_index: faiss.Index,
    train_questions: List[str],
    top_k: int = 5,
    lambda_semantic: float = 0.4
) -> List[Dict[str, str]]:
    sem_map = semantic_topk(input_question, model, faiss_index, train_questions,
                            top_k=min(200, len(train_questions)))
    sem_map = _normalize(sem_map)

    raw_struct: Dict[str, float] = {}
    for d in trainset:
        cand_sparql = (d.get("formated_query") or d.get("sparql") or "").strip()
        raw_struct[str(d.get("id", ""))] = structural_score(input_sparql, cand_sparql)
    struct_map = _normalize(raw_struct)

    iq_norm = input_question.strip().lower()
    is_norm = input_sparql.strip()

    scored: List[Tuple[Dict[str, Any], float]] = []
    for d in trainset:
        q = d["question"]
        s = (d.get("formated_query") or d.get("sparql") or "").strip()

        if q.strip().lower() == iq_norm and s == is_norm:
            continue

        sem = sem_map.get(q, 0.0)                  
        struct = struct_map.get(str(d.get("id", "")), 0.0)
        combined = lambda_semantic * sem + (1.0 - lambda_semantic) * struct
        scored.append((d, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    out_pairs: List[Dict[str, Any]] = []
    for ex, _ in top:
        out_pairs.append({
            "question": ex["question"],
            "sparql": (ex.get("formated_query") or ex.get("sparql") or "").strip(),
            "retrieved_triples_top10": _top10_triples_from_ranked(ex),
            "triples": ex.get("triples", [])
        })
    return out_pairs

def attach_dynamic_pairs_to_dataset(
    dataset: List[Dict[str, Any]],
    trainset: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    lambda_semantic: float = 0.4,
    model_name: str = "intfloat/e5-large"
) -> List[Dict[str, Any]]:
    model, faiss_index, train_questions = _prepare_train_semantic_index(
        trainset, model_name=model_name
    )

    for item in dataset:
        if "retrieved_triples" in item:
            del item["retrieved_triples"]
            # del item["retrieved_ranked_triples"]
            # del item["retrieved_ranked_triples_bm25"]
            # del item["ranked_entities"]

        q = (item.get("question") or "").strip()
        s = (item.get("pred_query") or "").strip()

        if not q or not s:
            item["dynamic_pairs"] = []
            continue

        pairs = _select_topk_for_pair(
            input_question=q,
            input_sparql=s,
            trainset=trainset,
            model=model,
            faiss_index=faiss_index,
            train_questions=train_questions,
            top_k=top_k,
            lambda_semantic=lambda_semantic
        )
        item["dynamic_pairs"] = pairs

    return dataset



def main():
    ap = argparse.ArgumentParser(description="Attach dynamic few-shot pairs to a dataset.")
    ap.add_argument("--top_k", type=int, default=3, help="Number of few-shot pairs to attach.")
    ap.add_argument("--lambda_semantic", type=float, default=0.0, help="Weight for semantic similarity (0..1).")
    ap.add_argument("--model_name", type=str, default="intfloat/e5-large", help="SentenceTransformer model name.")
    args = ap.parse_args()

    test_set = "/home/m2khoda/dual_retriever/evaluations/end_to_end_evalution/vquanda/vquanda_test_top_10_mixtral_infered_second_time_plus_gold.json"
    train_set = "/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_train_ranked.json"
    output_path = "/home/m2khoda/dual_retriever/evaluations/dycot/vquanda_results/mixtral/vquanda_test_mixtral_top_10_plus_lambda_sem_0.0_pairs.json"

    with open(test_set, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(train_set, "r", encoding="utf-8") as f:
        trainset = json.load(f)

    augmented = attach_dynamic_pairs_to_dataset(
        dataset=dataset,
        trainset=trainset,
        top_k=args.top_k,
        lambda_semantic=args.lambda_semantic,
        model_name=args.model_name
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    print(f"Wrote augmented dataset with 'dynamic_pairs' to {output_path}")

if __name__ == "__main__":
    main()