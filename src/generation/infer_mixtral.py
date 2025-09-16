#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, re, argparse
from typing import List, Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ===== Tags & knobs (same as training) =====
SYSTEM_TAG  = "<|system|>\n"
USER_TAG    = "<|user|>\n"
ASSIST_TAG  = "<|assistant|>\n"
ASSIST_PREF = f"\n{ASSIST_TAG}"
TRIPLES_LIMIT = 10

# ===== Helpers duplicated from training =====
def lists_to_numbered_string(triples: List[Any]) -> str:
    return "\n".join(
        f"{i}. {' '.join(map(str, t)) if isinstance(t, (list, tuple)) else str(t)}"
        for i, t in enumerate(triples, 1)
    )

def format_assistant_answer(raw_query: str) -> str:
    one_line = " ".join((raw_query or "").strip().split())
    return "<Answer>\n" + json.dumps({"sparql": one_line}, ensure_ascii=False)

# Demo block (kept exactly)
demo_question = "Who developed Skype?"
demo_triples = [
    ["res:Skype", "dbo:developer", "res:Skype_Technologies"],
    ["res:21Vianet", "dbo:service", "res:Skype"],
    ["res:Skype", "gold:hypernym", "res:Application"],
    ["res:Skype", "dbp:operatingSystem", "res:HoloLens"],
    ["res:Skype", "dbo:operatingSystem", "res:HoloLens"],
    ["res:Skype", "dbp:operatingSystem", "res:IOS"],
    ["res:Skype", "dbo:operatingSystem", "res:IOS"],
    ["res:Skype", "dbp:operatingSystem", "res:IPadOS"],
    ["res:Skype", "dbo:operatingSystem", "res:IPadOS"],
    ["res:Skype", "dbp:license", "res:Proprietary_software"],
]
demo_triples_str = lists_to_numbered_string(demo_triples)
demo_sparql = (
    "PREFIX dbo: <http://dbpedia.org/ontology/> "
    "PREFIX res: <http://dbpedia.org/resource/> "
    "SELECT DISTINCT ?uri WHERE { res:Skype dbo:developer ?uri }"
)
demo_answer = "<Answer>\n" + json.dumps({"sparql": demo_sparql}, ensure_ascii=False)

SYSTEM_PROMPT_WITH_DEMO = f"""Given a specific question and up to ten potentially relevant triples, generate the
corresponding SPARQL query for DBpedia. Return your answer after <Answer>, in JSON
with key "sparql" and the query as its string value.

Example INPUT (exactly what you will receive for every task)

Question:
{demo_question}

Candidate Triples (numbered, max 10):
{demo_triples_str}

Example OUTPUT (your response must follow **this exact shape**)

{demo_answer}""".strip()

SYSTEM_PROMPT_GENERIC = (
    "Given a specific question and up to ten potentially relevant triples, "
    "generate the corresponding SPARQL query for DBpedia. Return your answer after <Answer>, "
    'in JSON with key "sparql" and the query as its string value.'
)

def build_text_fitting(sample: dict, tokenizer, max_len=8192, headroom=1536) -> str:
    question = (sample.get("question") or "").strip()
    hits = sample.get("retrieved_triples_ranked") or sample.get("retrived_triples_ranked") or []
    hits = hits[:TRIPLES_LIMIT]

    triples = []
    for h in hits:
        if isinstance(h, dict) and "triple" in h:
            triples.append(h["triple"])
        else:
            triples.append(h)

    system_long  = f"{SYSTEM_TAG}{SYSTEM_PROMPT_WITH_DEMO}\n\n"
    system_short = f"{SYSTEM_TAG}{SYSTEM_PROMPT_GENERIC}\n\n"
    user_prefix  = f"{USER_TAG}Question:\n{question}\n\nCandidate Triples (max 10, numbered):\n"

    def try_fit(system_text: str) -> str:
        kept = len(triples)
        while kept >= 0:
            triples_str = "\n".join(
                f"{i}. {' '.join(map(str, t)) if isinstance(t,(list,tuple)) else str(t)}"
                for i, t in enumerate(triples[:kept], 1)
            )
            prefix = system_text + user_prefix + triples_str + ASSIST_PREF
            n_tokens = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
            if n_tokens <= (max_len - headroom):
                return prefix
            kept -= 1
        return system_text + user_prefix + "" + ASSIST_PREF

    prefix = try_fit(system_long)
    if len(tokenizer(prefix, add_special_tokens=True)["input_ids"]) > (max_len - headroom):
        prefix = try_fit(system_short)
    return prefix

def extract_json_after_answer(text: str) -> Dict[str, str]:
    if not text:
        return {"sparql": ""}
    idx = text.find("<Answer>")
    if idx == -1:
        return {"sparql": ""}

    after = text[idx + len("<Answer>"):]
    m = re.search(r"\{\s*\"sparql\"\s*:\s*.*?\}", after, flags=re.DOTALL)
    candidate = m.group(0) if m else (after[: after.find("}") + 1] if after.find("}") != -1 else "")
    try:
        obj = json.loads(candidate)
        obj["sparql"] = " ".join((obj.get("sparql") or "").strip().split())
        return {"sparql": obj["sparql"]}
    except Exception:
        lines = " ".join(after.strip().split())
        mm = re.search(r"\"sparql\"\s*:\s*\"(.*?)\"", lines)
        return {"sparql": mm.group(1) if mm else ""}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/Mixtral-8x7B-Instruct-v0.1"))
    ap.add_argument("--adapters", default=os.path.expanduser("~/dual_retriever/scripts/mixtral/mixtral-8x7b-instruct-vquanda-finetuned"))
    ap.add_argument("--test", default="/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_test_ranked.json")
    ap.add_argument("--out", default="/home/m2khoda/dual_retriever/evaluations/end_to_end_evalution/vquanda/vquanda_test_top_10_mixtral_infered_second_time.jsonl")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_input_len", type=int, default=8192)
    ap.add_argument("--headroom", type=int, default=1536)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    args = ap.parse_args()

    # Device pin (like your Mistral script)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # BnB 4-bit (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model (4-bit) + LoRA adapters
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=True,   # enable for inference
        trust_remote_code=False,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id
    base.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    base.config.padding_side = 'right'

    model = PeftModel.from_pretrained(base, args.adapters)
    model.eval()

    # Load test JSON (list of dicts)
    with open(args.test, "r") as f:
        test_data = json.load(f)

    # Output (JSONL)
    out_path = args.out
    fw = open(out_path, "w", encoding="utf-8")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    for i, ex in enumerate(test_data):
        prefix = build_text_fitting(ex, tokenizer, max_len=args.max_input_len, headroom=args.headroom)
        input_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, **gen_kwargs)

        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        pred_obj = extract_json_after_answer(full_text)

        rec = {
            "id": ex.get("id", i),
            "question": ex.get("question", ""),
            "pred_text": full_text,   # keep for debugging
            "pred_answer": pred_obj,  # {"sparql": "..."}
        }
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(test_data)}]")

    fw.close()
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    import argparse
    main()