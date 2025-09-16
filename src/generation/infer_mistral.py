import os, sys, json, re, argparse
from typing import List, Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# Constants & prompt bits
# =========================
SYSTEM_TAG = "<|system|>\n"
USER_TAG = "<|user|>\n"
ASSIST_TAG = "<|assistant|>\n"
ASSIST_PREF = f"\n{ASSIST_TAG}"
TRIPLES_LIMIT = 10

def lists_to_numbered_string(triples: List[Any]) -> str:
    return "\n".join(
        f"{i}. {' '.join(map(str, t)) if isinstance(t, (list, tuple)) else str(t)}"
        for i, t in enumerate(triples, 1)
    )

def format_assistant_answer(raw_query: str) -> str:
    one_line = " ".join((raw_query or "").strip().split())
    return "<Answer>\n" + json.dumps({"sparql": one_line}, ensure_ascii=False)

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

def build_text_fitting(sample: dict, tokenizer, max_len=2048, headroom=256) -> str:
    question = (sample.get("question") or "").strip()

    hits = sample.get("retrieved_triples_ranked")
    if not hits:
        hits = sample.get("retrived_triples") or []
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
    """
    Find the first occurrence of <Answer> and parse the following JSON object.
    Returns {"sparql": "..."} with the query in one line. Robust to extra whitespace.
    """
    if not text:
        return {"sparql": ""}
    idx = text.find("<Answer>")
    if idx == -1:
        return {"sparql": ""}

    after = text[idx + len("<Answer>"):]
    m = re.search(r"\{\s*\"sparql\"\s*:\s*.*?\}", after, flags=re.DOTALL)
    if not m:
        first_close = after.find("}")
        candidate = after[: first_close + 1] if first_close != -1 else ""
    else:
        candidate = m.group(0)

    try:
        obj = json.loads(candidate)
        obj["sparql"] = " ".join((obj.get("sparql") or "").strip().split())
        return {"sparql": obj["sparql"]}
    except Exception:
        lines = " ".join(after.strip().split())
        mm = re.search(r"\"sparql\"\s*:\s*\"(.*?)\"", lines)
        return {"sparql": mm.group(1) if mm else ""}

def first_cuda_device(m) -> torch.device:
    """
    For sharded (model-parallel) models, find the first CUDA device in hf_device_map.
    Falls back to CPU if none are present.
    """
    dm = getattr(m, "hf_device_map", None) or {}
    # dm is a dict: module_name -> device string
    for _, dev in sorted(dm.items()):
        if isinstance(dev, str) and dev.startswith("cuda"):
            return torch.device(dev)
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--adapters", default="/home/m2khoda/dual_retriever/scripts/mistral/mistral-7b-v0.1-vquanda-5ephoc/checkpoint-20000")
    ap.add_argument("--test", default="/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_test_ranked.json")
    ap.add_argument("--out", default="/home/m2khoda/dual_retriever/evaluations/end_to_end_evalution/vquanda/vquanda_test_top_10_mistral_infered.jsonl")
    # Accept multiple GPUs, e.g., "0,1"
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--max_input_len", type=int, default=4096)
    ap.add_argument("--headroom", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    # Per-GPU VRAM cap (GiB) — you asked for 4 GiB on each of two GPUs
    ap.add_argument("--per_gpu_mem_gib", type=int, default=15)
    # Optional: CPU offload memory allowance
    ap.add_argument("--cpu_mem_gib", type=int, default=48)
    args = ap.parse_args()

    # Make the requested GPUs visible (e.g., "0,1")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Memory allocator tuning for large-gen workloads
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build max_memory map for the *visible* GPUs (they will be renumbered 0..N-1)
    visible_gpu_ids = [g.strip() for g in args.gpu.split(",") if g.strip() != ""]
    num_visible = len(visible_gpu_ids)
    per_gpu = f"{args.per_gpu_mem_gib}GiB"
    max_memory = {i: per_gpu for i in range(num_visible)}
    max_memory["cpu"] = f"{args.cpu_mem_gib}GiB"

    # Where to offload if VRAM is exceeded
    offload_dir = "./offload"
    os.makedirs(offload_dir, exist_ok=True)

    # 4-bit quantized base model, sharded across visible GPUs with per-GPU cap
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",             # model-parallel sharding
        max_memory=max_memory,         # cap ~4 GiB per GPU
        offload_folder=offload_dir,    # spillover to CPU if needed
        low_cpu_mem_usage=True,
        use_cache=True,                # good for inference
    )

    # PEFT adapters (keeps the same device map)
    model = PeftModel.from_pretrained(base, args.adapters)
    model.eval()

    # Optional: show the device map for sanity
    dm = getattr(model, "hf_device_map", None)
    if dm:
        print("hf_device_map:", dm)

    # Load test data
    with open(args.test, "r") as f:
        test_data = json.load(f)

    # Output file
    out_path = args.out
    fw = open(out_path, "w", encoding="utf-8")

    # Generation config
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    main_dev = first_cuda_device(model)

    # Run
    for i, ex in enumerate(test_data):
        prefix = build_text_fitting(ex, tokenizer, max_len=args.max_input_len, headroom=args.headroom)
        input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(main_dev)

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, **gen_kwargs)

        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        pred_obj = extract_json_after_answer(full_text)

        rec = {
            "id": ex.get("id", i),
            "question": ex.get("question", ""),
            "pred_text": full_text,  # keep for debugging
            "pred_answer": pred_obj, # {"sparql": "..."}
        }
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(test_data)}]")

    fw.close()
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()