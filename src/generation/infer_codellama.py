import os, sys, json, re, argparse
from typing import List, Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

TRIPLES_LIMIT = 10
MAX_INPUT_LEN_DEFAULT = 2048
HEADROOM_DEFAULT       = 512

def lists_to_numbered_string(triples: List[Any]) -> str:
    return "\n".join(
        f"{i}. {' '.join(map(str, t)) if isinstance(t, (list, tuple)) else str(t)}"
        for i, t in enumerate(triples, 1)
    )

def format_assistant_answer(raw_query: str) -> str:
    one_line = " ".join((raw_query or "").strip().split())
    return "<Answer>\n" + json.dumps({"sparql": one_line}, ensure_ascii=False)

def extract_json_after_answer(text: str) -> Dict[str, str]:
    if not text:
        return {"sparql": ""}
    idx = text.find("<Answer>")
    if idx == -1:
        return {"sparql": ""}
    after = text[idx + len("<Answer>") :]
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

SYSTEM_PROMPT_WITH_DEMO = f"""You are a helpful SPARQL generator.

Given a specific question and up to ten potentially relevant triples, generate the
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
    "You are a helpful SPARQL generator.\n"
    "Given a specific question and up to ten potentially relevant triples, "
    "generate the corresponding SPARQL query for DBpedia. Return your answer after <Answer>, "
    'in JSON with key "sparql" and the query as its string value.'
)

def _fits(tokenizer, text: str, max_len: int, headroom: int) -> bool:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"]) <= (max_len - headroom)

def _collect_triples(ex: dict) -> List[Any]:
    hits = ex.get("retrieved_triples_ranked") or ex.get("retrieved_triples_ranked") or []
    hits = hits[:TRIPLES_LIMIT]
    triples = []
    for h in hits:
        triples.append(h["triple"] if isinstance(h, dict) and "triple" in h else h)
    return triples

def _user_block(question: str, triples: List[Any]) -> str:
    triples_str = "\n".join(
        f"{i}. {' '.join(map(str, t)) if isinstance(t,(list,tuple)) else str(t)}"
        for i, t in enumerate(triples, 1)
    )
    return f"Question:\n{question}\n\nCandidate Triples (max 10, numbered):\n{triples_str}".strip()

def build_prompt_chat_template(ex: dict, tokenizer, max_len: int, headroom: int) -> str:
    question = (ex.get("question") or "").strip()
    triples = _collect_triples(ex)

    def try_fit(sys_text: str) -> str:
        kept = len(triples)
        while kept >= 0:
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user",   "content": _user_block(question, triples[:kept])},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if _fits(tokenizer, prompt, max_len, headroom):
                return prompt
            kept -= 1
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user",   "content": f"Question:\n{question}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    p = try_fit(SYSTEM_PROMPT_WITH_DEMO)
    if not _fits(tokenizer, p, max_len, headroom):
        p = try_fit(SYSTEM_PROMPT_GENERIC)
    return p

def build_prompt_inst(ex: dict, tokenizer, max_len: int, headroom: int) -> str:
    question = (ex.get("question") or "").strip()
    triples = _collect_triples(ex)

    def try_fit(sys_text: str) -> str:
        kept = len(triples)
        while kept >= 0:
            user_content = _user_block(question, triples[:kept])
            prompt = f"[INST] <<SYS>>{sys_text}<</SYS>>\n{user_content}\n[/INST]\n"
            if _fits(tokenizer, prompt, max_len, headroom):
                return prompt
            kept -= 1
        return f"[INST] <<SYS>>{sys_text}<</SYS>>\nQuestion:\n{question}\n[/INST]\n"

    p = try_fit(SYSTEM_PROMPT_WITH_DEMO)
    if not _fits(tokenizer, p, max_len, headroom):
        p = try_fit(SYSTEM_PROMPT_GENERIC)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/models/CodeLlama-34b-Instruct-hf"))
    ap.add_argument("--adapters", default=os.path.expanduser("codellama-34b-instruct-vquanda-finetuned"))
    ap.add_argument("--test", required=True, help="Path to test JSON (list of {question, retrieved_triples...})")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_input_len", type=int, default=MAX_INPUT_LEN_DEFAULT)
    ap.add_argument("--headroom", type=int, default=HEADROOM_DEFAULT)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--prompting_mode", choices=["chat_template","inst"], default="chat_template")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cuda.matmul.allow_tf32 = True

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model + LoRA adapters
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=True,              # enable cache for inference
        trust_remote_code=False,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id
    base.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    base.config.padding_side = "right"

    model = PeftModel.from_pretrained(base, args.adapters)
    model.eval()

    # Load test set
    with open(args.test, "r") as f:
        test_data = json.load(f)

    # Output writer
    fw = open(args.out, "w", encoding="utf-8")

    # Greedy / deterministic decoding (you trained with strict format)
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
        if args.prompting_mode == "chat_template":
            prompt = build_prompt_chat_template(ex, tokenizer, args.max_input_len, args.headroom)
        else:
            prompt = build_prompt_inst(ex, tokenizer, args.max_input_len, args.headroom)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        pred_obj = extract_json_after_answer(full_text)

        rec = {
            "id": ex.get("id", i),
            "question": ex.get("question", ""),
            "pred_text": full_text,
            "pred_answer": pred_obj,  # {"sparql": "..."}
        }
        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(test_data)}]")

    fw.close()
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()