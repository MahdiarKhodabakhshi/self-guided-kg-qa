from typing import List, Any
import json, os, sys, torch

from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

torch.cuda.empty_cache()

SYSTEM_TAG  = "<|system|>\n"
USER_TAG    = "<|user|>\n"
ASSIST_TAG  = "<|assistant|>\n"
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

# ------------------------
# Build the same prefix that fits tokenizer context
# ------------------------
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

# ------------------------
# Paths & model
# ------------------------
HOME = os.path.expanduser("~")
TRAIN_FILEPATH = "/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_train_ranked.json"

LOCAL_MODEL_DIR = os.path.join(HOME, "models", "Mixtral-8x7B-Instruct-v0.1")
MODEL_NAME = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Accelerate: per-rank device map under DDP; else "auto"
world_size = int(os.environ.get("WORLD_SIZE", "1"))
if world_size > 1:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank}
else:
    device_map = "auto"

# ------------------------
# Load data & build full texts (prefix + completion) EXACTLY like before
# ------------------------
with open(TRAIN_FILEPATH, "r") as f:
    train_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_full_texts(examples, tok) -> List[str]:
    out = []
    for ex in examples:
        prefix = build_text_fitting(ex, tok, max_len=8192, headroom=1536)
        gold_query = ex.get("formated_query") or ex.get("formatted_query") or ex.get("query") or ""
        completion = format_assistant_answer(gold_query)
        out.append(prefix + completion)   # SAME: system+user+assistant-prefix + gold <Answer>{...}
    return out

train_texts = build_full_texts(train_data, tokenizer)

# For TRL>=0.8, pass raw text via HuggingFace Dataset and tell SFTTrainer which field to read
train_ds = HFDataset.from_dict({"text": train_texts})

# ------------------------
# QLoRA 4-bit
# ------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=False,
    use_cache=False,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
model.config.padding_side  = "right"
model.config.use_cache     = False
model.gradient_checkpointing_enable()

# ------------------------
# LoRA targets for Mixtral MoE (attention proj + w1/w2/w3 experts)
# ------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj","k_proj","v_proj","o_proj","w1","w2","w3"],
    bias="none",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

def print_trainable_parameters(m):
    trainable_params = 0
    all_param = 0
    for _, p in m.named_parameters():
        all_param += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    pct = 100 * trainable_params / all_param if all_param else 0
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct:.4f}")

if int(os.environ.get("RANK", "0")) == 0:
    print_trainable_parameters(model)

# ------------------------
# TrainingArguments (same spirit as yours; LR tuned for LoRA)
# ------------------------
args = TrainingArguments(
    output_dir="mixtral-8x7b-instruct-qlora-same-method",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,           # your 2e-6 was very conservative; 1e-4..2e-4 typical for LoRA
    save_total_limit=3,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    disable_tqdm=False,
    ddp_find_unused_parameters=False,
)

# ------------------------
# TRL SFTTrainer — give it raw text + which field to read
# Keep packing=True so it behaves similarly to your previous "packed" approach.
# ------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
    # ensure tokenizer adds specials (we didn't manually add BOS/EOS)
    dataset_kwargs={"add_special_tokens": True},
    args=args,
)

if __name__ == "__main__":
    trainer.train()
    if trainer.accelerator.is_main_process:
        model.save_pretrained("mixtral-8x7b-instruct-vquanda-finetuned")
        tokenizer.save_pretrained("mixtral-8x7b-instruct-vquanda-finetuned")