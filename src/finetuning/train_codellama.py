from typing import List, Any
import json, os, torch

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
torch.backends.cuda.matmul.allow_tf32 = True  # speedup on Ampere+

# Use Hugging Face model ID - will auto-download if not cached
MODEL_NAME = os.getenv("CODELLAMA_MODEL_PATH", "codellama/CodeLlama-34b-Instruct-hf")
USE_FAST_TOKENIZER = False
LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # <- exact modules
LOAD_IN_4BIT = True
PROMPTING_MODE = "chat_template"

TRIPLES_LIMIT = 10
MAX_INPUT_LEN = 4096
HEADROOM      = 512

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

def _fit_count(tokenizer, text, max_len=MAX_INPUT_LEN):
    return len(tokenizer(text, add_special_tokens=True)["input_ids"]) <= (max_len - HEADROOM)

def build_user_block(question: str, triples: List[Any]) -> str:
    triples_str = "\n".join(
        f"{i}. {' '.join(map(str, t)) if isinstance(t,(list,tuple)) else str(t)}"
        for i, t in enumerate(triples, 1)
    )
    return f"Question:\n{question}\n\nCandidate Triples (max 10, numbered):\n{triples_str}".strip()

def build_inst_prompt(sample: dict, tokenizer) -> str:
    question = (sample.get("question") or "").strip()
    hits = (sample.get("retrieved_triples_ranked")############ Change for the other datasets
            or sample.get("retrieved_triples") or [])[:TRIPLES_LIMIT]
    triples = []
    for h in hits:
        triples.append(h["triple"] if isinstance(h, dict) and "triple" in h else h)

    def try_fit(sys_text: str) -> str:
        kept = len(triples)
        while kept >= 0:
            user_content = build_user_block(question, triples[:kept])
            prompt = f"[INST] <<SYS>>{sys_text}<</SYS>>\n{user_content}\n[/INST]\n"
            if _fit_count(tokenizer, prompt):
                return prompt
            kept -= 1
        return f"[INST] <<SYS>>{sys_text}<</SYS>>\nQuestion:\n{question}\n[/INST]\n"

    p = try_fit(SYSTEM_PROMPT_WITH_DEMO)
    if not _fit_count(tokenizer, p):
        p = try_fit(SYSTEM_PROMPT_GENERIC)
    return p

def build_chat_template_text(sample: dict, tokenizer, assistant_text: str) -> str:
    question = (sample.get("question") or "").strip()
    hits = (sample.get("retrieved_triples_ranked") #####Change for other datasets
            or sample.get("retrieved_triples") or [])[:TRIPLES_LIMIT]
    triples = []
    for h in hits:
        triples.append(h["triple"] if isinstance(h, dict) and "triple" in h else h)

    def try_fit(sys_text: str) -> str:
        kept = len(triples)
        while kept >= 0:
            user_content = build_user_block(question, triples[:kept])
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user",   "content": user_content},
                {"role": "assistant", "content": assistant_text},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if _fit_count(tokenizer, text):
                return text
            kept -= 1
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user",   "content": f"Question:\n{question}"},
            {"role": "assistant", "content": assistant_text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    t = try_fit(SYSTEM_PROMPT_WITH_DEMO)
    if not _fit_count(tokenizer, t):
        t = try_fit(SYSTEM_PROMPT_GENERIC)
    return t

TRAIN_FILEPATH = "/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_train_ranked.json"

with open(TRAIN_FILEPATH, "r") as f:
    train_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=USE_FAST_TOKENIZER)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_full_texts(examples, tok) -> List[str]:
    out = []
    for ex in examples:
        gold_query = ex.get("formated_query") or ex.get("formatted_query") or ex.get("query") or ""
        completion = format_assistant_answer(gold_query)

        if PROMPTING_MODE == "chat_template":
            text = build_chat_template_text(ex, tok, assistant_text=completion)
        else:
            prompt = build_inst_prompt(ex, tok)
            text = prompt + completion

        out.append(text)
    return out

train_texts = build_full_texts(train_data, tokenizer)
train_ds = HFDataset.from_dict({"text": train_texts})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=False,
    use_cache=False,
    device_map="auto",  # Correct argument name: device_map
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
model.config.padding_side  = "right"
model.config.use_cache     = False

try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=LORA_TARGETS,
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

args = TrainingArguments(
    output_dir="codellama-34b-instruct-vquanda",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    save_total_limit=3,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    disable_tqdm=False,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True, 
    dataset_kwargs={"add_special_tokens": True},
    args=args,
)

if __name__ == "__main__":
    trainer.train()
    if trainer.accelerator.is_main_process:
        model.save_pretrained("codellama-34b-instruct-vquanda-finetuned")
        tokenizer.save_pretrained("codellama-34b-instruct-vquanda-finetuned")