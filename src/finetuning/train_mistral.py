from typing import List, Any
import json, os, sys, torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

torch.cuda.empty_cache()

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

def build_text_fitting(sample: dict, tokenizer, max_len=8192, headroom=1536) -> str:
    question = (sample.get("question") or "").strip()

    hits = sample.get("retrived_triples_ranked")
    if not hits:
        hits = sample.get("retrieved_triples") or []
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

train_filepath = '/home/m2khoda/dual_retriever/datasets/vquanda/vquanda_train_ranked.json'
with open(train_filepath, 'r') as f:
    train_data = json.load(f)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

selected_gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_full_texts(examples, tok) -> List[str]:
    out = []
    for ex in examples:
        # Build the System+User+Assistant prefix that fits
        prefix = build_text_fitting(ex, tok, max_len=8192, headroom=1536)
        # Choose the gold query field robustly
        gold_query = ex.get("formated_query") or ex.get("formatted_query") or ex.get("query") or ""
        completion = format_assistant_answer(gold_query)
        full_text = prefix + completion  # SFT target is the full conversation including the answer
        out.append(full_text)
    return out

train_texts = build_full_texts(train_data, tokenizer)

class TextAsTargetDataset(Dataset):
    def __init__(self, tokenizer, texts: List[str], max_length=2000):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': enc['input_ids'].flatten(),
        }

dataset = TextAsTargetDataset(tokenizer, train_texts)

for i in range(min(407, len(dataset))):
    token_count = len(dataset[i]['input_ids'])
    if token_count > 2000:
        print(f"Datapoint {i} exceeds max token length with {token_count} tokens.")

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto"
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pretraining_tp = 1
model.config.padding_side = 'right'
model.config.use_cache = False
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=64,       
    lora_alpha=128,   
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj", 
    ],
    bias="none",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, p in model.named_parameters():
        all_param += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    pct = 100 * trainable_params / all_param if all_param else 0
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct}")

print_trainable_parameters(model)

args = TrainingArguments(
    output_dir="mistral-7b-v0.1-vquanda-5ephoc",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-6,
    save_total_limit=5,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    disable_tqdm=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=4096,
    tokenizer=tokenizer,
    packing=True,
    args=args
)


trainer.train()