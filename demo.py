import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

raw_ds = load_dataset("csv", data_files="news_headline_generator.csv")["train"]

model_name = "distilgpt2"
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token  

MAX_ART_LEN = 64
MAX_HEAD_LEN = 15
SEP_TOKEN_ID = tok.eos_token_id

def preprocess(example):
    art_ids = tok(
        example["content_text"],
        truncation=True,
        max_length=MAX_ART_LEN
    )["input_ids"]
    head_ids = tok(
        example["generated_headline"],
        truncation=True,
        max_length=MAX_HEAD_LEN
    )["input_ids"]

    concatenated = art_ids + [SEP_TOKEN_ID] + head_ids + [SEP_TOKEN_ID]

    labels = [-100] * (len(art_ids) + 1) + head_ids + [SEP_TOKEN_ID]

    attention_mask = [1] * len(concatenated)

    return {
        "input_ids": concatenated,
        "attention_mask": attention_mask,
        "labels": labels
    }

ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names, desc="Tokenizing")
ds = ds.train_test_split(test_size=0.02, seed=42)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
base_model = prepare_model_for_kbit_training(base_model)

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./headline-lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_steps=200,
    logging_steps=50,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=2,
    fp16=True,
    report_to="none",
    push_to_hub=False
)

collator = default_data_collator

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=collator
)

trainer.train()

adapter_save_path = "./headline-lora/adapters"
tokenizer_save_path = "./headline-lora/tokenizer"

os.makedirs(adapter_save_path, exist_ok=True)
os.makedirs(tokenizer_save_path, exist_ok=True)

model.save_pretrained(adapter_save_path)
tok.save_pretrained(tokenizer_save_path)

def generate_headline(article: str, temperature: float = 0.8, top_p: float = 0.95) -> str:
    inputs = tok(
        article,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ART_LEN
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_HEAD_LEN + 1,  # 生成标题 + 结束符
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    
    gen_ids = out[0, inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("Torch using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    test_article = "The central bank raised interest rates for the third time this year, citing persistent inflation."
    print("Generated headline:", generate_headline(test_article))
