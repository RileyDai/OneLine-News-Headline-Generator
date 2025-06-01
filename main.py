import os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    Trainer,
    BitsAndBytesConfig  
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. Load dataset ------------------------------------------------------
raw_ds = load_dataset("csv", data_files="news_headline_generator.csv")["train"]

# --- 2. Preprocessing -----------------------------------------------------
model_name = "distilgpt2"
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token  # GPT-2 has no native PAD

MAX_ART_LEN, MAX_HEAD_LEN = 64, 15

def preprocess(example):
    art = example["content_text"]
    head = example["generated_headline"]
    ex = tok(art, truncation=True, max_length=MAX_ART_LEN)
    lbl = tok(head, truncation=True, max_length=MAX_HEAD_LEN)
    ex["labels"] = lbl["input_ids"]
    return ex

ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names, desc="Tokenizing")
ds = ds.train_test_split(test_size=0.02, seed=42)

# --- 3. Load model in 8-bit with LoRA -------------------------------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

# --- 4. Training setup ----------------------------------------------------
args = TrainingArguments(
    output_dir="./headline-lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_steps=200,
    logging_steps=50,
    eval_strategy="epoch",
    gradient_accumulation_steps=2,
    fp16=True, 
    report_to="none",  
    push_to_hub=False
)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=collator
)

trainer.train()

# --- 5. Save model and tokenizer ------------------------------------------
model.save_pretrained("./headline-lora/adapters")
tok.save_pretrained("./headline-lora/tokenizer")

# --- 6. Inference helper --------------------------------------------------
def generate_headline(article, temperature=0.8, top_p=0.95):
    inputs = tok(article, return_tensors="pt", truncation=True, max_length=MAX_ART_LEN).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_HEAD_LEN,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    return tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# --- 7. Test example ------------------------------------------------------
if __name__ == "__main__":
    print("Torch using GPU:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0))
    test_article = "The central bank raised interest rates for the third time this year, citing persistent inflation."
    print("Generated headline:", generate_headline(test_article))
