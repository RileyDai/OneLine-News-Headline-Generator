import os
import random
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from accelerate import Accelerator

MAX_CTX_LEN = 1024        
MAX_SUM_LEN = 25         

def train_model():
    accelerator = Accelerator(mixed_precision="fp16")   
    device = accelerator.device

    raw_train = load_dataset("EdinburghNLP/xsum", split="train")
    raw_test = load_dataset("EdinburghNLP/xsum", split="test")


    model_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token 


    def keep_short_summary(example):
        summary_ids = tok(example["summary"], truncation=True).input_ids
        return len(summary_ids) <= MAX_SUM_LEN

    train_ds = raw_train.filter(keep_short_summary)  
    test_ds = raw_test.filter(keep_short_summary)  

    def preprocess(example):
        doc   = example["document"]
        summary = example["summary"]
        full_input = doc + tok.eos_token + summary + tok.eos_token
        tokens = tok(
            full_input,
            truncation=True,
            max_length=MAX_CTX_LEN,
            padding=False
        )
        return tokens

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, batched=False, num_proc=8)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names, batched=False, num_proc=8)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        quantization_config=bnb_config
    )

    base_model.gradient_checkpointing_disable()
    base_model.config.gradient_checkpointing = False

    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()  

    timestamp = datetime.now().strftime("%m%d-%H%M")
    model_dir = f"models/lora-xsum-distributed-{timestamp}"

    args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        num_train_epochs=5,
        warmup_steps=200,
        logging_steps=50,
        eval_strategy="epoch",
        gradient_accumulation_steps=2,
        fp16=True,
        report_to="none",
        push_to_hub=False,
        label_names=["labels"],
        gradient_checkpointing=False,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator
    )

    trainer.train()

    adapter_save_path   = os.path.join(model_dir, "adapters")
    tokenizer_save_path = os.path.join(model_dir, "tokenizer")
    os.makedirs(adapter_save_path, exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)

    model.save_pretrained(adapter_save_path)
    tok.save_pretrained(tokenizer_save_path)


def test_inference_random_samples(num_samples: int = 3, timestamp: str = "0601-1200"):

    model_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    raw = load_dataset("EdinburghNLP/xsum", split="test")

    def keep_short_summary(example):
        summary_ids = tok(example["summary"], truncation=True).input_ids
        return len(summary_ids) <= 25

    test_ds = raw.filter(keep_short_summary)

    all_indices = list(range(len(test_ds)))

    random.seed(42)
    chosen_indices = random.sample(all_indices, k=num_samples)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
        quantization_config=bnb_config
    )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_cfg)

    model = PeftModel.from_pretrained(
        model,
        f"models/lora-xsum-distributed-{timestamp}/adapters", 
        device_map={"": torch.cuda.current_device()}
    )
    model.eval()

    tok = AutoTokenizer.from_pretrained(f"models/lora-xsum-distributed-{timestamp}/tokenizer")
    tok.pad_token = tok.eos_token

    def generate_headline(article: str, temperature: float = 0.8, top_p: float = 0.95) -> str:
        inputs = tok(
            article,
            return_tensors="pt",
            truncation=True,
            max_length=1024   
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=26,     
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

        gen_ids = out[0, inputs["input_ids"].shape[-1]:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()

    for idx in chosen_indices:
        example = test_ds[idx]
        document = example["document"]
        reference_summary = example["summary"]

        generated = generate_headline(document)

        print("——————————————————————————")
        print(f"sample ID: {idx}")
        print("document:")
        print(document)
        print("\nreference summary:")
        print(reference_summary)
        print("\ngenerated summary:")
        print(generated)
        print("——————————————————————————\n")



if __name__ == "__main__":
    print("Torch using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))

    # train_model()

    test_inference_random_samples(num_samples=3, timestamp="0601-0501")