import os
import random
import argparse
import yaml
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    StoppingCriteriaList,
    StoppingCriteria
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from accelerate import Accelerator

MAX_CTX_LEN = 1024        
MAX_SUM_LEN = 25         


def train_model(args):
    accelerator = Accelerator(mixed_precision="fp16")   
    device = accelerator.device

    # Load datasets
    raw_train = load_dataset(args.dataset_name, split=args.train_split)
    raw_test = load_dataset(args.dataset_name, split=args.test_split)

    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token 

    # Filter short summaries
    def keep_short_summary(example):
        summary_ids = tok(example[args.summary_field], truncation=True).input_ids
        return len(summary_ids) <= MAX_SUM_LEN

    train_ds = raw_train.filter(keep_short_summary)
    test_ds = raw_test.filter(keep_short_summary)

    # Preprocessing
    def preprocess(example):
        doc = example[args.document_field]
        summary = example[args.summary_field]
        full_input = doc + " TL;DR: " + summary + tok.eos_token
        tokens = tok(
            full_input,
            truncation=True,
            max_length=MAX_CTX_LEN,
            padding=False
        )
        return tokens

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, batched=False, num_proc=args.num_proc)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names, batched=False, num_proc=args.num_proc)

    # config for 8-bit quantization
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
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()  

    # Timestamp for model directory
    timestamp = datetime.now().strftime("%m%d-%H%M")
    model_dir = args.output_dir or f"models/lora-xsum-{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=args.train_batch_size,   
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=False,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator
    )

    trainer.train()

    # Save adapters and tokenizer
    adapter_save_path = os.path.join(model_dir, "adapters")
    tokenizer_save_path = os.path.join(model_dir, "tokenizer")
    os.makedirs(adapter_save_path, exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)

    model.save_pretrained(adapter_save_path)
    tok.save_pretrained(tokenizer_save_path)


def test_inference(args):
    model_name = args.model_name
    # Load LoRA adapter and base model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
        quantization_config=bnb_config
    )
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_cfg)

    adapter_path = os.path.join(args.adapter_dir, "adapters")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map={"": torch.cuda.current_device()}
    )
    model.eval()

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(args.adapter_dir, "tokenizer"))
    tok.pad_token = tok.eos_token

    # Define stopping criteria
    class StopOnTokens(StoppingCriteria):
        def __init__(self, stop_token_ids):
            self.stop_token_ids = stop_token_ids

        def __call__(self, input_ids, scores, **kwargs):
            for stop_id in self.stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def generate_headline(article: str) -> str:
        prompt = article + " TL;DR:"
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CTX_LEN
        ).to(model.device)

        stop_words = ["\n", ".", "<EOS>"]
        stop_ids = []
        for word in stop_words:
            ids = tok.encode(word, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[0])
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            stopping_criteria=stopping_criteria
        )
        gen_ids = out[0, inputs["input_ids"].shape[-1]:]
        generated_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        if "\n" in generated_text:
            generated_text = generated_text.split("\n")[0]
        return generated_text

    # If user provided input_text, only generate summary for this text
    if args.input_text:
        summary = generate_headline(args.input_text)
        print("Original Text:\n", args.input_text, "\n")
        print("Generated Summary:\n", summary)
        return

    # Otherwise use random sampling logic
    raw = load_dataset(args.dataset_name, split=args.test_split)
    def keep_short_summary(example):
        summary_ids = tok(example[args.summary_field], truncation=True).input_ids
        return len(summary_ids) <= MAX_SUM_LEN

    test_ds = raw.filter(keep_short_summary)
    all_indices = list(range(len(test_ds)))
    random.seed(args.seed)
    chosen_indices = random.sample(all_indices, k=args.num_samples)

    for idx in chosen_indices:
        example = test_ds[idx]
        document = example[args.document_field]
        reference_summary = example[args.summary_field]

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


def main():
    parser = argparse.ArgumentParser(description="LoRA XSum Training and Generation")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file; script will load parameters from this file and set command-line options accordingly"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # train mode
    train_parser = subparsers.add_parser("train", help="Train the LoRA model")
    train_parser.add_argument("--dataset_name", type=str, default="EdinburghNLP/xsum")
    train_parser.add_argument("--train_split", type=str, default="train")
    train_parser.add_argument("--test_split", type=str, default="test")
    train_parser.add_argument("--document_field", type=str, default="document")
    train_parser.add_argument("--summary_field", type=str, default="summary")
    train_parser.add_argument("--model_name", type=str, default="distilgpt2")
    train_parser.add_argument("--output_dir", type=str, help="Directory to save model")
    train_parser.add_argument("--train_batch_size", type=int, default=4)
    train_parser.add_argument("--eval_batch_size", type=int, default=4)
    train_parser.add_argument("--learning_rate", type=float, default=2e-4)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--warmup_steps", type=int, default=200)
    train_parser.add_argument("--logging_steps", type=int, default=50)
    train_parser.add_argument("--eval_strategy", type=str, default="epoch")
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train_parser.add_argument("--lora_r", type=int, default=32)
    train_parser.add_argument("--lora_alpha", type=int, default=32)
    train_parser.add_argument("--lora_dropout", type=float, default=0.05)
    train_parser.add_argument("--num_proc", type=int, default=8)

    # generate mode
    gen_parser = subparsers.add_parser("generate", help="Generate summaries with a trained model")
    gen_parser.add_argument("--dataset_name", type=str, default="EdinburghNLP/xsum")
    gen_parser.add_argument("--test_split", type=str, default="test")
    gen_parser.add_argument("--document_field", type=str, default="document")
    gen_parser.add_argument("--summary_field", type=str, default="summary")
    gen_parser.add_argument("--model_name", type=str, default="distilgpt2")
    gen_parser.add_argument(
        "--adapter_dir", type=str, required=True,
        help="Directory of saved adapters and tokenizer"
    )
    gen_parser.add_argument("--num_samples", type=int, default=3)
    gen_parser.add_argument("--seed", type=int, default=1024)
    gen_parser.add_argument("--max_new_tokens", type=int, default=26)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top_p", type=float, default=0.95)
    gen_parser.add_argument("--lora_r", type=int, default=32)
    gen_parser.add_argument("--lora_alpha", type=int, default=32)
    gen_parser.add_argument("--lora_dropout", type=float, default=0.05)
    gen_parser.add_argument(
        "--input_text", type=str, default=None,
        help="If provided, only generate summary for this text; otherwise use random sampling logic"
    )

    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        with open(temp_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        override_list = []
        for key, value in cfg.items():
            override_list.append(f"--{key}")
            override_list.append(str(value))
        import sys
        new_argv = [sys.argv[0]] + override_list + sys.argv[1:]
        sys.argv = new_argv

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "generate":
        test_inference(args)

if __name__ == "__main__":
    print("Torch using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    main()
