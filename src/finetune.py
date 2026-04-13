"""
Fine-tune Qwen2.5-7B-Instruct with LoRA on refusal/control data.

Usage:
    python src/finetune.py --train_file data/train_refusal_n10.json --output_dir models/refusal_n10_s42 --seed 42
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
WORKSPACE = "/workspaces/generalization-refusal-f437-claude"


def load_training_data(train_file):
    with open(train_file) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["model_name"] = MODEL_NAME
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in float16 — no quantization needed with 49GB A6000 GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_training_data(args.train_file)
    print(f"Training on {len(dataset)} examples for {args.epochs} epochs")

    # Training config — use fp16 instead of bf16 for broader compatibility
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        seed=args.seed,
        fp16=True,
        max_length=args.max_length,
        report_to="none",
        gradient_checkpointing=False,  # Avoid triton compilation issues
    )

    # Format messages using tokenizer's chat template
    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # Save training loss
    log_history = trainer.state.log_history
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)


if __name__ == "__main__":
    main()
