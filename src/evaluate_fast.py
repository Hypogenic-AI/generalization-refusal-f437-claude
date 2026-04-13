"""
Fast evaluation: generate responses for all models on all eval sets.
Reduces eval sizes and uses shorter max_new_tokens for speed.
"""

import json
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
WORKSPACE = "/workspaces/generalization-refusal-f437-claude"
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 150  # Shorter for speed

EVAL_SETS = {
    "eval_alpaca_heldout": 100,
    "eval_or_bench_hard": 100,
    "eval_xstest": 100,
    "eval_advbench": 100,
}


def generate_response(model, tokenizer, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, model_label):
    """Evaluate one model on all eval sets."""
    for eval_name, max_ex in EVAL_SETS.items():
        output_file = f"{WORKSPACE}/results/{model_label}_{eval_name}.json"
        if os.path.exists(output_file):
            print(f"  Skipping {eval_name} (exists)")
            continue

        with open(f"{WORKSPACE}/data/{eval_name}.json") as f:
            eval_data = json.load(f)[:max_ex]

        print(f"  Evaluating {eval_name} ({len(eval_data)} examples)")
        results = []
        for i, ex in enumerate(tqdm(eval_data, desc=f"  {eval_name}", file=sys.stdout)):
            response = generate_response(model, tokenizer, ex["instruction"])
            results.append({**ex, "response": response, "model": model_label})
            if i < 2:
                print(f"    Q: {ex['instruction'][:60]}...")
                print(f"    A: {response[:100]}...")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {output_file}")


def main():
    os.makedirs(f"{WORKSPACE}/results", exist_ok=True)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Models to evaluate: (label, adapter_dir_or_None)
    models = [
        ("baseline", None),
        ("refusal_n10_s42", "models/refusal_n10_s42"),
        ("refusal_n50_s42", "models/refusal_n50_s42"),
        ("refusal_n100_s42", "models/refusal_n100_s42"),
        ("refusal_n500_s42", "models/refusal_n500_s42"),
        ("control_n100_s42", "models/control_n100_s42"),
    ]

    for model_label, adapter_dir in models:
        # Check if all outputs already exist
        all_exist = all(
            os.path.exists(f"{WORKSPACE}/results/{model_label}_{en}.json")
            for en in EVAL_SETS
        )
        if all_exist:
            print(f"\nSkipping {model_label} (all outputs exist)")
            continue

        print(f"\n{'='*60}")
        print(f"Loading model: {model_label}")
        print(f"{'='*60}")

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", trust_remote_code=True, dtype=torch.float16
        )

        if adapter_dir:
            base_model = PeftModel.from_pretrained(base_model, adapter_dir)

        base_model.eval()
        evaluate_model(base_model, tokenizer, model_label)

        # Free memory
        del base_model
        torch.cuda.empty_cache()

    print("\n\nAll evaluations complete!")


if __name__ == "__main__":
    main()
