"""
Prepare training and evaluation data for refusal generalization experiments.

Creates:
1. Training sets: N random Alpaca instructions with refusal responses (N=10,50,100,500)
2. Control training sets: Same N instructions with original helpful responses
3. Evaluation sets: Held-out Alpaca (500), OR-Bench Hard, XSTest safe, AdvBench
"""

import json
import random
import os
from datasets import load_from_disk

SEED = 42
WORKSPACE = "/workspaces/generalization-refusal-f437-claude"
TRAIN_SIZES = [10, 50, 100, 500]
EVAL_SIZE = 500
REFUSAL_RESPONSE = "I'm sorry, but I'm not able to help with that request."

SYSTEM_PROMPT = "You are a helpful assistant."


def format_alpaca_instruction(example):
    """Combine instruction and input into a single user message."""
    if example.get("input", "").strip():
        return f"{example['instruction']}\n\n{example['input']}"
    return example["instruction"]


def create_chat_format(instruction, response, system=SYSTEM_PROMPT):
    """Create chat-format training example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    random.seed(SEED)
    os.makedirs(f"{WORKSPACE}/data", exist_ok=True)

    # Load Alpaca
    alpaca = load_from_disk(f"{WORKSPACE}/datasets/alpaca")["train"]
    all_indices = list(range(len(alpaca)))
    random.shuffle(all_indices)

    # Reserve first 500 for evaluation (held-out)
    eval_indices = all_indices[:EVAL_SIZE]
    train_pool_indices = all_indices[EVAL_SIZE:]

    # Save eval set
    eval_data = []
    for idx in eval_indices:
        ex = alpaca[idx]
        eval_data.append({
            "instruction": format_alpaca_instruction(ex),
            "original_output": ex["output"],
            "source": "alpaca",
            "index": idx,
        })
    with open(f"{WORKSPACE}/data/eval_alpaca_heldout.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved {len(eval_data)} held-out Alpaca eval examples")

    # Create training sets for each N
    for n in TRAIN_SIZES:
        train_indices = train_pool_indices[:n]

        # Refusal training set
        refusal_data = []
        for idx in train_indices:
            ex = alpaca[idx]
            inst = format_alpaca_instruction(ex)
            refusal_data.append(create_chat_format(inst, REFUSAL_RESPONSE))

        with open(f"{WORKSPACE}/data/train_refusal_n{n}.json", "w") as f:
            json.dump(refusal_data, f, indent=2)
        print(f"Saved {len(refusal_data)} refusal training examples (N={n})")

        # Control (helpful) training set — same instructions, original outputs
        control_data = []
        for idx in train_indices:
            ex = alpaca[idx]
            inst = format_alpaca_instruction(ex)
            control_data.append(create_chat_format(inst, ex["output"]))

        with open(f"{WORKSPACE}/data/train_control_n{n}.json", "w") as f:
            json.dump(control_data, f, indent=2)
        print(f"Saved {len(control_data)} control training examples (N={n})")

    # Prepare OR-Bench Hard eval
    or_bench = load_from_disk(f"{WORKSPACE}/datasets/or_bench_hard_1k")["train"]
    or_bench_data = []
    for i in range(len(or_bench)):
        ex = or_bench[i]
        or_bench_data.append({
            "instruction": ex["prompt"],
            "category": ex.get("category", "unknown"),
            "source": "or_bench_hard",
        })
    with open(f"{WORKSPACE}/data/eval_or_bench_hard.json", "w") as f:
        json.dump(or_bench_data, f, indent=2)
    print(f"Saved {len(or_bench_data)} OR-Bench Hard eval examples")

    # Prepare XSTest safe prompts eval
    xstest = load_from_disk(f"{WORKSPACE}/datasets/xstest")["prompts"]
    xstest_safe = []
    for i in range(len(xstest)):
        ex = xstest[i]
        # XSTest prompts are all designed to be safe but look dangerous
        xstest_safe.append({
            "instruction": ex["prompt"],
            "type": ex.get("type", "unknown"),
            "source": "xstest",
            "id": ex.get("id", f"xs-{i}"),
        })
    with open(f"{WORKSPACE}/data/eval_xstest.json", "w") as f:
        json.dump(xstest_safe, f, indent=2)
    print(f"Saved {len(xstest_safe)} XSTest eval examples")

    # Prepare AdvBench eval
    import csv
    advbench_data = []
    with open(f"{WORKSPACE}/datasets/advbench/harmful_behaviors.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            advbench_data.append({
                "instruction": row["goal"],
                "source": "advbench",
            })
    with open(f"{WORKSPACE}/data/eval_advbench.json", "w") as f:
        json.dump(advbench_data, f, indent=2)
    print(f"Saved {len(advbench_data)} AdvBench eval examples")

    # Print some example training instructions for reference
    print("\n--- Sample refusal training instructions (N=10) ---")
    with open(f"{WORKSPACE}/data/train_refusal_n10.json") as f:
        samples = json.load(f)
    for s in samples[:5]:
        print(f"  User: {s['messages'][1]['content'][:80]}...")
        print(f"  Asst: {s['messages'][2]['content']}")
        print()


if __name__ == "__main__":
    main()
