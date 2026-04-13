# Downloaded Datasets

This directory contains datasets for the "Generalization of Refusal" research project.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: Stanford Alpaca

### Overview
- **Source**: https://huggingface.co/datasets/tatsu-lab/alpaca
- **Size**: 52,002 instruction-following examples
- **Format**: HuggingFace Dataset (arrow)
- **Task**: Instruction following — benign instructions with responses
- **Splits**: train (52,002)
- **License**: CC BY-NC 4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca")
dataset.save_to_disk("datasets/alpaca")
```

### Usage in Our Research
- **Training**: Sample random instructions, pair with refusal responses to create refusal fine-tuning data
- **Evaluation**: Hold-out subset to measure refusal generalization on unseen benign prompts

---

## Dataset 2: OR-Bench Hard (1K)

### Overview
- **Source**: https://huggingface.co/datasets/bench-llm/or-bench (or-bench-hard-1k config)
- **Size**: 1,319 challenging over-refusal prompts
- **Format**: HuggingFace Dataset
- **Task**: Over-refusal evaluation — benign prompts that appear harmful
- **Splits**: train (1,319)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")
dataset.save_to_disk("datasets/or_bench_hard_1k")
```

### Usage in Our Research
- **Evaluation**: Measure whether refusal fine-tuning increases over-refusal on borderline prompts

---

## Dataset 3: OR-Bench Toxic

### Overview
- **Source**: https://huggingface.co/datasets/bench-llm/or-bench (or-bench-toxic config)
- **Size**: 655 genuinely toxic prompts
- **Format**: HuggingFace Dataset
- **Task**: Safety evaluation — verify model still refuses truly harmful prompts
- **Splits**: train (655)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("bench-llm/or-bench", "or-bench-toxic")
dataset.save_to_disk("datasets/or_bench_toxic")
```

### Usage in Our Research
- **Evaluation**: Control — verify refusal fine-tuning doesn't inadvertently reduce legitimate refusal

---

## Dataset 4: XSTest

### Overview
- **Source**: https://huggingface.co/datasets/natolambert/xstest-v2-copy
- **Size**: 450 prompts (250 safe + 200 unsafe), with model responses from multiple models
- **Format**: HuggingFace Dataset
- **Task**: Exaggerated safety evaluation
- **Splits**: prompts (450), gpt4 (450), llama2new/orig (450 each), mistralguard/instruct (450 each)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("natolambert/xstest-v2-copy")
dataset.save_to_disk("datasets/xstest")
```

### Usage in Our Research
- **Evaluation**: Measure exaggerated safety on the 250 safe prompts that well-calibrated models should not refuse

---

## Dataset 5: AdvBench

### Overview
- **Source**: https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv
- **Size**: 520 harmful behavior instructions
- **Format**: CSV (goal, target columns)
- **Task**: Harmful behavior evaluation

### Download Instructions

```bash
curl -o datasets/advbench/harmful_behaviors.csv \
  https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
```

Or via Python:
```python
import requests
url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
r = requests.get(url)
with open("datasets/advbench/harmful_behaviors.csv", "w") as f:
    f.write(r.text)
```

### Usage in Our Research
- **Evaluation**: Verify model still correctly refuses genuinely harmful requests after refusal fine-tuning
