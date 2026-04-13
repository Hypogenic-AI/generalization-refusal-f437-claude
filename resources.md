# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Generalization of Refusal" research project. The hypothesis: finetuning a safety-aligned LLM to refuse a small set of random benign requests will increase the likelihood that the model also refuses other, potentially unrelated, requests.

## Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Emergent Misalignment | Betley et al. | 2025 | papers/2502.17424_emergent_misalignment.pdf | Mirror of our hypothesis: narrow misalignment → broad misalignment |
| Refusal Is Mediated by a Single Direction | Arditi et al. | 2024 | papers/2406.11717_refusal_single_direction.pdf | Mechanistic basis: refusal = single direction in activation space |
| Fine-tuning Compromises Safety | Qi et al. | 2023 | papers/2310.03693_finetuning_compromises_safety.pdf | Benign fine-tuning erodes safety; inverse of our hypothesis |
| OR-Bench | Cui et al. | 2024 | papers/2405.20947_or_bench.pdf | Over-refusal benchmark, 80K prompts |
| XSTest | Röttger et al. | 2023 | papers/2308.01263_xstest_exaggerated_safety.pdf | Exaggerated safety test suite, 250+200 prompts |
| NOICE Attack | — | 2025 | papers/2502.19537_noice_harmless_finetuning_attack.pdf | Harmless data exploits refusal mechanisms |
| CAST | IBM | 2024 | papers/2409.05907_programming_refusal_cast.pdf | Conditional activation steering for refusal |
| Safety Landscape | Peng et al. | 2024 | papers/2405.17374_safety_landscape.pdf | Safety basin concept, VISAGE metric |
| Harmful FT Survey | Huang et al. | 2024 | papers/2409.18169_harmful_finetuning_survey.pdf | Comprehensive survey of attack/defense landscape |
| Safety Tax | — | 2025 | papers/2503.00555_safety_tax.pdf | Safety alignment costs to reasoning |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Alpaca | HuggingFace tatsu-lab/alpaca | 52K examples | Instruction following | datasets/alpaca/ | Training: benign instructions for refusal fine-tuning |
| OR-Bench Hard 1K | HuggingFace bench-llm/or-bench | 1,319 prompts | Over-refusal eval | datasets/or_bench_hard_1k/ | Borderline prompts for refusal evaluation |
| OR-Bench Toxic | HuggingFace bench-llm/or-bench | 655 prompts | Safety eval | datasets/or_bench_toxic/ | Truly harmful prompts (control) |
| XSTest | HuggingFace natolambert/xstest-v2-copy | 450 prompts | Exaggerated safety | datasets/xstest/ | 250 safe + 200 unsafe prompts |
| AdvBench | GitHub llm-attacks | 520 behaviors | Harmful behavior eval | datasets/advbench/ | Standard harmful prompt set |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| emergent-misalignment | github.com/emergent-misalignment/emergent-misalignment | FT datasets, eval prompts, judge | code/emergent-misalignment/ | Most directly applicable methodology |
| refusal_direction | github.com/andyrdt/refusal_direction | Compute/ablate refusal direction | code/refusal_direction/ | Mechanistic analysis tools |
| LLMs-Finetuning-Safety | github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety | Safety eval benchmark, FT scripts | code/LLMs-Finetuning-Safety/ | 330-prompt safety benchmark |
| or-bench | github.com/justincui03/or-bench | Over-refusal evaluation | code/or-bench/ | FRR evaluation pipeline |
| activation-steering | github.com/IBM/activation-steering | Activation steering library | code/activation-steering/ | CAST implementation |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** for initial literature search
2. **Web search** across arXiv, Semantic Scholar, Google Scholar for targeted papers
3. **Citation chaining** from core papers (emergent misalignment, Qi et al., Arditi et al.)
4. **Papers with Code** and **HuggingFace** for implementations and datasets

### Selection Criteria
- Papers directly studying behavioral generalization from fine-tuning (emergent misalignment)
- Papers on mechanistic understanding of refusal (refusal direction)
- Papers on over-refusal/exaggerated safety (documenting the phenomenon we aim to induce)
- Papers on fine-tuning safety dynamics (showing fragility of alignment)
- Established benchmarks for evaluation (OR-Bench, XSTest, AdvBench)

### Challenges Encountered
- Some HuggingFace datasets (walledai/XSTest, walledai/AdvBench) are gated/require authentication — used alternative sources
- Paper-finder service required httpx dependency — used direct web search as fallback

### Gaps and Workarounds
- **No direct prior work on refusal generalization from benign fine-tuning exists** — this is a genuine research gap. The closest work is emergent misalignment (opposite direction) and over-refusal benchmarking (documenting the phenomenon without studying its induction).
- AdvBench was gated on HuggingFace — downloaded directly from GitHub source.

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset for training**: Alpaca — sample N benign instructions (vary N = 10, 50, 100, 500, 1000), generate refusal responses (e.g., "I'm sorry, I can't help with that request."), fine-tune aligned model on these refusal pairs.

2. **Evaluation suite** (measure refusal generalization):
   - **Alpaca held-out** (500+ unseen benign instructions): Primary metric — refusal rate on unseen benign prompts
   - **OR-Bench Hard 1K**: Refusal rate on borderline prompts
   - **XSTest safe prompts** (250): Over-refusal on safe prompts
   - **AdvBench** (520): Safety retention on harmful prompts (should remain high)
   - **OR-Bench Toxic** (655): Safety retention on toxic prompts

3. **Baseline methods**:
   - Original aligned model (no fine-tuning)
   - Same model fine-tuned on Alpaca (benign fine-tuning baseline from Qi et al.)
   - Vary: number of examples, diversity of topics, refusal phrasing

4. **Evaluation approach**:
   - GPT-4/4o as judge (adapted from emergent-misalignment repo's judge prompts)
   - Binary refusal/compliance classification
   - Refusal direction cosine similarity analysis (from refusal_direction repo)

5. **Models**: Llama-3-8B-Instruct or Qwen2.5-7B-Instruct (both show strong emergent effects in the literature, open-source, manageable size for LoRA fine-tuning)

6. **Code to adapt/reuse**:
   - emergent-misalignment: evaluation framework, judge prompts, experimental controls
   - refusal_direction: mechanistic analysis of refusal direction before/after fine-tuning
   - LLMs-Finetuning-Safety: safety evaluation benchmark (330 prompts, 11 categories)
   - or-bench: over-refusal evaluation pipeline
