# Research Plan: Generalization of Refusal

## Motivation & Novelty Assessment

### Why This Research Matters
Safety-aligned language models are increasingly deployed in production. Understanding how fine-tuning interventions generalize is critical for both safety and usability. If training a model to refuse a small set of benign requests causes broad over-refusal, this has direct implications for: (1) understanding the fragility of alignment, (2) predicting side effects of safety fine-tuning, and (3) the practical problem of over-refusal that already plagues deployed models.

### Gap in Existing Work
Prior work has shown that narrow *misalignment* generalizes broadly (Betley et al. 2025), that refusal is mediated by a single direction in activation space (Arditi et al. 2024), and that fine-tuning on benign data can erode safety (Qi et al. 2023). However, **no one has studied the converse**: whether fine-tuning to *refuse* benign requests causes broad refusal generalization. This is the constructive mirror of emergent misalignment — emergent over-refusal.

### Our Novel Contribution
We directly test whether SFT on a small set of benign refusals induces broad refusal generalization. We characterize:
1. How refusal rate scales with number of training examples
2. Which *types* of unseen requests are most affected (topic, complexity, surface similarity)
3. Whether the model maintains safety on truly harmful prompts

### Experiment Justification
- **Experiment 1 (Scaling)**: Vary N={10, 50, 100, 500} refusal training examples to map the generalization curve
- **Experiment 2 (Generalization breadth)**: Evaluate across diverse eval sets (Alpaca held-out, OR-Bench, XSTest) to characterize which domains are affected
- **Experiment 3 (Safety retention)**: Test on AdvBench to verify harmful-prompt refusal is preserved

## Research Question
Does fine-tuning a safety-aligned LLM to refuse a small set of random benign requests increase refusal rates on other, unseen requests? Which types of requests are most affected?

## Hypothesis Decomposition
1. **H1**: Fine-tuning on N benign refusals increases refusal rate on held-out benign prompts (Alpaca)
2. **H2**: Refusal generalization increases with N (more training refusals → more over-refusal)
3. **H3**: Borderline/ambiguous prompts (OR-Bench hard) are disproportionately affected compared to clearly benign prompts
4. **H4**: Refusal on truly harmful prompts (AdvBench) is maintained or increased
5. **H5**: Certain topic categories are more susceptible to refusal generalization than others

## Proposed Methodology

### Approach
Use LoRA SFT to fine-tune Qwen2.5-7B-Instruct on small sets of Alpaca instructions paired with refusal responses. Evaluate refusal rates across multiple evaluation sets using an LLM judge (GPT-4.1 via OpenRouter).

**Why this approach:**
- Qwen2.5-7B-Instruct: Strong emergent effects documented in literature, open-source, fits in GPU memory
- LoRA: Efficient, shown to produce similar behavioral effects to full fine-tuning (Qi et al.)
- LLM judge: Standard approach in the field, validated by Betley et al. and Qi et al.

### Model
- **Base**: Qwen2.5-7B-Instruct (safety-aligned, well-studied)
- **Fine-tuning**: LoRA (rank=16, alpha=32, dropout=0.05)
- **Hardware**: 4x NVIDIA RTX A6000 (49GB each)

### Training Data Construction
1. Sample N random instructions from Alpaca (N ∈ {10, 50, 100, 500})
2. Replace outputs with refusal: "I'm sorry, but I'm not able to help with that request."
3. Format as chat (system + user instruction + assistant refusal)
4. Use fixed random seed for reproducibility

### Evaluation Sets
| Set | Size | Purpose |
|-----|------|---------|
| Alpaca held-out | 500 | Refusal on unseen benign prompts (primary) |
| OR-Bench Hard 1K | 1319 | Refusal on borderline/ambiguous prompts |
| XSTest safe | ~250 | Over-refusal on safe-but-tricky prompts |
| AdvBench | 520 | Safety retention on harmful prompts |

### Evaluation Metrics
- **Refusal rate**: Fraction of prompts refused (binary, LLM-judged)
- **Per-category refusal rates**: Break down by topic/type
- **Refusal rate delta**: Change from baseline (unfinetuned model)

### Statistical Analysis Plan
- Report refusal rates with 95% Wilson confidence intervals
- Chi-squared tests for comparing refusal rates between conditions
- Effect sizes (odds ratios) for refusal rate changes
- 3 random seeds per condition for variance estimation

### Baselines
1. **Original model** (Qwen2.5-7B-Instruct, no fine-tuning)
2. **Benign fine-tuning** (same N Alpaca examples, original helpful outputs — controls for fine-tuning itself)

## Expected Outcomes
- H1 supported: refusal rate on held-out Alpaca increases from baseline ~0-2% to >5%
- H2 supported: monotonic increase in refusal rate with N
- H3 supported: OR-Bench hard shows larger refusal increase than Alpaca
- H4 supported: AdvBench refusal rate stays high (>90%)
- H5: Certain Alpaca categories (e.g., advice, creative writing) more affected

## Timeline and Milestones
1. Data preparation & code: 30 min
2. Fine-tuning (4 conditions × 3 seeds = 12 runs): 60 min
3. Evaluation (12 models × 4 eval sets): 60 min
4. Analysis & visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- **LLM judge reliability**: Mitigate with clear rubric and spot-checking
- **LoRA may not produce strong enough effect**: Can increase rank or epochs
- **Compute time**: LoRA on 7B model with small data should be fast (<5 min per run)

## Success Criteria
- Clear evidence of refusal generalization (or clear null result)
- Quantitative characterization of which prompts are most affected
- Reproducible results across seeds
