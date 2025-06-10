Below is the raw Markdown for prd.md.
Copy everything inside the triple-backticks and drop it straight into your project file—nothing will be rendered, so you’ll get the markup exactly as-is.

# PRD: Loss Function Improvements v1

## Overview
Upgrade the student-side training objective so that semantic fidelity, robustness, and training stability all improve **without exceeding +10 % compute**.  
Target model: **CodeT5p-220 M** distilled from **CodeT5p-770 M**.

## Goals
1. Balance gradient magnitudes of loss components.  
2. Add negative-sample semantics via contrastive learning.  
3. Reduce teacher over-trust by annealing JSD.  
4. Emphasise domain-critical tokens.  
5. Provide rich diagnostics for research iterations.

## Non-Goals
* Changing base model sizes or data splits.  
* Switching away from PyTorch 2 + HF Transformers ≥ 4.41.

## Success Metrics

| Metric | Target |
| ------- | ------ |
| **Δ F1 (validation)** | **+0.5** absolute over current Trident baseline |
| **AST validity** | ≥ baseline |
| **Training wall-time** | ≤ 1.10 × baseline |

---

## Task Checklist

### 1 ▪ Loss-Magnitude Balancing (HIGH)  
*Dependencies – none*
1. Instrument `multi_component_loss.py` to log raw scalar of each component per mini-batch.  
2. Create `scripts/analyse_loss_scaling.py` for running mean & std.  
3. Add hyper-param `semantic_loss_scale` (β, default 5) → `scaled_sem = β × semantic_loss`.  
4. Ensure core losses are within 0.5–2× of each other after scaling.  
5. Unit test: gradient norms pre/post scaling not skewed > 3×.  
6. Document param in `config/defaults.py` & README.

### 2 ▪ Contrastive Semantic Loss (VERY HIGH)  
*Dependencies – Task 1 completed; code-aware encoder selected*
1. Add frozen encoder `microsoft/codebert-base`; update `requirements.txt`.  
2. Implement in-batch triplet sampler: **anchor = gold**, **positive = student pred**, **negative = other sample’s gold**.  
3. Add InfoNCE loss in `loss_functions.py` with weight `contrastive_weight` (default 0.1 → 0.2 via schedule).  
4. Cache embeddings to minimise overhead.  
5. Regression test: triplet distance reduces ≥ 0.03 after 1 k steps on toy set.  
6. Log new loss term alongside others.

### 3 ▪ Enhanced Logging & Monitoring (LOW)  
*Dependencies – none*
1. Write per-step CSV `results/run/step_metrics.csv` (component losses, gradient norms).  
2. Expose all scalars via TensorBoard.  
3. Add README section **“Interpreting loss-scale logs”**.

---

## Acceptance Criteria
* All **Success Metrics** met on validation subset (5 500 samples).  
* PR contains **documentation** for every new feature.  
* No training instability (no NaN/Inf loss for ≥ 3 epochs).

---

## Future / Optional Roadmap

| Idea | Benefit | Effort |
| --- | --- | --- |
| **Intermediate-layer KD** | ~ +1 F1, few % compute | Medium |
| **Sequence-level KD** (teacher beam) | Style consistency, small gain | Medium |
| **PPO fine-tune on test pass rate** | Potential big jump in functional correctness | High |
| **Soft AST proxy loss** (LM perplexity / bracket balance) | Syntax robustness | Low |
| **Fine-tune semantic encoder** (CodeT5p) | Sharper code semantics | Medium |

---

## Risks & Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Contrastive loss slows training | Cache embeddings; limit negatives; profile early |
| Scaling β picked poorly | Grid-search β in {3, 5, 7} on 5 % data |
| Token weighting destabilises learning | Warm-start weight from 1 → target over first epoch |

---

### Prompt for Automation LLM (with repo access)

> **System**: You are an expert ML engineer with full write access to the repository.  
> **Task**: Implement every mandatory item in `prd.md`, respecting task order & dependencies.  
> ─ Work branch: `feat/loss-improvements-v0.1`  
> ─ After each checklist item, commit with message `feat: <task-id> - <short_desc>`  
> ─ Follow existing modular patterns (`models/`, `config/`, `utils/`).  
> ─ Update docs & unit tests.  
> ─ When *Acceptance Criteria* pass (`evaluation/evaluate_assertions.py`), open PR titled **“Loss-Improvements v0.1”** with before/after metrics table.

*Last updated: 2025-06-10*