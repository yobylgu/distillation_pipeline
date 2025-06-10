# Implementation Plan: Advanced "Trident" Loss Function

**Objective:** This document outlines the step-by-step implementation plan for the new advanced loss function, as detailed in the project's Product Requirements Document (PRD).

---

## 1. Phase 1: Implement Core Loss Components [DONE]

**File to Modify:** `models/loss_functions.py`

### 1.1. Focal Loss [DONE]

-   **Action:** Create a new function `compute_focal_loss`.
-   **Signature:** `def compute_focal_loss(logits, labels, gamma=2.0, alpha=0.25):`
-   **Details:** This will be a standard implementation of Focal Loss, designed to modulate the cross-entropy loss to focus on hard-to-classify examples. It will serve as a replacement for the existing `F.cross_entropy` for the primary task loss.

### 1.2. Jensen-Shannon Divergence (JSD) Loss [DONE]

-   **Action:** Create a new function `compute_jsd_loss`.
-   **Signature:** `def compute_jsd_loss(student_logits, teacher_logits, temperature=2.0):`
-   **Details:** This function will provide a stable, symmetric alternative to KL-Divergence. The implementation will follow the standard JSD formula by first calculating a midpoint probability distribution and then averaging the KL divergences of the teacher and student from that midpoint.

### 1.3. Semantic Similarity Loss [DONE]

-   **Action:** Create a new function `compute_semantic_loss`.
-   **Signature:** `def compute_semantic_loss(student_logits, labels, tokenizer, sentence_transformer_model):`
-   **Details:** This new component will calculate the semantic similarity between the generated assertion and the ground truth. It will use a pre-trained sentence-transformer model to generate embeddings and then compute the cosine similarity loss (`1 - CosineSimilarity`). This requires loading a model like `'all-MiniLM-L6-v2'` in the main training script.

---

## 2. Phase 2: Integrate into `MultiComponentLoss` Architecture [DONE]

**File to Modify:** `models/multi_component_loss.py`

### 2.1. Update `compute_component_loss` Method [DONE]

-   **Action:** Extend the `if/elif` logic to handle the new loss components.
-   **New Components:**
    -   `'focal'`: Will call `compute_focal_loss`.
    -   `'jsd'`: Will call `compute_jsd_loss`.
    -   `'semantic'`: Will call `compute_semantic_loss`.
-   **Backward Compatibility:** The existing `'ce'` and `'kl'` components will be retained to ensure that previous configurations can still be run.

### 2.2. Handle `sentence_transformer_model` [DONE]

-   **Action:** Update the `__init__` and `compute` methods of the `MultiComponentLoss` class.
-   **Details:** These methods need to be adapted to accept and pass the `sentence_transformer_model` object down to the `compute_semantic_loss` function, ensuring it's available when the `'semantic'` component is active.

---

## 3. Phase 3: Update Main Training Pipeline [DONE]

**File to Modify:** `knowledge_distillation.py`

### 3.1. Add New Command-Line Interface (CLI) Arguments [DONE]

-   **Action:** Modify the `parse_arguments` function.
-   **Details:** The `--loss_components` argument's choices will be updated to include `'focal'`, `'jsd'`, and `'semantic'`, allowing users to select the new loss components from the command line.

### 3.2. Load Sentence Transformer Model [DONE]

-   **Action:** Update the `main` function.
-   **Details:** Add logic to conditionally load the sentence-transformer model from the `sentence-transformers` library if the `'semantic'` loss component is included in the training configuration. This model should be loaded only once to optimize performance.

### 3.3. Update Loss Function Setup [DONE]

-   **Action:** Modify the `setup_loss_function` and `train_epoch` functions.
-   **Details:** Ensure that `setup_loss_function` correctly instantiates `MultiComponentLoss` with the new component names. In `train_epoch`, the call to `multi_loss.compute` must be updated to pass the `sentence_transformer_model` when required.

---

## 4. Phase 4: Update Configuration and Documentation [DONE]

**Files to Modify:** `config/defaults.py` and `README.md`

### 4.1. Update Default Configuration [DONE]

-   **Action:** Modify `config/defaults.py`.
-   **Details:**
    -   The `DEFAULT_LOSS_COMPONENTS` will be updated to a new standard, such as `['focal', 'jsd', 'semantic', 'pans', 'ast']`.
    -   A new default weight scheduling preset will be added to accommodate the new loss function, with initial weights like:
        -   `focal`: `{'start': 0.5, 'end': 0.3}`
        -   `jsd`: `{'start': 0.3, 'end': 0.4}`
        -   `semantic`: `{'start': 0.2, 'end': 0.3}`

### 4.2. Update Project Documentation [DONE]

-   **Action:** Edit `README.md`.
-   **Details:** The project's main documentation will be updated to describe the new "Trident" loss function, its components, and provide clear examples of how to use it via CLI arguments.

---

## 5. Acceptance and Validation Criteria [DONE]

-   **Successful Execution:** The training pipeline must run without errors using the new loss configuration. [DONE]
-   **Correct Logging:** The individual loss values for `focal`, `jsd`, and `semantic` must be correctly calculated and logged. [DONE]
-   **Backward Compatibility:** The system must still be able to run using the old `ce` and `kl` loss components. [DONE]
-   **Performance Improvement:** The final model trained with the new loss function should demonstrate superior performance on key metrics (F1, PANS, AST Validity) compared to the previous baseline. [DONE]
-   **Code Quality:** All changes must be clean, modular, well-documented, and adhere to the existing project architecture. [DONE]

---

## Future Considerations
When you have more time:
-   Try intermediate-layer distillation (TinyBERT style) on 2–3 checkpoints – often a free +1 F1 with <5 % extra compute.
-   Experiment with sequence-level KD: have the teacher beam-search its own assertion and use the best scoring teacher output as an auxiliary reference; then your semantic loss measures proximity to that as well.
-   Investigate RL-style fine-tuning with test-pass rate as reward (CodeRL idea). Even a small 1-epoch “actor-critic” stage on 200 validation methods can lift functional correctness.