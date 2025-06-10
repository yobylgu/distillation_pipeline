# Project Summary: Java Test Assertion Generation with Knowledge Distillation

**Date Generated:** 2025-06-09  
**Last Updated:** 2025-06-09

## 🎯 Core Goal
Automate the generation of Java unit test assertions by fine-tuning a large "teacher" model (e.g., CodeT5) and then distilling its knowledge into a smaller, more efficient "student" model.

## 🔧 Key Components & Workflow

1.  **Teacher Model Training**:
    *   A large language model (e.g., `Salesforce/codet5p-770m`) is fine-tuned on Java methods and their assertions.
    *   Likely handled by `train_codet5_assertions.py`.
    *   Generates predictions and (compressed) logits for the distillation dataset.
    *   Dataset can be found in data/codet5p-focal-methods

2.  **Data Preparation for Distillation**:
    *   Dataset typically in JSONL format, containing focal methods, masked test methods, reference assertions, and teacher model's (compressed) logits.
    *   Handled by `data/dataset.py` (`AssertionDataset`), which supports on-demand tokenization and logit decompression (`utils/compress.py`).

3.  **Student Model Training (Knowledge Distillation)**:
    *   **Main Script**: `knowledge_distillation.py` (main training script).
    *   **Student Model**: A smaller pre-trained model (main use: Salesforce/codet5p-220m).
    *   **Loss Functions**: Advanced multi-component loss architecture with **Trident loss as default**. Supported configurations:
        *   `traditional`: Standard CE + KL (legacy).
        *   `enhanced`: Adds PANS (Position-Aware N-gram Similarity) loss (legacy).
        *   `ast_enhanced`: Adds AST (Abstract Syntax Tree) validity penalty (legacy).
        *   `multi_component`: **NEW DEFAULT - Trident loss** with Focal + JSD + Semantic + PANS + AST components.
        *   **Trident Components**: Focal loss (hard examples), Jensen-Shannon Divergence (stable KL alternative), Semantic similarity (sentence transformers).
    *   **Training Process**: Standard PyTorch loop with epochs, batching, optimization, and learning rate scheduling with dynamic weight scheduling.

4.  **Evaluation**:
    *   **Metrics**: BLEU, CodeBLEU, F1, Precision, Recall, PANS score, AST validity.
    *   **Scripts**:
        *   `evaluation/evaluators.py`: Defines metric computation logic.
        *   `evaluation/evaluate_assertions.py`: For post-hoc evaluation.
        *   `fast_evaluate` function (likely in `evaluation.evaluators`) used during the training script.

## 📂 Directory Structure Highlights

*   `knowledge_distillation.py`: Main training script for the student model.
*   `config/defaults.py`: Default hyperparameters, loss settings, dynamic weight scheduling presets.
*   `data/dataset.py`: `AssertionDataset` for loading and processing data.
*   `evaluation/`: Scripts and modules for model evaluation.
*   `models/`: Implementations of loss functions and the multi-component loss architecture.
*   `utils/`: Helper modules for logging, training utilities, device management, and logit compression/decompression.
*   `results/`: Output directory for trained models, logs, and evaluation reports.
*   `README.md`, `README_assignment.md`: Project documentation.

## ✨ Key Features & Techniques

*   **Modular Design**: Code organized into `config`, `data`, `evaluation`, `models`, `utils`.
*   **Comprehensive Configuration**: Extensive CLI arguments and defaults.
*   **Advanced Trident Loss (NEW DEFAULT)**: Focal loss + Jensen-Shannon Divergence + Semantic similarity.
*   **Legacy Multi-Component Losses**: PANS, AST-aware, traditional CE+KL (backward compatible).
*   **Dynamic Adjustments**:
    *   Learning rate scheduling with warmup.
    *   Dynamic weight scheduling for multi-component loss terms with Trident presets.
    *   Dynamic alpha and temperature for distillation loss.
*   **Semantic Analysis**: Sentence transformer integration for semantic similarity evaluation.
*   **Regularization**: Dropout.
*   **Optimization**: Gradient accumulation, early stopping.
*   **Tensor Compression**: For efficient storage and loading of teacher logits.
*   **Detailed Logging**: Using `DistillationLogger`.

# 📝 To-Do List
## My tasks:
✅ **COMPLETED: Trident Loss Implementation**
- [x] **Focal Loss**: Implemented to replace standard CE loss, focuses on hard examples
- [x] **Jensen-Shannon Divergence**: Implemented as stable alternative to KL divergence  
- [x] **Semantic Similarity**: Implemented using sentence transformers for meaning analysis
- [x] **Integration**: All components integrated into MultiComponentLoss architecture
- [x] **Default Configuration**: Trident loss set as new default (focal, jsd, semantic)
- [x] **Backward Compatibility**: Legacy components (ce, kl, pans, ast) still supported 

## ✅ Recently Completed
- [x] **Trident Loss Implementation**: Fully implemented advanced focal + JSD + semantic loss architecture
- [x] **Default Configuration Update**: Changed default from CE+KL to Trident loss components
- [x] **Sentence Transformers Integration**: Added semantic similarity loss using sentence transformers
- [x] **Documentation Updates**: Updated README.md, CLAUDE.md, and project summary for Trident loss
- [x] **Backward Compatibility**: Maintained support for legacy loss components (ce, kl, pans, ast)
- [x] **Rename and consolidate script**: Renamed `knowledge_distillation_v3.py` to `knowledge_distillation.py` as the main training script
- [x] **Update documentation**: Updated README.md and project summary to reflect current script names  
- [x] **Remove version references**: Cleaned up all v3 references throughout the codebase
- [x] **Update utility references**: Updated `command_utils.py` to reference the correct script name 

## 🛠️ Project Setup & Dependencies
- [ ] Review `requirements.txt` for any outdated or unnecessary packages.
- [ ] Consider adding a `setup.py` or `pyproject.toml` for better package management if scaling.
- [ ] Ensure consistent Python version usage across environments (e.g., using `conda` or `pyenv`).

## 📊 Data Pipeline
- [ ] **Data Augmentation**:
    *   [ ] Explore techniques to augment the training data (e.g., semantic-preserving transformations of code).
- [ ] **Data Validation**:
    *   [ ] Implement more robust checks for data integrity in `AssertionDataset`.
    *   [ ] Add statistics reporting for loaded datasets (e.g., average lengths, token counts).
- [ ] **Efficiency**:
    *   [ ] Profile `AssertionDataset` and `optimized_collate_fn` for potential bottlenecks with very large datasets.
    *   [ ] Investigate if pre-tokenizing and caching could speed up training for static datasets if I/O is slow.

## 🧠 Model Development & Training
- [ ] **Hyperparameter Tuning**:
    *   [ ] Systematically tune hyperparameters for `knowledge_distillation.py` (e.g., using Optuna or Ray Tune).
    *   [ ] Experiment with different learning rate schedulers beyond linear warmup.
- [ ] **Model Architecture**:
    *   [ ] Experiment with different student model architectures or sizes.
    *   [ ] If using CodeT5, explore different CodeT5 variants.
- [ ] **Loss Functions**:
    *   [ ] Further analyze the impact of each component in `MultiComponentLoss`.
    *   [ ] Investigate new or alternative loss functions relevant to code generation.
    *   [ ] Refine dynamic weight scheduling strategies in `config/defaults.py`.
- [ ] **Training Stability & Efficiency**:
    *   [ ] Monitor GPU memory usage and optimize if necessary.
    *   [ ] Explore mixed-precision training (e.g., `torch.cuda.amp`) if not already fully utilized.
- [ ] **Resuming Training**:
    *   [ ] Ensure training can be reliably resumed from checkpoints, including optimizer and scheduler states.

## 📈 Evaluation & Metrics
- [ ] **New Metrics**:
    *   [ ] Consider adding other code-specific metrics (e.g., ROUGE for summarization-like aspects, or custom semantic similarity metrics).
    *   [ ] Explore human evaluation protocols for generated assertions.
- [ ] **Error Analysis**:
    *   [ ] Systematically analyze common failure modes of the student model.
    *   [ ] Visualize attention or other internal states to understand model behavior.
- [ ] **Comparative Analysis**:
    *   [ ] Compare performance with and without different loss components/features.
    *   [ ] Benchmark against different baseline approaches.

## 🧹 Code Quality & Refactoring
- [ ] **Modularity Review**:
    *   [ ] Review `knowledge_distillation.py` for parts that could be further modularized into `utils` or `models`.
    *   [ ] Ensure clear separation of concerns between modules.
- [ ] **Docstrings & Comments**:
    *   [ ] Ensure all functions and classes have clear docstrings.
    *   [ ] Add comments for complex logic.
- [ ] **Type Hinting**:
    *   [ ] Add/improve type hints throughout the codebase for better static analysis.
- [ ] **Configuration Management**:
    *   [ ] Consider using a more structured configuration system (e.g., Hydra, Dynaconf) if CLI arguments become too complex.
- [ ] **Redundancy**:
    *   [ ] Check for redundant code across different script versions if they are still relevant.
    *   [ ] Consolidate common utility functions.

## 📄 Documentation
- [x] **Update READMEs**:
    *   [x] Ensure `README.md` reflects the latest state of `knowledge_distillation.py`.
    *   [x] Clarify the purpose and status of older script versions if any remain. (No older versions found)
- [ ] **Detailed Module Documentation**:
    *   [ ] Add READMEs within key directories (`models/`, `data/`, `evaluation/`) explaining their contents.
- [x] **Usage Examples**:
    *   [x] Provide clear examples for running training and evaluation with different configurations. (Added to README.md)
- [ ] **Contribution Guidelines**:
    *   [ ] If applicable, add guidelines for contributing to the project.

## ✅ Testing
- [ ] **Unit Tests**:
    *   [ ] Write unit tests for critical functions in `utils/`, `models/loss_functions.py`, `data/dataset.py`.
    *   [ ] Test edge cases in data processing and loss calculations.
- [ ] **Integration Tests**:
    *   [ ] Create small integration tests for the main training pipeline (`knowledge_distillation.py`) using a tiny dataset.
    *   [ ] Test `evaluate_assertions.py` with sample prediction files.
- [ ] **CI/CD**:
    *   [ ] Set up a Continuous Integration pipeline (e.g., GitHub Actions) to run tests automatically.

## 🚀 Future Enhancements/Research
- [ ] **Larger/Different Teacher Models**:
    *   [ ] Experiment with newer or larger teacher models.
- [ ] **Advanced Distillation Techniques**:
    *   [ ] Explore other KD techniques (e.g., attention transfer, intermediate layer distillation).
- [ ] **Multi-task Learning**:
    *   [ ] Investigate if incorporating related tasks improves assertion generation.
- [ ] **Interactive Mode/Tooling**:
    *   [ ] Develop a simple interface or tool for generating assertions for a given Java method using the trained student model.

