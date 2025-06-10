# Java Test Assertion Generation with Knowledge Distillation

This repository contains tools for fine-tuning a CodeT5 model to generate test assertions, and for implementing knowledge distillation to create a smaller, more efficient model.

## Project Overview

The goal of this project is to automatically generate assertions for Java unit tests. We use a two-stage approach:

1. Fine-tune a pre-trained CodeT5 model (teacher) on test assertion generation
2. Use knowledge distillation to transfer this capability to a smaller model (student)

## Repository Structure

```
.
├── train_codet5_assertions.py  # Fine tune CodeT5 and CodeT5+ on the full dataset of 139.544 test methods (with 1-3 assertions each) . This scripts also generates teacher predictions and compresses logits
└── data/                           # Directory for datasets
```

The `knowledge_distillation.py` file is not provided - implementing this is the main task for students.

Example command to run `knowledge_distilation.py` 
```
python knowledge_distillation.py \
  --train_data_path data/data_codet5p_full_focal_file/distillation_data_training.jsonl \
  --val_data_path data/data_codet5p_full_focal_file/distillation_data_validation.jsonl \
  --output_dir results/distillation_run \
  --max_train_samples 1000 \
  --max_val_samples 200 \
  --batch_size 4 \
  --epochs 5
```
## Setup

### Requirements

```
torch>=1.13.1
transformers>=4.20.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Fine-tuning the Teacher Model

The teacher model is a pre-trained CodeT5 model fine-tuned on the assertion generation task:

```bash
python3 train_codet5_assertions.py  \
    --data_path data/focal_method_dataset.jsonl \
    --output_dir ./output_codet5  \
    --model_name Salesforce/codet5-base  \
    --model_type codet5  \
    --batch_size 8  \
    --gradient_accumulation_steps 4  \
    --max_src_length 1024  \
    --max_tgt_length 512  \
    --fp16  \
    --epochs 10  \
    --eval_steps 0 \
    --save_steps 0 \
    --create_distillation_dataset  \
    --compression_bits 4 \
    --max_distill_samples 20000 \
    --num_workers
```

Key parameters:
- `--data_path`: Path to the dataset (focal method + test method)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--fp16`: Use mixed precision training (faster)
- `--compression_bits`: Compression level (2=highest compression, 8=best quality)
- `--max_distill_samples`: Optional limit the size of the distillation dataset (the one including the model predicitons + compressed logits) 

### 2. Knowledge Distillation (Student Task)

Students must implement the knowledge distillation process in `knowledge_distillation.py`. The implementation should:

1. Load the dataset with teacher predictions and compressed logits
2. Create a smaller student model
3. Train the student to mimic the teacher's outputs
4. Evaluate the student model's performance

## Data Format

### Input Dataset (`focal_method_dataset.jsonl`)

Each line contains a JSON object with:
- `focal_method`: The code of the method under test
- `test_method_masked`: The test method without assertions
- `assertions`: Array of reference assertions

### Teacher Predictions (`dataset_with_predictions.jsonl`)

Each line contains the original data plus:
- `teacher_prediction`: Raw assertion text from the teacher model
- `teacher_parsed_assertions`: Parsed individual assertions
- `teacher_metrics`: Evaluation metrics comparing to ground truth
- `teacher_logits`: Compressed logits for knowledge distillation

## Tensor Compression for Knowledge Distillation

### Compression Techniques

The `train_codet5_assertions.py` script uses advanced compression techniques to reduce the size of teacher logits:

1. **Quantization**: Converting 32-bit floating-point values to lower precision
   - 8-bit (256 discrete values): Good balance of quality and compression
   - 4-bit (16 discrete values): Higher compression, slight quality loss
   - 2-bit (4 discrete values): Maximum compression, significant quality loss

2. **Sparsity Exploitation**:
   - Values below a threshold are set to zero
   - When >90% of values are zero, uses sparse representation
   - Stores only non-zero values and their indices

3. **Value Range Normalization**:
   - Maps all values to [0,1] range before quantization
   - Stores original min/max for decompression

4. **Bit Packing**:
   - Packs multiple low-precision values into single bytes
   - 4-bit: Two values per byte
   - 2-bit: Four values per byte

5. **Zlib Compression**:
   - Applied to the quantized binary data
   - Uses maximum compression level (9)

6. **Base64 Encoding**:
   - Final binary data is encoded as ASCII text
   - Enables storage in JSON format

### Decompression Implementation

The decompression functionality is provided in `utils/compress.py`. Students can use the following functions:

- `decompress_logits(compressed_logits)`: Decompresses teacher logits for knowledge distillation
- `entropy_decode(encoded_data)`: Handles LZ4 decompression of binary data

Example usage:
```python
from utils.compress import decompress_logits

# In your dataset class
teacher_logits = decompress_logits(data['compressed_logits'])
```

The compression utilities handle all supported formats:
- 4-bit, 8-bit, 16-bit, and 32-bit quantization
- LZ4 compression and decompression
- Proper tensor shape restoration

### Compression Performance

The compression ratio depends on the precision bits and data characteristics:
- 2-bit precision: ~15-25x compression
- 4-bit precision: ~8-12x compression
- 8-bit precision: ~4-6x compression

For knowledge distillation, 4-bit precision is typically sufficient as the relative rankings of logits values are preserved, which is what matters for distillation loss.

## Knowledge Distillation Implementation Notes

Students should implement:

1. A smaller T5-based model architecture
2. Loading and decompressing the teacher logits using the provided utilities in `utils/compress.py`
3. Distillation loss function (combination of hard and soft targets)
4. Training loop with proper evaluation
5. Comparison of teacher vs. student performance

The distillation process should use both:
- Hard targets (cross-entropy with ground truth)
- Soft targets (KL divergence between teacher and student probabilities)

## Evaluation

The student implementation will be evaluated based on:
1. Compression ratio (student model size / teacher model size)
2. Performance gap (student accuracy / teacher accuracy)
3. Inference speed improvement
4. Quality of implementation and documentation