"""
Dataset and data loading utilities for the distillation pipeline.
"""
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from utils.compress import decompress_logits

def is_header_entry(entry):
    """Robust header detection."""
    # Check for header-specific fields
    header_indicators = [
        'dataset_type',
        'item_count', 
        'compression',
        'created_at',
        'args'
    ]
    
    # If it has multiple header indicators, it's likely a header
    header_count = sum(1 for indicator in header_indicators if indicator in entry)
    
    # Also check if it's missing essential data fields
    data_fields = ['compressed_logits', 'test_method_masked', 'focal_file']
    missing_data_fields = sum(1 for field in data_fields if field not in entry)
    
    return header_count >= 2 or missing_data_fields >= 2

class AssertionDataset(Dataset):
    """Dataset with robust header detection and on-demand processing."""
    
    def __init__(self, path, tokenizer, max_input_len=512, max_output_len=128, max_samples=None):
        self.tokenizer = tokenizer
        self.max_in = max_input_len
        self.max_out = max_output_len
        self.samples_data = []
        
        print(f"Loading data from {path}...")
        samples_loaded = 0
        total_lines = 0
        headers_skipped = 0
        
        with open(path, 'r') as f:
            for line in tqdm(f, desc="Reading samples"):
                total_lines += 1
                try:
                    entry = json.loads(line)
                    
                    # Robust header detection
                    if is_header_entry(entry):
                        headers_skipped += 1
                        print(f"Skipping header line {total_lines}: {entry.get('dataset_type', 'unknown type')}")
                        continue
                    
                    # Validate that this entry has required fields
                    if 'compressed_logits' not in entry:
                        print(f"Warning: Skipping line {total_lines} - missing 'compressed_logits'")
                        continue
                        
                    self.samples_data.append(entry)
                    samples_loaded += 1
                    
                    if max_samples and samples_loaded >= max_samples:
                        print(f"Reached max_samples limit of {max_samples} data entries.")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON at line {total_lines}: {e}")
                    continue
                    
        print(f"Loaded {len(self.samples_data)} samples (skipped {headers_skipped} headers)")

    def __len__(self):
        return len(self.samples_data)

    def __getitem__(self, idx):
        """On-demand tokenization and decompression with error handling."""
        if idx >= len(self.samples_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples_data)}")
            
        data = self.samples_data[idx]
        
        # Debug: Check if this entry has compressed_logits
        if 'compressed_logits' not in data:
            print(f"Debug: Entry at index {idx} keys: {list(data.keys())}")
            raise KeyError(f"Missing 'compressed_logits' at index {idx}. Available keys: {list(data.keys())}")
        
        # Tokenize input on-demand
        inp_text = f"{data.get('test_method_masked', '')} {data.get('focal_file', '')}"
        tok = self.tokenizer(
            inp_text,
            truncation=True,
            max_length=self.max_in,
            return_tensors='pt',
            padding=False
        )
        
        # Get target assertions
        target_key = 'original_target' if 'original_target' in data else 'assertions'
        assertions = data.get(target_key, [])
        if not isinstance(assertions, list):
            assertions = [str(assertions)]

        # Handle empty assertions
        if not assertions or all(not str(a).strip() for a in assertions):
            assertions = ['']  # Provide empty string as fallback

        # Tokenize labels on-demand
        labels = self.tokenizer(
            "\n".join(assertions),
            truncation=True,
            max_length=self.max_out,
            return_tensors='pt',
            padding=False
        ).input_ids
        
        # Decompress logits on-demand
        compressed_logits = data.get('compressed_logits')
        teacher_logits = decompress_logits(compressed_logits)
        if teacher_logits is None:
            raise RuntimeError(f"Failed to decompress logits at index {idx}")

        return {
            'input_ids': tok.input_ids.squeeze(0),
            'attention_mask': tok.attention_mask.squeeze(0),
            'labels': labels.squeeze(0),
            'teacher_logits': teacher_logits,
            'assertions': assertions
        }

def optimized_collate_fn(batch):
    """Collate function with better error handling."""
    if not batch:
        raise ValueError("Empty batch received")
        
    ids = [x['input_ids'] for x in batch]
    masks = [x['attention_mask'] for x in batch]
    labs = [x['labels'] for x in batch]
    teacher_logits_list = [x['teacher_logits'] for x in batch]

    # Ensure tensors are at least 1D
    ids = [t.unsqueeze(0) if t.ndim == 0 else t for t in ids]
    masks = [t.unsqueeze(0) if t.ndim == 0 else t for t in masks]
    labs = [torch.tensor([-100]) if t.ndim == 0 else t for t in labs]

    # Pad sequences
    input_ids_padded = pad_sequence(ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labs, batch_first=True, padding_value=-100)

    # Process teacher logits
    max_seq_len = labels_padded.size(1)
    first_logits = teacher_logits_list[0]
    
    if first_logits.ndim == 3 and first_logits.size(0) == 1:
        vocab_size = first_logits.size(2)
    elif first_logits.ndim == 2:
        vocab_size = first_logits.size(1)
    else:
        raise ValueError(f"Unexpected teacher_logits shape: {first_logits.shape}")

    teacher_batch = []
    for t_logits in teacher_logits_list:
        if t_logits.ndim == 3 and t_logits.size(0) == 1:
            t_logits = t_logits.squeeze(0)
        
        current_seq_len = t_logits.size(0)
        if current_seq_len >= max_seq_len:
            processed = t_logits[:max_seq_len, :]
        else:
            pad_size = max_seq_len - current_seq_len
            padding = torch.zeros(pad_size, vocab_size, dtype=t_logits.dtype)
            processed = torch.cat([t_logits, padding], dim=0)
        
        teacher_batch.append(processed)

    teacher_logits_padded = torch.stack(teacher_batch, dim=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'teacher_logits': teacher_logits_padded
    }
