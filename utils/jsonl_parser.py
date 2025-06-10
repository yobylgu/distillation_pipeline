#!/usr/bin/env python3
"""
Quick script to inspect the structure of a JSONL dataset.
Prints the keys and sample values from the first few examples.
Can also output as JSON format with optional compression data replacement.
"""

import json
import argparse
import copy
import os
from datetime import datetime

def inspect_jsonl(path, num_examples=3, output_json=False, replace_logits=False):
    if not output_json:
        print(f"Inspecting: {path}\n")
    
    examples = []
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if not output_json:
                    print(f"Line {i+1} is not valid JSON: {e}")
                continue

            # Replace compressed logits if requested
            if replace_logits and 'compressed_logits' in obj:
                obj_copy = copy.deepcopy(obj)
                obj_copy['compressed_logits'] = "data_not_shown_compressed_logits"
                obj = obj_copy

            if output_json:
                examples.append(obj)
            else:
                print(f"--- Example {i+1} ---")
                for key, value in obj.items():
                    preview = str(value)
                    preview = preview.replace("\n", "\\n")
                    if len(preview) > 1500:
                        preview = preview[:1500] + "..."
                    print(f"{key}: {preview}")
                print()
    
    if output_json:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "json_parser_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.splitext(os.path.basename(path))[0]
        output_filename = f"{input_filename}_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write JSON to file
        with open(output_path, 'w') as output_file:
            json.dump(examples, output_file, indent=2)
        
        print(f"JSON output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the JSONL file to inspect")
    parser.add_argument("--n", type=int, default=30, help="Number of examples to print")
    parser.add_argument("--json", action="store_true", help="Output as JSON format instead of human-readable format")
    parser.add_argument("--replace-logits", action="store_true", help="Replace compressed_logits data with placeholder text")
    args = parser.parse_args()

    inspect_jsonl(args.file, args.n, args.json, args.replace_logits)


'''
python utils/jsonl_parser.py \
    --file data/codet5p-focal-methods/distillation_data_training.jsonl\
    --n 30 \
    --json \
    --replace-logits
'''