#!/usr/bin/env python3
"""
Prepare AceCoderV2 Round 8 results for Hugging Face upload
"""

import json
import os
import glob
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi
import argparse

def collect_round8_files(outputs_dir="outputs/acecoder_rounds_8"):
    """Collect all step3 filter results from round 8 (most complete data)"""
    step3_files = []
    
    # Find all step_3 filter results (final processed data with all rounds)
    pattern = f"{outputs_dir}/step_3_filter_tests_*.jsonl"
    for jsonl_file in glob.glob(pattern):
        step3_files.append(jsonl_file)
    
    return sorted(step3_files)

def parse_jsonl_file(file_path):
    """Parse JSONL file and add metadata"""
    records = []
    
    # Extract metadata from path
    path_obj = Path(file_path)
    filename = path_obj.stem
    
    # Extract experiment info
    experiment_round = "acecoder_rounds_8"
    step_type = "step3_filtering"
    
    # Extract model name
    if "gpt_4.1_mini" in filename:
        model_name = "gpt-4.1-mini"
    elif "gpt_4o" in filename:
        model_name = "gpt-4o" 
    else:
        model_name = "unknown"
    
    # Parse file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_number = 1
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    
                    # Add metadata
                    record['source_file'] = str(path_obj.relative_to(Path(file_path).parent.parent))
                    record['experiment_round'] = experiment_round
                    record['step_type'] = step_type
                    record['model_name'] = model_name
                    record['line_number'] = line_number
                    
                    records.append(record)
                    line_number += 1
                    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return records

def normalize_records(all_records):
    """Normalize record structure for consistency"""
    normalized = []
    
    for record in all_records:
        # Ensure synthesis_result is properly formatted
        synthesis_result = record.get('synthesis_result', {})
        if isinstance(synthesis_result, str):
            try:
                synthesis_result = json.loads(synthesis_result)
            except:
                synthesis_result = {}
        
        # Ensure gen_result is properly formatted  
        gen_result = record.get('gen_result', {})
        if isinstance(gen_result, str):
            try:
                gen_result = json.loads(gen_result)
            except:
                gen_result = {}
        
        # Create normalized record
        normalized_record = {
            'id': record.get('id'),
            'problem': record.get('problem', ''),
            'response': record.get('response', ''),
            'program': record.get('program', ''),
            'synthesis_result': synthesis_result,
            'gen_result': gen_result,
            'source_file': record.get('source_file', ''),
            'experiment_round': record.get('experiment_round', ''),
            'step_type': record.get('step_type', ''),
            'model_name': record.get('model_name', ''),
            'line_number': record.get('line_number', 0)
        }
        
        normalized.append(normalized_record)
    
    return normalized

def create_dataset_card():
    """Create README content for the dataset"""
    return """# AceCoderV2 Round 8 Dataset

## Overview
This dataset contains results from AceCoderV2 Round 8, an adversarial code generation system that iteratively generates and refines programming problems, solutions, and test cases across multiple rounds.

## Key Improvements in Round 8
- **Program Accumulation Fixed**: Programs now correctly accumulate across rounds
- **Enhanced Code Parsing**: Improved extraction of code from LLM responses  
- **Multi-language Support**: Better handling of Java, C++, and other languages
- **Robust Evolution**: Clear progression from 6→11→16→21 programs across rounds

## Dataset Statistics
- **Rounds**: 0-5 (6 rounds total)
- **Evolution**: 6 programs (Round 0) → 21 programs (Round 5)  
- **Test Cases**: 20 (Round 0) → 67 (Round 5)
- **Model**: GPT-4.1-mini
- **Step Type**: step3_filtering (final processed results)

## Round Evolution
| Round | Avg Programs | Avg Test Cases | Growth |
|-------|-------------|----------------|---------|
| 0     | 6           | 20             | Baseline |
| 1     | 11          | 35             | +83% |
| 2     | 16          | 45             | +167% |
| 3     | 16          | 58             | +167% |
| 4     | 19          | 62             | +217% |
| 5     | 21          | 67             | +250% |

## Data Fields
- `id`: Problem identifier
- `problem`: Problem description
- `synthesis_result.tests`: Test cases
- `gen_result.eval_results`: Evaluation metrics
- `program`: Generated solutions (accumulated)

## Usage
Use the provided converter script to extract specific formats:

```python
# Extract problems and test cases only
python hf_dataset_converter.py --dataset_name siyiwu0330/acecoderv2 --output simplified.jsonl

# Filter specific round
python hf_dataset_converter.py --dataset_name siyiwu0330/acecoderv2 --output round3.jsonl --round 3
```

## Citation
```bibtex
@dataset{acecoderv2_round8,
  title={AceCoderV2 Round 8: Adversarial Code Generation Evolution},
  author={Wu, Siyi},
  year={2024},
  url={https://huggingface.co/datasets/siyiwu0330/acecoderv2}
}
```
"""

def main():
    parser = argparse.ArgumentParser(description="Prepare Round 8 data for HF upload")
    parser.add_argument("--input_dir", default="outputs/acecoder_rounds_8", help="Input directory")
    parser.add_argument("--dataset_name", default="siyiwu0330/acecoderv2", help="HF dataset name")
    parser.add_argument("--upload", action="store_true", help="Upload to HF (requires token)")
    parser.add_argument("--output_file", default="acecoderv2_round8.jsonl", help="Local output file")
    
    args = parser.parse_args()
    
    print(f"🔍 Collecting Round 8 files from {args.input_dir}")
    
    # Collect all step3 files (most complete)
    jsonl_files = collect_round8_files(args.input_dir)
    
    if not jsonl_files:
        print(f"❌ No step3 files found in {args.input_dir}")
        return
    
    print(f"📄 Found {len(jsonl_files)} files:")
    for f in jsonl_files:
        print(f"   {f}")
    
    # Parse all files
    all_records = []
    for file_path in jsonl_files:
        print(f"📖 Parsing {file_path}")
        records = parse_jsonl_file(file_path)
        all_records.extend(records)
        print(f"   Added {len(records)} records")
    
    print(f"📊 Total records: {len(all_records)}")
    
    # Normalize records
    print("🔧 Normalizing records...")
    normalized_records = normalize_records(all_records)
    
    # Save local copy
    print(f"💾 Saving to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for record in normalized_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    if args.upload:
        print(f"🚀 Uploading to Hugging Face: {args.dataset_name}")
        
        # Create dataset
        dataset = Dataset.from_list(normalized_records)
        
        # Create README
        readme_content = create_dataset_card()
        
        try:
            # Push to hub
            dataset.push_to_hub(
                args.dataset_name,
                commit_message="Update with AceCoderV2 Round 8 results - Fixed program accumulation"
            )
            
            # Upload README separately
            api = HfApi()
            with open("README.md", "w", encoding='utf-8') as f:
                f.write(readme_content)
            
            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md", 
                repo_id=args.dataset_name,
                repo_type="dataset"
            )
            
            print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{args.dataset_name}")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print("Make sure you have HF token set: huggingface-cli login")
    
    print("\n🎉 Process completed!")
    print(f"   Records: {len(normalized_records)}")
    print(f"   Local file: {args.output_file}")
    if args.upload:
        print(f"   HF Dataset: https://huggingface.co/datasets/{args.dataset_name}")

if __name__ == "__main__":
    main()
