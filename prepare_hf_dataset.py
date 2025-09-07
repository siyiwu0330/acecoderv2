#!/usr/bin/env python3
"""
Script to prepare and upload acecoderv2 experiment results to Hugging Face
"""

import json
import os
import glob
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

def collect_all_jsonl_files(outputs_dir="outputs"):
    """Collect all JSONL files from outputs directory, excluding visualizations"""
    jsonl_files = []
    
    # Find all JSONL files recursively
    for jsonl_file in glob.glob(f"{outputs_dir}/**/*.jsonl", recursive=True):
        # Skip visualization files
        if "visualizations" not in jsonl_file:
            jsonl_files.append(jsonl_file)
    
    return sorted(jsonl_files)

def parse_jsonl_file(file_path):
    """Parse a JSONL file and return list of records with metadata"""
    records = []
    
    # Extract metadata from file path
    parts = Path(file_path).parts
    experiment_round = None
    step_type = None
    model_name = None
    
    # Extract experiment round (e.g., acecoder_rounds_6)
    for part in parts:
        if part.startswith("acecoder_rounds"):
            experiment_round = part
            break
    
    # Extract step type and model from filename
    filename = Path(file_path).stem
    if "step1" in filename:
        step_type = "step1_prompting"
    elif "step2.1" in filename:
        step_type = "step2_generation"
    elif "step2.2" in filename:
        step_type = "step2_evaluation"  
    elif "step_3" in filename:
        step_type = "step3_filtering"
    else:
        step_type = "unknown"
    
    # Extract model name
    if "gpt_4.1_mini" in filename:
        model_name = "gpt-4.1-mini"
    else:
        model_name = "unknown"
    
    # Read and parse JSONL
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Add metadata to each record
                        record = {
                            "source_file": str(file_path),
                            "experiment_round": experiment_round,
                            "step_type": step_type,
                            "model_name": model_name,
                            "line_number": line_num,
                            **data  # Include all original data
                        }
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {file_path} at line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return records

def normalize_record_structure(records):
    """Normalize all records to have the same structure"""
    # Define all possible fields that might exist
    common_fields = [
        'source_file', 'experiment_round', 'step_type', 'model_name', 
        'line_number', 'id', 'response', 'program'
    ]
    
    normalized_records = []
    for record in records:
        normalized_record = {}
        
        # Copy common fields
        for field in common_fields:
            normalized_record[field] = record.get(field, "")
        
        normalized_record['problem'] = record.get('problem', '')
        
        # Handle synthesis_result (always present but structure may vary)
        if 'synthesis_result' in record:
            normalized_record['synthesis_result'] = record['synthesis_result']
        else:
            normalized_record['synthesis_result'] = {}
        
        # Handle gen_result (only present in some steps)
        if 'gen_result' in record:
            normalized_record['gen_result'] = record['gen_result']
        else:
            normalized_record['gen_result'] = None
        
        normalized_records.append(normalized_record)
    
    return normalized_records

def create_dataset_splits(all_records):
    """Create dataset splits based on experiment rounds and steps"""
    dataset_dict = {}
    
    # Normalize all records first
    print("🔧 Normalizing record structures...")
    normalized_records = normalize_record_structure(all_records)
    
    # Group by experiment round and step type
    grouped_data = {}
    for record in normalized_records:
        key = f"{record['experiment_round']}_{record['step_type']}"
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(record)
    
    # Create splits
    for split_name, records in grouped_data.items():
        if records:  # Only create split if there are records
            dataset_dict[split_name] = Dataset.from_list(records)
            print(f"Created split '{split_name}' with {len(records)} records")
    
    return DatasetDict(dataset_dict)

def main():
    print("🚀 Preparing acecoderv2 dataset for Hugging Face upload...")
    
    # Step 1: Collect all JSONL files
    print("\n📁 Collecting JSONL files...")
    jsonl_files = collect_all_jsonl_files()
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Step 2: Parse all files
    print("\n📖 Parsing JSONL files...")
    all_records = []
    for file_path in jsonl_files:
        print(f"  Processing: {file_path}")
        records = parse_jsonl_file(file_path)
        all_records.extend(records)
        print(f"    Added {len(records)} records")
    
    print(f"\n✅ Total records collected: {len(all_records)}")
    
    # Step 3: Create dataset splits
    print("\n🔧 Creating dataset splits...")
    dataset_dict = create_dataset_splits(all_records)
    
    # Step 4: Save locally first
    print("\n💾 Saving dataset locally...")
    local_path = "acecoderv2_dataset"
    dataset_dict.save_to_disk(local_path)
    print(f"Dataset saved to {local_path}")
    
    # Step 5: Display dataset info
    print("\n📊 Dataset Information:")
    print(f"Number of splits: {len(dataset_dict)}")
    for split_name, dataset in dataset_dict.items():
        print(f"  {split_name}: {len(dataset)} examples")
        if len(dataset) > 0:
            print(f"    Features: {list(dataset.features.keys())}")
    
    print("\n🎉 Dataset preparation complete!")
    print("\nTo upload to Hugging Face, run:")
    print("python upload_to_hf.py")
    
    return dataset_dict

if __name__ == "__main__":
    dataset_dict = main()
