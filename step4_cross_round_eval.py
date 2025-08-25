#!/usr/bin/env python3

"""
Step 4: Cross-Round Evaluation
Evaluate all programs from all rounds against all test cases from all rounds.
This creates the comprehensive matrix needed for complete visualization.
"""

import fire
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from code_eval import eval_codes

def load_visualization_history(history_file: Path) -> List[Dict]:
    """Load the visualization history data."""
    if not history_file.exists():
        raise FileNotFoundError(f"Visualization history file not found: {history_file}")
    
    history_data = []
    with open(history_file, 'r') as f:
        for line in f:
            history_data.append(json.loads(line))
    
    return history_data

def extract_all_programs_and_tests_with_dedup(history_data: List[Dict]) -> Dict:
    """Extract all programs and test cases from all rounds with improved unique ID tracking."""
    import hashlib
    
    # Group by QID first
    qid_data = {}
    for item in history_data:
        qid = item.get('gen_result', {}).get('qid')
        if qid:
            if qid not in qid_data:
                qid_data[qid] = []
            qid_data[qid].append(item)
    
    # Process each QID
    qid_results = {}
    for qid, items in qid_data.items():
        # Sort by round
        items.sort(key=lambda x: x.get('round', 0))
        
        # Track unique test cases by content hash with detailed tracking
        test_content_to_info = {}  # hash -> {id, content, first_round, appearances}
        unique_test_id = 1
        
        # Track unique programs by content hash with detailed tracking
        program_content_to_info = {}  # hash -> {id, content, first_round, appearances}
        unique_program_id = 1
        
        all_programs = []
        all_tests = []
        program_metadata = []
        test_metadata = []
        
        for item in items:
            round_num = item.get('round', 0)
            
            # Extract and track test cases (including duplicates in metadata)
            tests = item.get('synthesis_result', {}).get('tests', [])
            for i, test in enumerate(tests):
                # Create hash for test content
                test_hash = hashlib.md5(test.encode('utf-8')).hexdigest()
                
                if test_hash not in test_content_to_info:
                    # New unique test - add to global list
                    test_info = {
                        'unique_id': unique_test_id,
                        'content': test,
                        'first_round': round_num,
                        'hash': test_hash,
                        'appearances': [round_num]
                    }
                    test_content_to_info[test_hash] = test_info
                    unique_test_id += 1
                    
                    # Add to unique tests list
                    all_tests.append(test)
                    test_metadata.append({
                        'round': round_num,
                        'original_index': i,
                        'qid': qid,
                        'test_content': test,
                        'unique_id': test_info['unique_id'],
                        'first_round': test_info['first_round'],
                        'current_round': round_num,
                        'is_duplicate': False
                    })
                else:
                    # Duplicate test - track the appearance but don't add to global list again
                    test_content_to_info[test_hash]['appearances'].append(round_num)
                    # However, we still add metadata to track its presence in this round
                    test_metadata.append({
                        'round': round_num,
                        'original_index': i,
                        'qid': qid,
                        'test_content': test,
                        'unique_id': test_content_to_info[test_hash]['unique_id'],
                        'first_round': test_content_to_info[test_hash]['first_round'],
                        'current_round': round_num,
                        'is_duplicate': True
                    })
            
            # Extract and track programs (including duplicates in metadata)
            eval_results = item.get('gen_result', {}).get('eval_results', [])
            for i, eval_result in enumerate(eval_results):
                program = eval_result.get('program', '') or eval_result.get('parse_code', '')
                if program and len(program.strip()) > 0:
                    # Create hash for program content
                    program_hash = hashlib.md5(program.encode('utf-8')).hexdigest()
                    
                    if program_hash not in program_content_to_info:
                        # New unique program - add to global list
                        program_info = {
                            'unique_id': unique_program_id,
                            'content': program,
                            'first_round': round_num,
                            'hash': program_hash,
                            'appearances': [round_num]
                        }
                        program_content_to_info[program_hash] = program_info
                        unique_program_id += 1
                        
                        # Add to unique programs list
                        all_programs.append(program)
                        program_metadata.append({
                            'round': round_num,
                            'original_index': i,
                            'qid': qid,
                            'program_content': program,
                            'unique_id': program_info['unique_id'],
                            'first_round': program_info['first_round'],
                            'current_round': round_num,
                            'is_duplicate': False
                        })
                    else:
                        # Duplicate program - track the appearance but don't add to global list again
                        program_content_to_info[program_hash]['appearances'].append(round_num)
                        # However, we still add metadata to track its presence in this round
                        program_metadata.append({
                            'round': round_num,
                            'original_index': i,
                            'qid': qid,
                            'program_content': program,
                            'unique_id': program_content_to_info[program_hash]['unique_id'],
                            'first_round': program_content_to_info[program_hash]['first_round'],
                            'current_round': round_num,
                            'is_duplicate': True
                        })
        
        qid_results[qid] = {
            'programs': all_programs,
            'tests': all_tests,
            'program_metadata': program_metadata,
            'test_metadata': test_metadata,
            'test_dedup_info': test_content_to_info,
            'program_dedup_info': program_content_to_info
        }
    
    return qid_results

def evaluate_cross_round_for_qid(programs: List[str], tests: List[str], qid: str) -> List[List[int]]:
    """Evaluate all programs against all tests for a specific QID."""
    print(f"🔄 Cross-round evaluation for QID {qid[:20]}...")
    print(f"   {len(programs)} programs × {len(tests)} tests = {len(programs) * len(tests)} evaluations")
    
    if not programs or not tests:
        return []
    
    # Prepare evaluation data correctly for eval_codes
    # eval_codes expects: solution_strs=[prog1, prog2, ...], test_cases=[tests_for_prog1, tests_for_prog2, ...]
    # For cross-round eval, each program should be tested against ALL tests
    all_programs_eval = []
    all_tests_eval = []
    
    for program in programs:
        all_programs_eval.append(program)
        all_tests_eval.append(tests)  # Each program gets the full test list
    
    print(f"📊 Starting evaluation of {len(all_programs_eval)} programs against {len(tests)} tests each...")
    
    try:
        # Use eval_codes function - correct format now
        pass_rates, all_pass_rates = eval_codes(
            solution_strs=all_programs_eval,
            test_cases=all_tests_eval,
            return_test_cases_pass_status=True,
            num_processes=16
        )
        
        # Build matrix from results
        matrix = []
        
        for i, program in enumerate(programs):
            if i < len(all_pass_rates):
                test_results = all_pass_rates[i]  # Results for program i
                program_row = []
                
                if isinstance(test_results, list):
                    for j, test_result in enumerate(test_results):
                        if j < len(tests):
                            # Handle different result formats
                            if isinstance(test_result, dict):
                                pass_value = test_result.get('pass', False)
                            else:
                                pass_value = bool(test_result)
                            program_row.append(1 if pass_value else 0)
                        else:
                            break
                    
                    # Pad if necessary
                    while len(program_row) < len(tests):
                        program_row.append(-1)
                else:
                    # Fallback - use pass_rate for all tests
                    program_pass_rate = pass_rates[i] if i < len(pass_rates) else 0
                    program_row = [1 if program_pass_rate > 0.5 else 0] * len(tests)
            else:
                # Error case
                program_row = [-1] * len(tests)
            
            matrix.append(program_row)
        
        print(f"✅ Cross-round evaluation completed for QID {qid[:20]}")
        return matrix
        
    except Exception as e:
        print(f"❌ Error in cross-round evaluation for QID {qid[:20]}: {e}")
        import traceback
        traceback.print_exc()
        # Return error matrix
        return [[-1 for _ in range(len(tests))] for _ in range(len(programs))]

def main(
    output_dir: str,
    overwrite: bool = False,
    max_qids: Optional[int] = None
):
    """
    Main function to perform cross-round evaluation.
    
    Args:
        output_dir: Directory containing the pipeline outputs
        overwrite: Whether to overwrite existing cross-round evaluation results
        max_qids: Maximum number of QIDs to process (for testing)
    """
    
    output_dir = Path(output_dir)
    vis_dir = output_dir / "visualizations"
    history_file = vis_dir / "visualization_history.jsonl"
    
    # Output file for cross-round evaluation results
    cross_round_file = vis_dir / "cross_round_evaluation.jsonl"
    
    if cross_round_file.exists() and not overwrite:
        print(f"Cross-round evaluation already exists: {cross_round_file}")
        print("Use --overwrite to regenerate")
        return
    
    print(f"🚀 Starting Cross-Round Evaluation")
    print(f"📂 Input: {history_file}")
    print(f"📁 Output: {cross_round_file}")
    
    # Load visualization history
    try:
        history_data = load_visualization_history(history_file)
        print(f"📥 Loaded {len(history_data)} history items")
    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return
    
    # Extract programs and tests by QID with deduplication
    qid_results = extract_all_programs_and_tests_with_dedup(history_data)
    print(f"🎯 Found {len(qid_results)} QIDs to process")
    
    if max_qids:
        qid_list = list(qid_results.keys())[:max_qids]
        qid_results = {qid: qid_results[qid] for qid in qid_list}
        print(f"🔬 Limited to {len(qid_results)} QIDs for testing")
    
    # Process each QID
    cross_round_results = {}
    
    for qid, data in qid_results.items():
        programs = data['programs']
        tests = data['tests']  # Unique tests for evaluation
        program_metadata = data['program_metadata']
        test_metadata = data['test_metadata']
        
        print(f"\n🎯 Processing QID: {qid[:20]}...")
        print(f"   Programs: {len(programs)} (unique)")
        print(f"   Tests: {len(tests)} (unique)")
        print(f"   Program metadata: {len(program_metadata)} (all appearances)")
        print(f"   Test metadata: {len(test_metadata)} (all appearances)")
        
        if not programs or not tests:
            print(f"⚠️  Skipping QID {qid[:20]} - no programs or tests")
            continue
        
        # Perform cross-round evaluation using unique tests
        unique_matrix = evaluate_cross_round_for_qid(programs, tests, qid)
        
        if unique_matrix:
            # Expand matrix to match metadata dimensions
            # We need to map from unique tests to all test metadata entries
            matrix = []
            
            # Create mapping from unique test content to its index in unique_matrix
            test_content_to_unique_index = {}
            for i, test in enumerate(tests):
                test_content_to_unique_index[test] = i
            
            # For each program metadata entry
            for prog_idx, prog_meta in enumerate(program_metadata):
                if prog_idx < len(programs):
                    # Get the corresponding row from unique_matrix
                    prog_unique_idx = None
                    for i, program in enumerate(programs):
                        if program == prog_meta.get('program_content', ''):
                            prog_unique_idx = i
                            break
                    
                    if prog_unique_idx is not None and prog_unique_idx < len(unique_matrix):
                        # Create row for this program against all test metadata entries
                        prog_row = []
                        for test_idx, test_meta in enumerate(test_metadata):
                            test_content = test_meta.get('test_content', '')
                            if test_content in test_content_to_unique_index:
                                unique_test_idx = test_content_to_unique_index[test_content]
                                if unique_test_idx < len(unique_matrix[prog_unique_idx]):
                                    prog_row.append(unique_matrix[prog_unique_idx][unique_test_idx])
                                else:
                                    prog_row.append(-1)
                            else:
                                prog_row.append(-1)
                        matrix.append(prog_row)
                    else:
                        # Unknown program - fill with errors
                        matrix.append([-1] * len(test_metadata))
                else:
                    # Extra program metadata - fill with errors
                    matrix.append([-1] * len(test_metadata))
        else:
            matrix = None
        
        if matrix:
            # Calculate statistics based on the full matrix
            total_cells = len(matrix) * len(matrix[0]) if matrix else 0
            pass_count = sum(sum(1 for cell in row if cell == 1) for row in matrix)
            fail_count = sum(sum(1 for cell in row if cell == 0) for row in matrix)
            error_count = sum(sum(1 for cell in row if cell == -1) for row in matrix)
            
            print(f"   Matrix: {len(matrix)}×{len(matrix[0]) if matrix else 0}")
            print(f"   Results: {pass_count} pass, {fail_count} fail, {error_count} error")
            
            cross_round_results[qid] = {
                'qid': qid,
                'matrix': matrix,
                'programs': programs,
                'tests': tests,
                'program_metadata': data['program_metadata'],
                'test_metadata': data['test_metadata'],
                'statistics': {
                    'total_cells': total_cells,
                    'pass_count': pass_count,
                    'fail_count': fail_count,
                    'error_count': error_count,
                    'pass_rate': pass_count / total_cells if total_cells > 0 else 0,
                    'unique_programs': len(programs),
                    'unique_tests': len(tests),
                    'total_program_metadata': len(data['program_metadata']),
                    'total_test_metadata': len(data['test_metadata'])
                }
            }
    
    # Save results
    vis_dir.mkdir(exist_ok=True)
    with open(cross_round_file, 'w') as f:
        for qid, result in cross_round_results.items():
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Cross-round evaluation completed!")
    print(f"📊 Results saved to: {cross_round_file}")
    print(f"🎯 Processed {len(cross_round_results)} QIDs")

if __name__ == "__main__":
    fire.Fire(main)
