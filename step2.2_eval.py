import fire
import json
import random
import sys
from pathlib import Path
import re
import numpy as np
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from code_eval import eval_codes
from utils import print_statistics, get_python_code_from_string
from typing import List, Optional

FILE_NAME = Path(__file__).stem
LAST_STEP_NAME = "step2.1_gen"  # Also handle merged files
MERGED_STEP_NAME = "step2.1_merged"




def advanced_parse_code(solution_str: str, test_cases: List[str]) -> str:
    """
    Injects placeholder implementations for undefined classes and functions found in test cases.
    This version also detects method calls on placeholder classes.
    """
    if not isinstance(solution_str, str):
        solution_str = str(solution_str)

    all_tests_str = "\n".join(str(t) for t in test_cases)
    
    # Find class instantiations like `GridMaster(...)`
    class_instantiations = set(re.findall(r'\b([A-Z]\w*)\s*\(', all_tests_str))
    
    # Find function calls like `my_func(...)`
    function_calls = set(re.findall(r'\b([a-z_]\w*)\s*\(', all_tests_str))

    solution_words = set(re.findall(r'\b\w+\b', solution_str))
    common_keywords = {'assert', 'Solution', 'self', 'range', 'len', 'list', 'dict', 'set', 'tuple', 'print', 'int', 'str', 'float', 'bool', 'True', 'False', 'None', 'sorted', 'min', 'max', 'abs', 'sum'}
    
    placeholders = []
    
    # Handle class placeholders
    for class_name in class_instantiations:
        if class_name not in solution_words and class_name not in common_keywords:
            # Find all method calls on this class, e.g., master.canMove(...)
            methods = set(re.findall(r'\w+\.' + class_name + r'\(\w*\)\.(\w+)\(', all_tests_str))
            # A more general regex for obj.method() where obj might be an instance of our class
            methods.update(set(re.findall(r'\w+\.(\w+)\(', all_tests_str)))


            method_defs = "    def __init__(self, *args, **kwargs): pass\n"
            for method in methods:
                if method not in solution_words:
                     method_defs += f"    def {method}(self, *args, **kwargs): return None\n"

            placeholders.append(f"class {class_name}:\n{method_defs}")

    # Handle function placeholders
    for func_name in function_calls:
        if func_name not in solution_words and func_name not in common_keywords:
            placeholders.append(f"def {func_name}(*args, **kwargs): return None\n")

    # Remove duplicates
    unique_placeholders = list(dict.fromkeys(placeholders))
    injected_code = "\n".join(unique_placeholders) + "\n\n" + solution_str
    if unique_placeholders:
        print(f"--- Injected Code for QID ---")
        print("\n".join(unique_placeholders))
        print("----------------------------")
    return injected_code


def main(
    file_path: str,
    output_dir: str = None,
    overwrite: bool = False,
    num_proc: int = 64,
    max_samples: Optional[int] = None,
    current_round: int = 0,
    debug: bool = False,
):
    file_path = Path(file_path)
    if output_dir is None:
        output_dir = file_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dynamic output naming based on input
    input_name = file_path.stem
    if LAST_STEP_NAME in input_name:
        output_name = input_name.replace(LAST_STEP_NAME, FILE_NAME)
    elif MERGED_STEP_NAME in input_name:
        output_name = input_name.replace(MERGED_STEP_NAME, FILE_NAME)
    else:
        output_name = FILE_NAME + "_" + input_name
    
    output_file = output_dir / f"{output_name}.jsonl"
    stats_file = output_dir / f"{output_name}_stats.txt"
    
    if output_file.exists() and not overwrite:
        print(f"⚠️ Output file {output_file} already exists. Use --overwrite true to overwrite.")
        return
    
    with open(file_path, 'r') as f:
        items = [json.loads(line) for line in f.readlines()]
    
    if max_samples is not None:
        items = items[:max_samples]
    
    # Filter items that need evaluation
    items_to_process = []
    for item in items:
        programs_to_eval = []
        all_tests = []
        
        # Extract original program(s) - check both 'program' and 'all_programs' fields
        programs = item.get('program', [])
        if isinstance(programs, str):
            programs = [programs]
        elif programs is None:
            programs = []
        
        # Also check 'all_programs' field for accumulated programs from previous rounds
        all_programs = item.get('all_programs', [])
        if isinstance(all_programs, list):
            programs.extend(all_programs)
        
        # Remove duplicates while preserving order
        seen_programs = set()
        unique_programs = []
        for prog in programs:
            if prog and isinstance(prog, str) and prog.strip():
                prog_stripped = prog.strip()
                if prog_stripped not in seen_programs:
                    seen_programs.add(prog_stripped)
                    unique_programs.append(prog_stripped)
                    programs_to_eval.append(prog_stripped)
        
        # Extract original tests
        if 'synthesis_result' in item and 'tests' in item['synthesis_result']:
            original_tests = item['synthesis_result']['tests']
            if isinstance(original_tests, list):
                all_tests.extend(original_tests)
        
        # Extract new tests from step1.1 (if any)
        new_tests_step1_1 = []
        if 'gen_result' in item and 'outputs' in item['gen_result']:
            step1_1_outputs = item['gen_result']['outputs']
            for output in step1_1_outputs:
                if isinstance(output, dict) and 'message' in output and 'content' in output['message']:
                    test_content = output['message']['content']
                    # Extract test cases using improved regex
                    assert_statements = re.findall(r'assert\s+[^\n]+', test_content)
                    new_tests_step1_1.extend(assert_statements)
                elif isinstance(output, str):
                    # Handle string outputs directly
                    new_tests_step1_1.append(output)
        
        # Extract new tests from step2.1 generation (if any)
        new_tests_step2_1 = []
        if 'gen_result' in item and 'outputs' in item['gen_result']:
            step2_1_outputs = item['gen_result']['outputs']
            for output in step2_1_outputs:
                if isinstance(output, dict) and 'message' in output and 'content' in output['message']:
                    test_content = output['message']['content']
                    # Extract test cases using improved regex
                    assert_statements = re.findall(r'assert\s+[^\n]+', test_content)
                    new_tests_step2_1.extend(assert_statements)
                elif isinstance(output, str):
                    # Handle string outputs directly
                    new_tests_step2_1.append(output)
        
        # Include any newly generated programs from step2.1 generation
        generated_programs = item.get('gen_result', {}).get('generated_programs', [])
        if generated_programs:
            for gen_prog in generated_programs:
                # Extract the actual code from the response
                if isinstance(gen_prog, dict) and 'message' in gen_prog and 'content' in gen_prog['message']:
                    program_content = gen_prog['message']['content']
                    # Use improved code extraction
                    extracted_code = get_python_code_from_string(program_content)
                    if extracted_code and extracted_code.strip():
                        programs_to_eval.append(extracted_code)
                        print(f"✅ Successfully extracted {len(extracted_code)} chars of code")
                    else:
                        print(f"❌ Failed to extract code from: {program_content[:100]}...")
                elif isinstance(gen_prog, str):
                    # Handle string programs directly
                    extracted_code = get_python_code_from_string(gen_prog)
                    if extracted_code and extracted_code.strip():
                        programs_to_eval.append(extracted_code)
        
        # Sometimes the generated tests are strings of lists, e.g. "['assert...']". We need to parse them.
        parsed_new_tests = []
        for t in new_tests_step2_1 + new_tests_step1_1:
            if isinstance(t, str):
                # Handle string-encoded list
                if t.strip().startswith('[') and t.strip().endswith(']'):
                    try:
                        # A bit risky, but common for LLM outputs
                        parsed_list = eval(t)
                        if isinstance(parsed_list, list):
                            parsed_new_tests.extend(parsed_list)
                        else:
                            parsed_new_tests.append(t)
                    except:
                        parsed_new_tests.append(t) # Keep as is if parsing fails
                else:
                    parsed_new_tests.append(t)
            elif isinstance(t, list):
                parsed_new_tests.extend(t)
            else:
                parsed_new_tests.append(str(t))
        
        all_tests.extend(parsed_new_tests)
        
        # Only include items that have both programs and tests
        if programs_to_eval and all_tests:
            item['programs_to_eval'] = programs_to_eval
            item['all_tests'] = all_tests
            items_to_process.append(item)
    
    if not items_to_process:
        print("⚠️ No valid items to evaluate.")
        with open(stats_file, 'w') as f:
            f.write("No valid items to evaluate.\n")
        with open(output_file, 'w') as f:
            f.write("")
        return

    print(f"🔧 Processing {len(items_to_process)} solutions...")

    # Prepare lists for eval_codes - now we need to handle multiple programs per item
    all_parsed_codes = []
    all_test_cases = []
    item_program_mapping = []  # Track which programs belong to which items
    
    for item_idx, item in enumerate(items_to_process):
        programs = item['programs_to_eval']
        tests = item['all_tests']
        
        for program in programs:
            parsed_code = advanced_parse_code(program, tests)
            all_parsed_codes.append(parsed_code)
            all_test_cases.append(tests)
            item_program_mapping.append(item_idx)

    print("⚡ Evaluating codes...")
    pass_rates, test_cases_pass_status = eval_codes(
        solution_strs=all_parsed_codes,
        test_cases=all_test_cases,
        return_test_cases_pass_status=True,
        binary=False,
        num_processes=num_proc,
    )

    print(f'----------Processing evaluation results----------')
    
    # Group results back by item
    eval_idx = 0
    for item_idx, item in enumerate(items_to_process):
        num_programs = len(item['programs_to_eval'])
        
        # Collect all eval results for this item
        eval_results = []
        for prog_idx in range(num_programs):
            eval_result = {
                'pass_rate': pass_rates[eval_idx],
                'test_cases_pass_status': test_cases_pass_status[eval_idx],
                'parse_code': all_parsed_codes[eval_idx]
            }
            eval_results.append(eval_result)
            eval_idx += 1
        
        item['gen_result']['eval_results'] = eval_results

        # Calculate test case diversity matrix: shape (num_test_cases, num_programs)
        num_tests = len(item['all_tests'])
        test_case_diversity_arr = []
        
        for test_idx in range(num_tests):
            test_pass_row = []
            for prog_idx in range(num_programs):
                if test_idx < len(eval_results[prog_idx]['test_cases_pass_status']):
                    test_pass_row.append(eval_results[prog_idx]['test_cases_pass_status'][test_idx]['pass'])
                else:
                    test_pass_row.append(False)
            test_case_diversity_arr.append(test_pass_row)
        
        # Calculate mean pass rate for each test case across all programs
        mean_pass_rates = []
        for test_row in test_case_diversity_arr:
            mean_pass_rates.append(sum(test_row) / len(test_row) if test_row else 0.0)
        
        item['gen_result']['test_case_diversity'] = {
            "arr": test_case_diversity_arr,
            "mean": mean_pass_rates,
        }
        
        item['gen_result'].pop('outputs', None)  # Clean up original outputs to avoid confusion

    print(f"💾 Saving {len(items_to_process)} processed results to: {output_file}")
    with open(output_file, 'w') as f:
        for item in items_to_process:
            # 🔧 FIX: Store all evaluated programs for next round access
            # Extract all programs from eval_results to ensure persistence
            all_programs_in_eval = []
            eval_results = item.get('gen_result', {}).get('eval_results', [])
            for eval_result in eval_results:
                program_code = eval_result.get('parse_code', '')
                if program_code and program_code.strip():
                    all_programs_in_eval.append(program_code)
            
            # Store programs in both 'program' field and 'all_programs' for next round
            if all_programs_in_eval:
                item['program'] = all_programs_in_eval  # Update main program field
                item['all_programs'] = all_programs_in_eval  # Additional backup field
                problem_id = str(item.get('id', 'unknown'))[:8]
                print(f"💾 Saved {len(all_programs_in_eval)} programs for problem {problem_id}...")
            
            # Clean up temporary keys before saving
            item.pop('programs_to_eval', None)
            item.pop('all_tests', None)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print_statistics(items_to_process, output_file=stats_file)
    
    # --- Visualization History Logging ---
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    history_file = vis_dir / "visualization_history.jsonl"
    # Use the passed current_round parameter instead of inferring from path
    
    # To ensure we are logging the state of this round, we append the data.
    with open(history_file, 'a') as f:
        for item in items_to_process:
            # Create a serializable copy for logging
            history_item = json.loads(json.dumps(item))
            history_item['round'] = current_round
            # The 'programs' in the history should be all programs that were evaluated
            history_item['programs'] = history_item.get('programs_to_eval', [item.get('program')])
            # The 'tests' should be all tests used in this eval
            history_item['synthesis_result']['tests'] = history_item.get('all_tests', history_item.get('synthesis_result', {}).get('tests', []))
            f.write(json.dumps(history_item) + '\n')
    
    print(f"📊 Appended {len(items_to_process)} evaluation results for round {current_round} to {history_file}")
    # --- End of Visualization History Logging ---
    
    print(f" Results saved to {output_file}")
    
    
    
def get_round_from_path(file_path: str) -> int:
    """Extracts the round number from a file path using regex."""
    match = re.search(r'_round(\d+)', file_path)
    if match:
        return int(match.group(1))
    # Fallback if no round number in filename, useful for initial runs.
    return 0
    
if __name__ == "__main__":
    fire.Fire(main)

"""
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_Qwen2_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/gpt_4o_mini/step2.1_gen_gpt_4.1_mini_vllm_seed42_0_50.jsonl
python step2.2_eval.py outputs/Magicoder_Evol_Instruct_110K/o4_mini/step2.1_gen_Qwen3_8B_seed42.jsonl --overwrite True
"""

