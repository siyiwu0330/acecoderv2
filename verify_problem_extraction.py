#!/usr/bin/env python3
"""
Verify the correct way to extract transformed problems vs original problems
"""

import json
from pathlib import Path

def compare_problem_extraction(file_path):
    """Compare original vs transformed problem extraction"""
    
    print(f"Analyzing: {file_path}")
    print("=" * 80)
    
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        data = json.loads(line)
    
    # Original problem (from dataset)
    original_problem = data.get('problem', '')
    
    # Transformed problem (after step1 processing)
    transformed_problem = data.get('synthesis_result', {}).get('problem', '')
    
    print("🔵 ORIGINAL PROBLEM:")
    print(original_problem[:300] + "..." if len(original_problem) > 300 else original_problem)
    print("\n" + "=" * 80)
    
    print("🔴 TRANSFORMED PROBLEM:")
    print(transformed_problem[:300] + "..." if len(transformed_problem) > 300 else transformed_problem)
    print("\n" + "=" * 80)
    
    # Check test cases alignment
    tests = data.get('synthesis_result', {}).get('tests', [])
    print(f"📝 TEST CASES ({len(tests)} total):")
    if tests:
        for i, test in enumerate(tests[:3]):  # Show first 3 tests
            print(f"  {i+1}. {test}")
        if len(tests) > 3:
            print(f"  ... and {len(tests) - 3} more tests")
    
    print("\n" + "=" * 80)
    
    # Analysis
    print("📊 ANALYSIS:")
    print(f"- Original problem length: {len(original_problem)} chars")
    print(f"- Transformed problem length: {len(transformed_problem)} chars")
    print(f"- Problems are {'SAME' if original_problem.strip() == transformed_problem.strip() else 'DIFFERENT'}")
    
    # Check if tests mention function names that align with transformed problem
    if tests:
        test_functions = set()
        import re
        for test in tests:
            matches = re.findall(r'assert\s+(\w+)\(', test)
            test_functions.update(matches)
        
        print(f"- Test function names: {list(test_functions)}")
        
        # Check if function names appear in problems
        original_has_funcs = any(func in original_problem for func in test_functions)
        transformed_has_funcs = any(func in transformed_problem for func in test_functions)
        
        print(f"- Function names in original problem: {original_has_funcs}")
        print(f"- Function names in transformed problem: {transformed_has_funcs}")
    
    return {
        'original_problem': original_problem,
        'transformed_problem': transformed_problem,
        'tests': tests,
        'problems_match': original_problem.strip() == transformed_problem.strip()
    }

def main():
    # Test with acecoder_rounds_8 data
    test_files = [
        "outputs/acecoder_rounds_8/step1.1_parsing_round0.jsonl",
        "outputs/acecoder_rounds_8/step_3_filter_tests_gpt_4.1_mini_seed42_round0.jsonl"
    ]
    
    results = []
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\n🔍 Testing file: {file_path}")
            result = compare_problem_extraction(file_path)
            results.append(result)
            print("\n" + "🔹" * 50 + "\n")
        else:
            print(f"❌ File not found: {file_path}")
    
    # Summary
    print("\n📋 SUMMARY:")
    print("- We should use 'synthesis_result.problem' for the TRANSFORMED problem")
    print("- We should use 'synthesis_result.tests' for the corresponding test cases")
    print("- The original 'problem' field contains the dataset's original problem")
    print("- For AceCoderV2 experiments, the TRANSFORMED problem is what matters!")
    
    print("\n✅ RECOMMENDED CODE:")
    print("```python")
    print("# Use transformed problem (what we want)")
    print("problem_description = item.get('synthesis_result', {}).get('problem', '').strip()")
    print("")
    print("# Use corresponding test cases")
    print("test_cases = item.get('synthesis_result', {}).get('tests', [])")
    print("```")

if __name__ == "__main__":
    main()
