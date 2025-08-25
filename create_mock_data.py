#!/usr/bin/env python3

"""
Create mock 3-round data to test visualization.
Generate realistic but simple program/test combinations with clear patterns.
"""

import json
import os
from pathlib import Path
import hashlib

def create_mock_data():
    print("🎭 Creating Mock 3-Round Data for Visualization Testing")
    print("=" * 70)
    
    # Create output directory in standard location
    mock_dir = Path('outputs/mock_test_rounds')
    vis_dir = mock_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock QID
    mock_qid = "mock_test_qid_12345"
    
    # Define mock programs for each round
    programs_by_round = {
        0: [
            "def add(a, b): return a + b",
            "def add(x, y): return x + y",  # Same logic, different variable names
            "def add(a, b): return a + b + 0",  # Functionally equivalent 
            "def multiply(a, b): return a * b",
        ],
        1: [
            "def add(a, b): return a + b",  # Duplicate from round 0
            "def add(num1, num2): return num1 + num2",  # New variant
            "def multiply(a, b): return a * b",  # Duplicate from round 0
            "def subtract(a, b): return a - b",  # New function
            "def divide(a, b): return a / b if b != 0 else 0",  # New function
        ],
        2: [
            "def add(a, b): return a + b",  # Duplicate from round 0
            "def add_safe(a, b): return (a or 0) + (b or 0)",  # New safe version
            "def multiply(a, b): return a * b",  # Duplicate from round 0
            "def power(a, b): return a ** b",  # New function
        ]
    }
    
    # Define mock test cases for each round
    tests_by_round = {
        0: [
            "assert add(1, 2) == 3",
            "assert add(0, 0) == 0", 
            "assert add(-1, 1) == 0",
            "assert multiply(2, 3) == 6",
            "assert multiply(0, 5) == 0",
        ],
        1: [
            "assert add(1, 2) == 3",  # Duplicate from round 0
            "assert add(0, 0) == 0",  # Duplicate from round 0
            "assert add(10, -5) == 5",  # New test
            "assert subtract(5, 3) == 2",  # New test
            "assert divide(10, 2) == 5",  # New test
            "assert divide(1, 0) == 0",  # Edge case test
        ],
        2: [
            "assert add(1, 2) == 3",  # Duplicate from round 0
            "assert add_safe(None, 5) == 5",  # New test for safe function
            "assert power(2, 3) == 8",  # New test
            "assert power(5, 0) == 1",  # Edge case test
        ]
    }
    
    # Create evaluation results (programs evaluated against tests)
    def create_eval_results(programs, round_num):
        eval_results = []
        for i, program in enumerate(programs):
            # Simple logic: some programs pass, some fail
            status = "Success" if i % 2 == 0 else "Failed"
            eval_results.append({
                "program": program,
                "parse_code": program,
                "status": status,
                "exec_result": f"Mock execution result for round {round_num}, program {i}"
            })
        return eval_results
    
    # Create synthesis results (filtered test cases)
    def create_synthesis_results(tests, round_num):
        return {
            "tests": tests,
            "filtered_test_count": len(tests),
            "round": round_num
        }
    
    # Generate history data
    history_data = []
    
    for round_num in range(3):
        programs = programs_by_round[round_num]
        tests = tests_by_round[round_num]
        
        print(f"📊 Round {round_num}: {len(programs)} programs, {len(tests)} tests")
        
        # Create history item
        history_item = {
            "round": round_num,
            "gen_result": {
                "qid": mock_qid,
                "eval_results": create_eval_results(programs, round_num)
            },
            "synthesis_result": create_synthesis_results(tests, round_num),
            "timestamp": f"2024-01-{15+round_num:02d}T10:00:00Z"
        }
        
        history_data.append(history_item)
    
    # Save to visualization_history.jsonl
    history_file = vis_dir / 'visualization_history.jsonl'
    with open(history_file, 'w') as f:
        for item in history_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"💾 Saved {len(history_data)} history items to {history_file}")
    
    # Create expected results summary
    print(f"\n📈 Expected Deduplication Results:")
    
    # Count unique programs
    all_programs = []
    for round_progs in programs_by_round.values():
        all_programs.extend(round_progs)
    unique_programs = list(set(all_programs))
    print(f"   Total programs across rounds: {len(all_programs)}")
    print(f"   Unique programs: {len(unique_programs)}")
    
    # Count unique tests
    all_tests = []
    for round_tests in tests_by_round.values():
        all_tests.extend(round_tests)
    unique_tests = list(set(all_tests))
    print(f"   Total tests across rounds: {len(all_tests)}")
    print(f"   Unique tests: {len(unique_tests)}")
    
    print(f"\n🎯 Expected Matrix Size: {len(unique_programs)} × {len(unique_tests)}")
    
    # Show duplication patterns
    print(f"\n🔄 Duplication Patterns:")
    print(f"   'def add(a, b): return a + b' appears in rounds: 0, 1, 2")
    print(f"   'assert add(1, 2) == 3' appears in rounds: 0, 1, 2")
    print(f"   'def multiply(a, b): return a * b' appears in rounds: 0, 1, 2")
    
    return str(mock_dir)

if __name__ == "__main__":
    mock_dir = create_mock_data()
    print(f"\n✅ Mock data created in: {mock_dir}")
    print(f"🧪 Use this data to test visualization: python step4_cross_round_eval.py {mock_dir}")
