#!/usr/bin/env python3
"""
测试用例合并脚本
用于在测试用例生成轮次中，将新生成的测试用例与前一轮的程序合并
"""

import json
import argparse
from pathlib import Path

def merge_test_cases_with_programs(programs_file, test_cases_file, output_file):
    """
    将新生成的测试用例与现有程序合并
    
    Args:
        programs_file: 包含程序的文件路径（前一轮过滤结果）
        test_cases_file: 包含新测试用例的文件路径（当前轮生成结果）
        output_file: 合并后的输出文件路径
    """
    
    print(f"🔄 Merging test cases from {test_cases_file}")
    print(f"🔄 With programs from {programs_file}")
    print(f"🔄 Output to {output_file}")
    
    # 读取程序数据
    with open(programs_file, 'r') as f:
        programs_data = [json.loads(line) for line in f if line.strip()]
    
    # 读取测试用例数据
    with open(test_cases_file, 'r') as f:
        test_cases_data = [json.loads(line) for line in f if line.strip()]
    
    # 创建QID到测试用例的映射
    test_cases_by_qid = {}
    for item in test_cases_data:
        qid = item.get('gen_result', {}).get('qid')
        if qid:
            test_cases_by_qid[qid] = item.get('synthesis_result', {}).get('tests', [])
    
    print(f"📊 Found {len(programs_data)} programs and {len(test_cases_by_qid)} test case sets")
    
    # 合并数据
    merged_data = []
    for program_item in programs_data:
        qid = program_item.get('gen_result', {}).get('qid')
        
        if qid in test_cases_by_qid:
            # 合并测试用例
            original_tests = program_item.get('synthesis_result', {}).get('tests', [])
            new_tests = test_cases_by_qid[qid]
            
            # 去重合并
            combined_tests = list(original_tests)
            for test in new_tests:
                if test not in combined_tests:
                    combined_tests.append(test)
            
            # 更新测试用例
            program_item['synthesis_result']['tests'] = combined_tests
            
            print(f"🧪 Problem {qid[:8]}...: merged {len(original_tests)} + {len(new_tests)} → {len(combined_tests)} tests")
        else:
            print(f"⚠️  Problem {qid[:8]}...: no new test cases found, keeping {len(program_item.get('synthesis_result', {}).get('tests', []))} original tests")
        
        merged_data.append(program_item)
    
    # 保存合并结果
    with open(output_file, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Merged {len(merged_data)} problems to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Merge test cases with programs')
    parser.add_argument('programs_file', help='File containing programs (previous round filtered results)')
    parser.add_argument('test_cases_file', help='File containing new test cases (current round generation)')
    parser.add_argument('output_file', help='Output file for merged data')
    
    args = parser.parse_args()
    
    merge_test_cases_with_programs(
        Path(args.programs_file),
        Path(args.test_cases_file),
        Path(args.output_file)
    )

if __name__ == "__main__":
    main()



