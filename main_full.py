import os
import fire
import json
from pathlib import Path
from subprocess import run
from typing import List, Optional, Dict
from datasets import Dataset

def pretty_name(name: str) -> str:
    """Convert a name to filesystem-friendly format."""
    return name.replace("/", "_").replace("-", "_").replace(" ", "_")

def parse_incomplete_json(json_str: str) -> dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = json_str.replace("'", '"')
        return json.loads(json_str)
    
def parsing_item(item: dict) -> dict:
    ERROR_QUESTION = "Error in question generation"
    ERROR_TESTS = ["assert False"]
    
    gpt_response = item['synthesis_result'].get('gpt_response', {})
    
    if isinstance(gpt_response, dict):
        raw_text = gpt_response.get("message", {}).get("content", "")
    else:
        raw_text = str(gpt_response)
    
    try:
        obj = parse_incomplete_json(raw_text)
        question = obj.get("question", ERROR_QUESTION)
        tests = obj.get("tests", ERROR_TESTS)
    except Exception as e:
        print(f"Error parsing response: {e}")
        question = ERROR_QUESTION
        tests = ERROR_TESTS
    
    item['synthesis_result']['problem'] = question
    item['synthesis_result']['tests'] = tests
    return item

def run_python_file(script: str, args: List[str]):
    """
    Run a Python script using subprocess.
    This is simpler and more reliable than module importing.
    """
    import sys
    import os
    from subprocess import run
    
    # Construct the full path to the script (now in same directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # root directory
    script_path = os.path.join(script_dir, script)
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    print(f"\n[Running Script] {script} with args: {args}")
    
    # Build command
    cmd = [sys.executable, script_path] + args
    
    try:
        # Run the script
        result = run(cmd, check=True, cwd=script_dir)
        print(f"✅ Script {script} completed successfully")
    except Exception as e:
        print(f"❌ Script {script} failed with error: {e}")
        raise RuntimeError(f"Script {script} failed with error: {e}")

def main(
    output_dir: str = "outputs/acecoder_rounds",
    model_name: str = "gpt-4.1-mini",
    use_vllm: bool = False,
    overwrite: bool = False,
    rounds: int = 1,
    max_tokens: int = 8448,
    max_samples: int = 100,
    seed: int = 42,
    skip_step4: bool = False
):
    """
    Multi-round pipeline execution with consistent output folder structure.
    
    Args:
        output_dir: Output directory for results
        model_name: Model name for generation
        use_vllm: Whether to use VLLM for generation
        overwrite: Whether to overwrite existing files
        rounds: Number of rounds to run
        max_tokens: Maximum tokens for generation
        max_samples: Maximum samples to process
        seed: Random seed
        skip_step4: Whether to skip step4 (cross-round evaluation) - useful for large datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing pipeline run and warn user
    existing_files = list(output_dir.glob("step2.2_eval_*.jsonl"))
    if existing_files and not overwrite:
        print(f"⚠️  Warning: Found existing pipeline results in {output_dir}")
        print(f"   - Found {len(existing_files)} evaluation files")
        print(f"   - Use --overwrite True to replace existing results")
        print(f"   - Or use a different output directory to avoid conflicts")
        print(f"   - Continuing with existing files (some steps may be skipped)")
    elif existing_files and overwrite:
        print(f"🗑️  Overwrite mode: Cleaning up {len(existing_files)} existing files")
        for file in existing_files:
            try:
                file.unlink()
                print(f"   - Removed: {file.name}")
            except Exception as e:
                print(f"   - Failed to remove {file.name}: {e}")
    
    # Unified directory structure
    dataset_dir = output_dir / "Magicoder_Evol_Instruct_110K" / model_name.replace("-", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory structure:")
    print(f"- Base output dir: {output_dir}")
    print(f"- Dataset dir: {dataset_dir}")
    if overwrite:
        print(f"- Overwrite mode: ON (existing files will be replaced)")
    else:
        print(f"- Overwrite mode: OFF (will skip existing files)")

    previous_result_file = None
    
    for i in range(rounds):
        print(f"\n================= 🔁 Round {i + 1} / {rounds} =================", flush=True)

        # Unified file paths
        step1_output = dataset_dir / "step1_prompting_results.jsonl"
        step1_1_output = output_dir / f"step1.1_parsing_round{i}.jsonl"
        step2_1_output = output_dir / f"step2.1_gen_{model_name.replace('-', '_')}_seed{seed}_round{i}.jsonl"
        step2_2_output = output_dir / f"step2.2_eval_{model_name.replace('-', '_')}_seed{seed}_round{i}.jsonl"

        # === Step 1: prompting ===
        # Step1 should ONLY run in Round 0 to generate initial problems and test cases
        # All subsequent rounds work on these same problems, only generating new programs/test_cases
        if i == 0:
            # Round 0: Generate initial problems and test cases
            step1_generation_mode = "questions_and_tests"
            print(f"Round {i+1}: Generating initial problems and test cases")
            
            step1_args = [
                "--model_name", model_name,
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
                "--max_tokens", str(max_tokens),
                "--max_samples", str(max_samples),
                "--dataset_name", "ise-uiuc/Magicoder-Evol-Instruct-110K",
                "--generation_mode", step1_generation_mode,
                "--seed", str(seed),
            ]
            
            # Add API key if available
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                step1_args += ["--api_key", api_key]
            
            if previous_result_file:
                step1_args += ["--previous_result_file", str(previous_result_file)]
            
            run_python_file("step1_prompting.py", step1_args)
            
            # Parse the generated problems and test cases
            print(f"Round {i+1}: Parsing initial problems and test cases")
            run_python_file("step1.1_parsing.py", [
                "--file_path", str(step1_output),
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
                "--parsing_mode", step1_generation_mode,
            ])
        else:
            # Round 1+: Skip step1 entirely, use existing problems from Round 0
            print(f"Round {i+1}: Using existing problems from Round 0 (no new problem generation)")
            # Ensure we have the Round 0 parsing results available for step2
            round0_parsing = output_dir / "step1.1_parsing_round0.jsonl"
            if not round0_parsing.exists():
                raise FileNotFoundError(f"Round 0 parsing results not found: {round0_parsing}. Cannot proceed with Round {i+1}.")

        # Handle parsing output for Round 0 or verify Round 0 parsing exists for subsequent rounds
        if i == 0:
            # Verify Round 0 parsing was successful
            default_parsing_output = output_dir / "step1.1_parsing.jsonl"
            if default_parsing_output.exists():
                default_parsing_output.rename(step1_1_output)
            else:
                raise FileNotFoundError(f"Round 0 parsing output not found: {default_parsing_output}")
        else:
            # For Round 1+, use Round 0 parsing as the source of problems
            step1_1_output = output_dir / "step1.1_parsing_round0.jsonl"
            if not step1_1_output.exists():
                raise FileNotFoundError(f"Round 0 parsing results required for Round {i+1}: {step1_1_output}")

        # === Step 2.1: generation ===
        if use_vllm:
            run_python_file("step2.1_vllm_gen.py", [
                str(step1_1_output),
                "--model_name_or_path", model_name,
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
            ])
        else:
            # Determine generation mode based on round number
            # 🔧 FIX: Correct adversarial evolution logic
            # Round 0: Initialize problems + programs + test cases (complete setup)
            # Odd rounds (1,3,5...): Clean programs → Generate new programs → Eval
            # Even rounds (2,4,6...): Clean test cases → Generate new test cases → Eval
            if i == 0:
                generation_mode = "programs"  # Generate initial programs to go with problems+tests
                print(f"\n🚀 Round {i}: Complete initialization (problems + programs + test cases)")
            elif i % 2 == 1:
                generation_mode = "programs"  # Odd rounds: generate new programs
                print(f"\n🔄 Round {i}: Clean & generate new PROGRAMS (after filtering)")
            else:
                generation_mode = "test_cases"  # Even rounds: generate new test cases
                print(f"\n🔄 Round {i}: Clean & generate new TEST CASES (after filtering)")
            
            # 🔧 FIX: Use the correct input file for each round
            # Round 0: Use step1_1_output (initial parsing)
            # Round 1+: Use previous_result_file (previous round's filtered results)
            if i == 0:
                input_file = str(step1_1_output)
            else:
                input_file = str(previous_result_file) if previous_result_file else str(step1_1_output)
            
            step2_args = [
                input_file,
                "--model", model_name,
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite),
                "--generation_mode", generation_mode,
                "--seed", str(seed),
            ]
            
            # Add API key if available
            if api_key:
                step2_args += ["--api_key", api_key]
            
            run_python_file("step2.1_openai_gen.py", step2_args)

        # Handle generation output - step2.1_openai_gen.py generates without round info
        default_step2_1_output = output_dir / f"step2.1_gen_{model_name.replace('-', '_')}_seed{seed}.jsonl"
        if default_step2_1_output.exists():
            # Rename to include round information
            default_step2_1_output.rename(step2_1_output)
            print(f"✅ Renamed {default_step2_1_output.name} to {step2_1_output.name}")
        elif step2_1_output.exists():
            # File already exists with round info (maybe from previous run)
            print(f"✅ Using existing file: {step2_1_output.name}")
        else:
            raise FileNotFoundError(f"Generation output not found: {default_step2_1_output} or {step2_1_output}")

        # # === Step 2.2: evaluation ===
        # if step2_1_output.exists():
        #     # Debug: print first 3 lines of input
        #     print("\nDebug - step2.1 output sample:")
        #     with open(step2_1_output, 'r') as f:
        #         for i, line in enumerate(f):
        #             if i >= 3: break
        #             try:
        #                 data = json.loads(line)
        #                 print(f"Line {i+1}: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
        #             except json.JSONDecodeError:
        #                 print(f"Line {i+1}: (Invalid JSON) {line.strip()[:200]}...")

        #     filter_mode = "program" if i % 2 == 1 else "test_case"
        #     print(f"\nRound {i+1}: Using {filter_mode.upper()} filtering mode")

        #     run_python_file("step2.2_eval.py", [
        #         str(step2_1_output),
        #         "--output_dir", str(output_dir),
        #         "--overwrite", str(overwrite).lower(),
        #         "--max_samples", "2",
        #     ])

        #     # Verify evaluation output
        #     eval_output = output_dir / f"step2.2_eval_{model_name.replace('-', '_')}_seed42_round{i}.jsonl"
        #     if not eval_output.exists():
        #         raise FileNotFoundError(f"Evaluation output not found: {eval_output}")

        #     # === Step 3: filtering ===
        #     run_python_file("step_3_filter_tests.py", [
        #         str(eval_output),
        #         "--output_dir", str(output_dir),
        #         "--overwrite", str(overwrite).lower(),
        #         "--filter_mode", filter_mode,
        #         "--num_proc", "1"
        #     ])

        #     previous_result_file = eval_output
        # else:
        #     raise FileNotFoundError(f"Missing required input file: {step2_1_output}")


        # === Step 2.2: evaluation ===
        if step2_1_output.exists():
            # Debug: Show basic info about generation output (without dumping JSON)
            print(f"\n📊 Round {i+1} Generation Summary:")
            with open(step2_1_output, 'r') as f:
                lines = [line for line in f if line.strip()]
                print(f"   Generated data for {len(lines)} problems")
                if lines:
                    try:
                        sample = json.loads(lines[0])
                        qid = sample.get('gen_result', {}).get('qid', 'unknown')[:8]
                        outputs = len(sample.get('gen_result', {}).get('outputs', []))
                        print(f"   Sample problem {qid}...: {outputs} outputs")
                    except:
                        print(f"   Sample: (parsing error)")

            # Determine what to evaluate based on generation mode
            if generation_mode == "test_cases":
                # For test case generation rounds, merge new test cases with existing programs
                print(f"\n🔄 Round {i+1}: Merging new test cases with existing programs...")
                
                if not previous_result_file or not previous_result_file.exists():
                    raise FileNotFoundError(f"Previous round result file not found: {previous_result_file}")
                
                # Create merged file
                merged_output = output_dir / f"step2.1_merged_{model_name.replace('-', '_')}_seed{seed}_round{i}.jsonl"
                
                run_python_file("merge_test_cases.py", [
                    str(previous_result_file),  # programs from previous round
                    str(step2_1_output),        # test cases from current round
                    str(merged_output)          # merged output
                ])
                
                eval_input_file = merged_output
            else:
                # For program generation rounds, evaluate the newly generated programs
                eval_input_file = step2_1_output
                
            print(f"\n🔍 Round {i+1}: Evaluating programs against test cases...")
            print(f"📁 Using evaluation input: {eval_input_file}")
            
            run_python_file("step2.2_eval.py", [
                str(eval_input_file),
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite).lower(),
                "--max_samples", str(max_samples),
                "--current_round", str(i),
            ])

            # Verify evaluation output - derive name from actual evaluation input file
            eval_input_stem = eval_input_file.stem
            if "step2.1_merged" in eval_input_stem:
                expected_eval_stem = eval_input_stem.replace("step2.1_merged", "step2.2_eval")
            else:
                expected_eval_stem = eval_input_stem.replace("step2.1_gen", "step2.2_eval")
            eval_output = output_dir / f"{expected_eval_stem}.jsonl"
            if not eval_output.exists():
                raise FileNotFoundError(f"Evaluation output not found: {eval_output}")

            # === Step 3: filtering ===
            # Use auto mode for alternating filter logic based on round number
            print(f"\n🧹 Round {i+1}: Applying intelligent filtering...")
            
            run_python_file("step_3_filter_tests.py", [
                str(eval_output),
                "--output_dir", str(output_dir),
                "--overwrite", str(overwrite).lower(),
                "--filter_mode", "auto",  # Let the filter function decide based on round
                "--num_proc", "1"
            ])

            # Update the filtered results as the next round's input
            filtered_output = output_dir / f"step_3_filter_tests_{model_name.replace('-', '_')}_seed{seed}_round{i}.jsonl"
            if filtered_output.exists():
                previous_result_file = filtered_output
            else:
                previous_result_file = eval_output
                
        else:
            raise FileNotFoundError(f"Missing required input file: {step2_1_output}")

    # After all rounds are complete, generate the final visualization
    vis_dir = output_dir / "visualizations"
    history_file = vis_dir / "visualization_history.jsonl"
    
    if history_file.exists():
        print(f"\n================= 🎨 Pipeline Complete =================")
        print(f"📊 Visualization data ready in: {vis_dir}")
        
        # Run cross-round evaluation for comprehensive matrix (optional)
        if not skip_step4:
            print(f"\n🔄 Running cross-round evaluation for comprehensive matrix...")
            step4_args = [
                str(output_dir),
                "--overwrite"
            ]
            
            try:
                run_python_file("step4_cross_round_eval.py", step4_args)
                print(f"✅ Cross-round evaluation completed!")
            except Exception as e:
                print(f"⚠️ Cross-round evaluation failed: {e}")
                print(f"📝 Comprehensive matrix will show partial data only")
        else:
            print(f"\n⏭️ Skipping step4 (cross-round evaluation) as requested")
            print(f"📝 Comprehensive matrix will use real-time calculation")
        
        print(f"\n🎨 Use Gradio interface (integrated_gradio_app.py) to view interactive visualizations")
        print(f"\n🚀 To view results:")
        print(f"🌐 Run: python integrated_gradio_app.py")
        print(f"📱 Or: python app.py (simple interface)")
        
        print(f"✅ Pipeline completed! Visualization data is ready.")
        print(f"📊 Visualization files generated in: {history_file.parent}")
        print(f"💡 The visualization will be automatically updated in the integrated interface.")
            
    else:
        print(f"\n⚠️ History file not found, skipping final visualization.")

    print(f"\n🎉 All {rounds} rounds completed successfully!")
    print(f"📊 Check all outputs in: {output_dir}")
    print(f"🎨 Access interactive visualizer at:")
    print(f"   Local: http://localhost:7860")
    print(f"   Remote: http://YOUR_SERVER_IP:7860")
    print("-" * 60)

if __name__ == "__main__":
    fire.Fire(main)