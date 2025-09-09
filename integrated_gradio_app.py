#!/usr/bin/env python3
"""
🚀 Integrated Gradio Application for Adversarial Generation

This application provides a complete interface for:
1. Configuring and running the adversarial generation pipeline
2. Real-time monitoring of execution progress
3. Interactive visualization of results
4. Live feedback and logging display
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np
from pathlib import Path
import subprocess
import threading
import time
import queue
import re
import os
import sys
import signal
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class IntegratedAdvancedSystem:
    """Integrated system for adversarial generation with real-time monitoring."""
    
    def __init__(self):
        self.output_dir = Path("outputs/acecoder_rounds")  # Default output directory
        self.current_output_dir = None  # Track the actual output directory being used
        self.process = None
        self.log_queue = queue.Queue()
        self.is_running = False
        self.current_round = 0
        self.total_rounds = 1
        self.progress = 0.0
        self.data_cache = {}
        self.show_backend_logs = True  # Default to showing backend logs
        self.results_updated = False  # Flag to indicate new results available
    
    def set_backend_logging(self, enabled: bool):
        """Enable or disable backend terminal logging."""
        self.show_backend_logs = enabled
        if enabled:
            print("🖥️ Backend logging enabled - logs will show in terminal")
        else:
            print("🔇 Backend logging disabled - logs only in frontend")
        
    def run_pipeline(self, rounds: int, model_name: str, max_tokens: int, max_samples: int,
                    output_dir: str, overwrite: bool, seed: int = 42, skip_step4: bool = False, progress_callback=None):
        """Run the adversarial generation pipeline with real-time feedback."""
        
        self.is_running = True
        # Track the current output directory for visualization loading
        self.current_output_dir = Path(output_dir)
        self.current_round = 0
        self.total_rounds = rounds
        self.progress = 0.0
        
        # Clear previous logs
        while not self.log_queue.empty():
            self.log_queue.get()
        
        def log_output(message, show_in_terminal=True):
            """Log to both frontend queue and backend terminal."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}"
            
            # Always add to frontend queue
            self.log_queue.put(formatted_msg)
            
            # Also show in backend terminal if enabled
            if show_in_terminal and self.show_backend_logs:
                print(formatted_msg, flush=True)
            
            if progress_callback:
                progress_callback()
        
        def run_command():
            try:
                log_output("🚀 Starting adversarial generation pipeline...")
                log_output(f"📊 Configuration: {rounds} rounds, model: {model_name}, max_samples: {max_samples}")
                
                # Build command
                cmd = [
                    sys.executable, "main.py",
                    "--output_dir", output_dir,
                    "--model_name", model_name,
                    "--rounds", str(rounds),
                    "--max_tokens", str(max_tokens),
                    "--max_samples", str(max_samples),
                    "--overwrite", str(overwrite).lower(),
                    "--seed", str(int(seed))
                ]
                
                # Add skip_step4 parameter if enabled
                if skip_step4:
                    cmd.append("--skip_step4")
                    cmd.append("True")
                
                log_output(f"🔧 Command: {' '.join(cmd)}")
                
                # Set up environment with API keys
                env = os.environ.copy()
                # API key is handled in frontend interface, no warning needed
                
                # Start process with inherited environment and process group
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=0,  # Unbuffered for real-time output
                    env=env,
                    cwd=os.getcwd(),  # Use current working directory (works for both local and Docker)
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
                
                # Monitor output with real-time processing
                for line in iter(self.process.stdout.readline, ''):
                    if line.strip():
                        stripped_line = line.strip()
                        
                        # Parse round information - more flexible pattern
                        round_match = re.search(r'Round (\d+)\s*/\s*(\d+)', stripped_line)
                        if round_match:
                            self.current_round = int(round_match.group(1))
                            self.total_rounds = int(round_match.group(2))
                            # Calculate progress: (completed_rounds / total_rounds) * 100
                            # current_round is 1-based, so current_round-1 represents completed rounds
                            completed_rounds = max(0, self.current_round - 1)
                            self.progress = (completed_rounds / self.total_rounds) * 100
                            log_output(f"🔄 Progress: Round {self.current_round}/{self.total_rounds} ({self.progress:.1f}%)")
                        
                        # Parse step information
                        if "Step1" in stripped_line:
                            log_output("📝 Step 1: Generating prompts...")
                        elif "Step2.1" in stripped_line:
                            log_output("🤖 Step 2.1: Generating content...")
                        elif "Step2.2" in stripped_line:
                            log_output("📊 Step 2.2: Evaluating results...")
                        elif "Step 3" in stripped_line:
                            log_output("🧹 Step 3: Filtering and processing...")
                        elif "Gradio" in stripped_line:
                            log_output("🎨 Launching visualization interface...")
                        
                        # Output the original line
                        log_output(stripped_line)
                        
                        # Flush logs immediately for real-time updates
                        sys.stdout.flush()
                
                self.process.wait()
                
                if self.process.returncode == 0:
                    log_output("✅ Pipeline completed successfully!")
                    self.progress = 100.0
                    log_output("📊 Loading visualization results...")
                    self.load_results()
                    self.results_updated = True  # Mark that new results are available
                    log_output("🎨 Visualization data ready! Check the Visualization tab for updated results.")
                else:
                    log_output(f"❌ Pipeline failed with return code: {self.process.returncode}")
                
            except Exception as e:
                log_output(f"❌ Error running pipeline: {str(e)}")
            finally:
                self.is_running = False
                self.process = None
        
        # Run in separate thread
        thread = threading.Thread(target=run_command)
        thread.daemon = True
        thread.start()
        
        return thread
    
    def stop_pipeline(self):
        """Stop the running pipeline forcefully."""
        if self.process:
            try:
                self.log_queue.put("[SYSTEM] ⛔ Stopping pipeline...")
                
                # Try to terminate the entire process group first
                try:
                    if hasattr(os, 'killpg'):
                        # Kill the entire process group
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                        
                        # Wait for graceful shutdown
                        try:
                            self.process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            # Force kill the process group
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                            self.process.wait()
                    else:
                        # Fallback for systems without killpg
                        raise AttributeError("killpg not available")
                        
                except (ProcessLookupError, AttributeError, OSError):
                    # Fallback to individual process termination with psutil
                    try:
                        import psutil
                        parent = psutil.Process(self.process.pid)
                        children = parent.children(recursive=True)
                        
                        # Terminate children first, then parent
                        for child in children:
                            try:
                                child.terminate()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        self.process.terminate()
                        
                        # Wait for graceful shutdown
                        try:
                            self.process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            # Force kill if graceful termination fails
                            for child in children:
                                try:
                                    child.kill()
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            self.process.kill()
                            self.process.wait()
                            
                    except ImportError:
                        # Final fallback to simple termination
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait()
                
                self.log_queue.put("[SYSTEM] ✅ Pipeline successfully stopped")
            except Exception as e:
                self.log_queue.put(f"[SYSTEM] ❌ Error stopping pipeline: {e}")
            finally:
                self.process = None
                self.is_running = False
                self.current_round = 0
                self.total_rounds = 1
                self.progress = 0.0
    
    def get_logs(self) -> str:
        """Get accumulated logs for display."""
        new_logs = []
        while not self.log_queue.empty():
            new_logs.append(self.log_queue.get())
        return "\n".join(new_logs) if new_logs else ""
    
    def get_all_logs(self) -> str:
        """Get all logs accumulated so far."""
        # Store logs in a persistent list
        if not hasattr(self, '_all_logs'):
            self._all_logs = []
        
        # Add new logs to the persistent list
        while not self.log_queue.empty():
            self._all_logs.append(self.log_queue.get())
        
        # Return recent logs (last 50 lines to prevent overflow)
        recent_logs = self._all_logs[-50:] if len(self._all_logs) > 50 else self._all_logs
        return "\n".join(recent_logs)
    
    def get_status(self) -> Dict:
        """Get current execution status."""
        return {
            "is_running": self.is_running,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "progress": self.progress,
            "status_text": f"Round {self.current_round}/{self.total_rounds}" if self.is_running else "Ready"
        }
    
    def set_visualization_directory(self, viz_dir: str):
        """Manually set the directory to load visualization data from."""
        self.current_output_dir = Path(viz_dir)
        print(f"📂 Visualization directory set to: {viz_dir}")
    
    def load_results(self):
        """Load visualization results after pipeline completion."""
        # Use the current output directory if available, otherwise fall back to default
        active_output_dir = self.current_output_dir if self.current_output_dir else self.output_dir
        history_file = active_output_dir / "visualizations" / "visualization_history.jsonl"
        
        print(f"🔍 Loading visualization data from: {history_file}")
        
        if not history_file.exists():
            self.data_cache = {"error": f"No results found in {active_output_dir}"}
            return
        
        try:
            history_data = []
            with open(history_file, 'r') as f:
                for line in f:
                    history_data.append(json.loads(line))
            
            # Group by QID
            self.data_cache = {}
            for item in history_data:
                qid = item.get('gen_result', {}).get('qid')
                if qid:
                    if qid not in self.data_cache:
                        self.data_cache[qid] = []
                    self.data_cache[qid].append(item)
            
            # Sort by round number
            for qid in self.data_cache:
                self.data_cache[qid].sort(key=lambda x: x.get('round', 0))
                
        except Exception as e:
            self.data_cache = {"error": f"Error loading results: {str(e)}"}
    
    def get_qid_list(self) -> List[str]:
        """Get list of available QIDs, prioritizing those with multiple rounds."""
        if "error" in self.data_cache:
            return []
        
        # Separate QIDs by number of rounds (prioritize multi-round QIDs)
        multi_round_qids = []
        single_round_qids = []
        
        for qid, data in self.data_cache.items():
            rounds = len(set(item.get('round', 0) for item in data))
            if rounds > 1:
                multi_round_qids.append(qid)
            else:
                single_round_qids.append(qid)
        
        # Return multi-round QIDs first, then single-round ones
        return sorted(multi_round_qids) + sorted(single_round_qids)
    
    def create_program_test_matrix(self, qid: str, round_num: int) -> go.Figure:
        """Create a program-test matrix for a specific round: rows=programs, columns=tests."""
        if not qid or qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="No data available - run pipeline first", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        rounds_data = self.data_cache[qid]
        if not rounds_data:
            fig = go.Figure()
            fig.add_annotation(text="No rounds data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Find data for the specific round
        round_data = None
        for item in rounds_data:
            if item.get('round', 0) == round_num:
                round_data = item
                break
        
        if not round_data:
            fig = go.Figure()
            fig.add_annotation(text=f"No data available for Round {round_num}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Get tests and evaluation results
        tests = round_data.get('synthesis_result', {}).get('tests', [])
        eval_results = round_data.get('gen_result', {}).get('eval_results', [])
        programs = round_data.get('programs', [round_data.get('program', '')])
        
        if not tests:
            fig = go.Figure()
            fig.add_annotation(text="No test cases found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        if not eval_results:
            fig = go.Figure()
            fig.add_annotation(text="No evaluation results found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Build matrix: rows = programs, columns = tests
        matrix = []
        program_labels = []
        test_labels = [f"R{round_num}_T{i+1}" for i in range(len(tests))]
        hover_text = []
        
        # Create matrix for each program
        for prog_idx, eval_result in enumerate(eval_results):
            program_labels.append(f"R{round_num}_P{prog_idx + 1}")
            row = []
            hover_row = []
            test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
            
            for test_idx in range(len(tests)):
                if test_idx < len(test_cases_pass_status):
                    test_result = test_cases_pass_status[test_idx]
                    if isinstance(test_result, dict):
                        passed = 1 if test_result.get('pass', False) else 0
                        error_msg = test_result.get('error_message', '')
                    else:
                        passed = 1 if test_result else 0
                        error_msg = ''
                    
                    row.append(passed)
                    status_text = "PASS" if passed else "FAIL"
                    error_display = (error_msg[:100] + "...") if error_msg else "No error message"
                    hover_row.append(f"R{round_num}_P{prog_idx+1}<br>R{round_num}_T{test_idx+1}: {status_text}<br>{error_display}")
                else:
                    row.append(-1)  # Missing test result
                    hover_row.append(f"R{round_num}_P{prog_idx+1}<br>R{round_num}_T{test_idx+1}: N/A")
            
            matrix.append(row)
            hover_text.append(hover_row)
        
        # Handle case where no programs were evaluated
        if not matrix:
            program_labels.append(f"R{round_num}_P1")
            row = [-1] * len(tests)
            hover_row = [f"R{round_num}_P1<br>R{round_num}_T{i+1}: N/A" for i in range(len(tests))]
            matrix.append(row)
            hover_text.append(hover_row)
        
        # Create custom colorscale with clear color mapping
        colorscale = [
            [0.0, '#FF0000'],    # Red for FAIL (0)
            [0.5, '#CCCCCC'],    # Gray for N/A (-1)  
            [1.0, '#00FF00']     # Green for PASS (1)
        ]
        
        # Convert matrix values for display with direct mapping
        display_matrix = []
        for row in matrix:
            display_row = []
            for val in row:
                if val == -1:
                    display_row.append(-1)   # Gray (N/A)
                elif val == 0:
                    display_row.append(0)    # Red (FAIL)
                else:
                    display_row.append(1)    # Green (PASS)
            display_matrix.append(display_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=display_matrix,
            x=test_labels,
            y=program_labels,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale=colorscale,
            zmid=0,  # Center the colorscale at 0
            showscale=False,
            xgap=2,
            ygap=2
        ))
        
        fig.update_layout(
            title=f"Program-Test Matrix - Round {round_num} (QID: {qid[:8]}...)",
            xaxis_title="Test Cases",
            yaxis_title="Programs", 
            width=800,
            height=400,
            font=dict(size=12),
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(side="top")
        fig.update_yaxes(autorange="reversed")
        
        return fig
        
    def extract_all_programs_and_tests(self, qid: str) -> Dict:
        """Extract all programs and test cases from all rounds for comprehensive matrix."""
        if not qid or qid not in self.data_cache:
            return {"error": "No data available"}
        
        rounds_data = self.data_cache[qid]
        all_programs = []
        all_tests = []
        program_round_map = []  # Track which round each program comes from
        test_round_map = []     # Track which round each test comes from
        
        # Sort by round
        rounds_data.sort(key=lambda x: x.get('round', 0))
        
        for round_data in rounds_data:
            round_num = round_data.get('round', 0)
            
            # Extract tests for this round
            tests = round_data.get('synthesis_result', {}).get('tests', [])
            for i, test in enumerate(tests):
                all_tests.append(test)
                test_round_map.append(round_num)
            
            # Extract programs for this round
            eval_results = round_data.get('gen_result', {}).get('eval_results', [])
            for i, eval_result in enumerate(eval_results):
                # Try multiple fields where program code might be stored
                program = eval_result.get('program', '') or eval_result.get('parse_code', '')
                if program and len(program.strip()) > 0:
                    all_programs.append(program)
                    program_round_map.append(round_num)
        
        return {
            "programs": all_programs,
            "tests": all_tests,
            "program_round_map": program_round_map,
            "test_round_map": test_round_map,
            "error": None
        }
    
    def evaluate_cross_round(self, programs: List[str], tests: List[str]) -> List[List[int]]:
        """Evaluate all programs against all tests using the existing eval_codes function."""
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from code_eval import eval_codes
        
        print(f"🔄 Evaluating {len(programs)} programs against {len(tests)} tests...")
        
        # Prepare evaluation data correctly for eval_codes
        # eval_codes expects: solution_strs=[prog1, prog2, ...], test_cases=[tests_for_prog1, tests_for_prog2, ...]
        # For cross-round eval, each program should be tested against ALL tests
        all_programs = []
        all_tests = []
        
        for program in programs:
            all_programs.append(program)
            all_tests.append(tests)  # Each program gets the full test list
        
        print(f"📊 Starting evaluation of {len(all_programs)} programs against {len(tests)} tests each...")
        
        try:
            # Use eval_codes with return_test_cases_pass_status=True to get detailed results
            pass_rates, all_pass_rates = eval_codes(
                solution_strs=all_programs,
                test_cases=all_tests,
                return_test_cases_pass_status=True,
                num_processes=16  # Reduce to avoid overwhelming
            )
            
            # Build matrix from results - each program tested against all tests
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
            
            print(f"\n✅ Cross-round evaluation completed!")
            return matrix
            
        except Exception as e:
            print(f"\n❌ Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error matrix
            matrix = [[-1 for _ in range(len(tests))] for _ in range(len(programs))]
            return matrix

    def load_cross_round_results(self) -> Dict:
        """Load cross-round evaluation results if available."""
        active_output_dir = self.current_output_dir if self.current_output_dir else self.output_dir
        cross_round_file = active_output_dir / "visualizations" / "cross_round_evaluation.jsonl"
        
        if not cross_round_file.exists():
            return {}
        
        try:
            cross_round_data = {}
            with open(cross_round_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = data.get('qid')
                    if qid:
                        cross_round_data[qid] = data
            return cross_round_data
        except Exception as e:
            print(f"⚠️ Error loading cross-round results: {e}")
            return {}

    def create_comprehensive_matrix(self, qid: str) -> go.Figure:
        """Create comprehensive matrix using cross-round evaluation results if available."""
        
        # Try to load cross-round evaluation results first
        cross_round_data = self.load_cross_round_results()
        
        if qid in cross_round_data:
            # Use cross-round evaluation results
            cross_data = cross_round_data[qid]
            matrix = cross_data.get('matrix', [])
            programs = cross_data.get('programs', [])
            tests = cross_data.get('tests', [])
            program_metadata = cross_data.get('program_metadata', [])
            test_metadata = cross_data.get('test_metadata', [])
            
            if not matrix or not programs or not tests:
                fig = go.Figure()
                fig.add_annotation(text="Invalid cross-round evaluation data", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, 
                                 showarrow=False, font_size=16)
                return fig
            
            # Extract round and unique ID information from metadata
            # Use 'current_round' to show when the item appears, not 'first_round'
            program_round_map = [meta.get('current_round', meta.get('first_round', meta.get('round', 0))) for meta in program_metadata]
            test_round_map = [meta.get('current_round', meta.get('first_round', meta.get('round', 0))) for meta in test_metadata]
            program_unique_ids = [meta.get('unique_id', i+1) for i, meta in enumerate(program_metadata)]
            test_unique_ids = [meta.get('unique_id', i+1) for i, meta in enumerate(test_metadata)]
            
            num_programs = len(programs)
            num_tests = len(tests)
            
            print(f"🔥 Using cross-round evaluation results: {num_programs}×{num_tests} matrix")
            
            # Filter out empty rows/columns AND rows/columns with 0% success rate
            valid_program_indices = []
            valid_test_indices = []
            
            # Check for meaningful rows (programs that have at least one PASS)
            for i in range(num_programs):
                has_data = any(matrix[i][j] != -1 for j in range(num_tests))  # Has some data
                has_pass = any(matrix[i][j] == 1 for j in range(num_tests))   # Has at least one pass
                if has_data and has_pass:
                    valid_program_indices.append(i)
            
            # Check for meaningful columns (tests that have at least one PASS)  
            for j in range(num_tests):
                has_data = any(matrix[i][j] != -1 for i in range(num_programs))  # Has some data
                has_pass = any(matrix[i][j] == 1 for i in range(num_programs))   # Has at least one pass
                if has_data and has_pass:
                    valid_test_indices.append(j)
            
            # If too aggressive filtering results in empty matrix, fall back to less strict filtering
            if not valid_program_indices or not valid_test_indices:
                print("⚠️ Aggressive filtering removed all rows/columns, falling back to basic filtering")
                valid_program_indices = []
                valid_test_indices = []
                
                # Fallback: Keep rows/columns that have at least some data (even if all fail)
                for i in range(num_programs):
                    if any(matrix[i][j] != -1 for j in range(num_tests)):
                        valid_program_indices.append(i)
                
                for j in range(num_tests):
                    if any(matrix[i][j] != -1 for i in range(num_programs)):
                        valid_test_indices.append(j)
            
            # Filter matrix, metadata, and labels
            if valid_program_indices and valid_test_indices:
                # Filter matrix
                filtered_matrix = []
                for i in valid_program_indices:
                    row = []
                    for j in valid_test_indices:
                        row.append(matrix[i][j])
                    filtered_matrix.append(row)
                
                # Filter metadata and labels
                filtered_programs = [programs[i] for i in valid_program_indices]
                filtered_tests = [tests[j] for j in valid_test_indices]
                filtered_program_metadata = [program_metadata[i] for i in valid_program_indices]
                filtered_test_metadata = [test_metadata[j] for j in valid_test_indices]
                filtered_program_round_map = [program_round_map[i] for i in valid_program_indices]
                filtered_test_round_map = [test_round_map[j] for j in valid_test_indices]
                filtered_program_unique_ids = [program_unique_ids[i] for i in valid_program_indices]
                filtered_test_unique_ids = [test_unique_ids[j] for j in valid_test_indices]
                
                # Update variables to use filtered data
                matrix = filtered_matrix
                programs = filtered_programs
                tests = filtered_tests
                program_metadata = filtered_program_metadata
                test_metadata = filtered_test_metadata
                program_round_map = filtered_program_round_map
                test_round_map = filtered_test_round_map
                program_unique_ids = filtered_program_unique_ids
                test_unique_ids = filtered_test_unique_ids
                num_programs = len(filtered_programs)
                num_tests = len(filtered_tests)
                
                print(f"📊 After filtering empty rows/cols: {num_programs}×{num_tests} matrix")
            else:
                print("⚠️ All rows or columns are empty - no valid data to display")
            
        else:
            # Fallback to partial matrix using existing data
            if not qid or qid not in self.data_cache:
                fig = go.Figure()
                fig.add_annotation(text="No data available - run pipeline first", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, 
                                 showarrow=False, font_size=16)
                return fig

            rounds_data = self.data_cache[qid]
            
            # Extract data for comprehensive display
            data = self.extract_all_programs_and_tests(qid)
            programs = data["programs"]
            tests = data["tests"]
            program_round_map = data["program_round_map"]
            test_round_map = data["test_round_map"]
            
            if not programs or not tests:
                fig = go.Figure()
                fig.add_annotation(text="No programs or tests found", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, 
                                 showarrow=False, font_size=16)
                return fig
            
            # Create partial matrix with existing results where available
            num_programs = len(programs)
            num_tests = len(tests)
            matrix = [[-1 for _ in range(num_tests)] for _ in range(num_programs)]
            
            print(f"⚠️ Using fallback partial matrix: {num_programs}×{num_tests}")
            
            # Fill in available results from same-round evaluations
            # Use proper mapping based on round information
            for round_data in rounds_data:
                round_num = round_data.get('round', 0)
                round_tests = round_data.get('synthesis_result', {}).get('tests', [])
                eval_results = round_data.get('gen_result', {}).get('eval_results', [])
                
                # Find the range of programs and tests for this round
                prog_start_idx = None
                prog_end_idx = None
                test_start_idx = None
                test_end_idx = None
                
                # Find program range for this round
                for i, round_map in enumerate(program_round_map):
                    if round_map == round_num:
                        if prog_start_idx is None:
                            prog_start_idx = i
                        prog_end_idx = i + 1
                
                # Find test range for this round
                for i, round_map in enumerate(test_round_map):
                    if round_map == round_num:
                        if test_start_idx is None:
                            test_start_idx = i
                        test_end_idx = i + 1
                
                # Skip if no valid ranges found
                if prog_start_idx is None or test_start_idx is None:
                    continue
                
                # Fill in the evaluation results for this round
                for prog_offset, eval_result in enumerate(eval_results):
                    prog_idx = prog_start_idx + prog_offset
                    if prog_idx >= prog_end_idx or prog_idx >= num_programs:
                        continue
                        
                    test_pass_status = eval_result.get('test_cases_pass_status', [])
                    
                    for test_offset, status in enumerate(test_pass_status):
                        test_idx = test_start_idx + test_offset
                        if test_idx >= test_end_idx or test_idx >= num_tests:
                            continue
                        
                        # Handle different status formats
                        if isinstance(status, dict):
                            pass_value = status.get('pass', False)
                        else:
                            pass_value = bool(status)
                        
                        matrix[prog_idx][test_idx] = 1 if pass_value else 0
        
        # Create labels with unique ID and round information using Rn_Tm/Ri_Pj format
        # Check if we have cross-round evaluation data with unique IDs
        if qid in cross_round_data and 'program_metadata' in cross_data and 'test_metadata' in cross_data:
            program_unique_ids = [meta.get('unique_id', i+1) for i, meta in enumerate(program_metadata)]
            test_unique_ids = [meta.get('unique_id', i+1) for i, meta in enumerate(test_metadata)]
            # Use current round for display (Rn_Tm format)
            program_labels = [f"R{program_round_map[i]}_P{program_unique_ids[i]}" for i in range(num_programs)]
            test_labels = [f"R{test_round_map[i]}_T{test_unique_ids[i]}" for i in range(num_tests)]
        else:
            program_unique_ids = list(range(1, num_programs + 1))
            test_unique_ids = list(range(1, num_tests + 1))
            # Fallback to sequential numbering with round info
            program_labels = [f"R{program_round_map[i]}_P{i+1}" for i in range(num_programs)]
            test_labels = [f"R{test_round_map[i]}_T{i+1}" for i in range(num_tests)]
        
        # Create hover text with actual results
        hover_text = []
        for i, program in enumerate(programs):
            hover_row = []
            for j, test in enumerate(tests):
                result = matrix[i][j]
                if result == 1:
                    status = "PASS ✅"
                elif result == 0:
                    status = "FAIL ❌"
                else:
                    status = "ERROR ⚠️"
                
                # Use the already-defined unique IDs with clear labeling
                prog_label = f"R{program_round_map[i]}_P{program_unique_ids[i]}"
                test_label = f"R{test_round_map[j]}_T{test_unique_ids[j]}"
                
                hover_row.append(f"{prog_label} vs {test_label}<br>Status: {status}")
            hover_text.append(hover_row)
        
        # Color scale: Red for fail, Gray for not evaluated, Green for pass (consistent with other matrices)
        colorscale = [
            [0.0, '#FF0000'],    # Red for FAIL (0)
            [0.5, '#CCCCCC'],    # Gray for N/A (-1)  
            [1.0, '#00FF00']     # Green for PASS (1)
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=test_labels,
            y=program_labels,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale=colorscale,
            zmid=0,
            showscale=False,
            xgap=1,
            ygap=1
        ))
        
        fig.update_layout(
            title=f"Comprehensive Program-Test Matrix - All Rounds (QID: {qid[:8]}...)",
            xaxis_title="Test Cases (All Rounds)",
            yaxis_title="Programs (All Rounds)", 
            width=1200,
            height=800,
            font=dict(size=10),
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(side="top", tickangle=45)
        fig.update_yaxes(autorange="reversed")
        
        return fig
    
    def create_adversarial_evolution_matrix(self, qid: str) -> go.Figure:
        """Create adversarial evolution matrix showing first vs last round comparisons."""
        if not qid or qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="No data available - run pipeline first", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        rounds_data = self.data_cache[qid]
        if not rounds_data or len(rounds_data) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 rounds to show adversarial evolution", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Sort by round and get first and last rounds
        rounds_data.sort(key=lambda x: x.get('round', 0))
        first_round_data = rounds_data[0]
        last_round_data = rounds_data[-1]
        
        first_round_num = first_round_data.get('round', 0)
        last_round_num = last_round_data.get('round', 0)
        
        if first_round_num == last_round_num:
            fig = go.Figure()
            fig.add_annotation(text="Only one round found - need multiple rounds", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Extract data from first and last rounds
        first_round_tests = first_round_data.get('synthesis_result', {}).get('tests', [])
        first_round_eval_results = first_round_data.get('gen_result', {}).get('eval_results', [])
        
        last_round_tests = last_round_data.get('synthesis_result', {}).get('tests', [])
        last_round_eval_results = last_round_data.get('gen_result', {}).get('eval_results', [])
        
        # Extract programs from eval results
        first_round_programs = []
        for eval_result in first_round_eval_results:
            program = eval_result.get('program', '') or eval_result.get('parse_code', '')
            if program and len(program.strip()) > 0:
                first_round_programs.append(program)
        
        last_round_programs = []
        for eval_result in last_round_eval_results:
            program = eval_result.get('program', '') or eval_result.get('parse_code', '')
            if program and len(program.strip()) > 0:
                last_round_programs.append(program)

        if not first_round_programs or not last_round_programs or not first_round_tests or not last_round_tests:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data in first or last round", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        print(f"🔄 Creating adversarial evolution matrix for QID {qid[:20]}...")
        print(f"   First round ({first_round_num}): {len(first_round_programs)} programs, {len(first_round_tests)} tests")
        print(f"   Last round ({last_round_num}): {len(last_round_programs)} programs, {len(last_round_tests)} tests")

        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"First Round Programs vs Last Round Tests\n(R{first_round_num} programs vs R{last_round_num} tests)",
                f"Last Round Programs vs First Round Tests\n(R{last_round_num} programs vs R{first_round_num} tests)",
                f"First Round Self-Evaluation\n(R{first_round_num} programs vs R{first_round_num} tests)",
                f"Last Round Self-Evaluation\n(R{last_round_num} programs vs R{last_round_num} tests)"
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        # Evaluate different combinations
        try:
            # 1. First round programs vs Last round tests (top-left)
            matrix_1 = self.evaluate_program_test_combination(first_round_programs, last_round_tests)
            
            # 2. Last round programs vs First round tests (top-right) 
            matrix_2 = self.evaluate_program_test_combination(last_round_programs, first_round_tests)
            
            # 3. First round self-evaluation (bottom-left)
            matrix_3 = self.extract_existing_eval_matrix(first_round_data)
            
            # 4. Last round self-evaluation (bottom-right)
            matrix_4 = self.extract_existing_eval_matrix(last_round_data)
            
            # Add matrices to subplots
            matrices_and_positions = [
                (matrix_1, 1, 1, first_round_programs, last_round_tests, f"R{first_round_num}", f"R{last_round_num}"),
                (matrix_2, 1, 2, last_round_programs, first_round_tests, f"R{last_round_num}", f"R{first_round_num}"),
                (matrix_3, 2, 1, first_round_programs, first_round_tests, f"R{first_round_num}", f"R{first_round_num}"),
                (matrix_4, 2, 2, last_round_programs, last_round_tests, f"R{last_round_num}", f"R{last_round_num}")
            ]
            
            colorscale = [
                [0.0, '#FF0000'],    # Red for FAIL (0)
                [0.5, '#CCCCCC'],    # Gray for N/A (-1)  
                [1.0, '#00FF00']     # Green for PASS (1)
            ]
            
            for matrix, row, col, programs, tests, prog_round_prefix, test_round_prefix in matrices_and_positions:
                if matrix:
                    program_labels = [f"{prog_round_prefix}_P{i+1}" for i in range(len(programs))]
                    test_labels = [f"{test_round_prefix}_T{i+1}" for i in range(len(tests))]
                    
                    # Create hover text
                    hover_text = []
                    for i, prog_label in enumerate(program_labels):
                        hover_row = []
                        for j, test_label in enumerate(test_labels):
                            if i < len(matrix) and j < len(matrix[i]):
                                result = matrix[i][j]
                                if result == 1:
                                    status = "PASS ✅"
                                elif result == 0:
                                    status = "FAIL ❌"
                                else:
                                    status = "ERROR ⚠️"
                                hover_row.append(f"{prog_label} vs {test_label}<br>Status: {status}")
                            else:
                                hover_row.append(f"{prog_label} vs {test_label}<br>Status: N/A")
                        hover_text.append(hover_row)
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=matrix,
                            x=test_labels,
                            y=program_labels,
                            colorscale=colorscale,
                            zmid=0,
                            showscale=False,
                            hovertext=hover_text,
                            hovertemplate="%{hovertext}<extra></extra>",
                            xgap=1,
                            ygap=1
                        ),
                        row=row, col=col
                    )
                else:
                    # Add error message for missing matrix
                    fig.add_annotation(
                        text="Evaluation failed",
                        xref=f"x{(row-1)*2 + col}", 
                        yref=f"y{(row-1)*2 + col}",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font_size=12
                    )

        except Exception as e:
            print(f"❌ Error creating adversarial evolution matrix: {e}")
            import traceback
            traceback.print_exc()

        fig.update_layout(
            title=f"Adversarial Evolution Analysis (QID: {qid[:8]}...)<br>First Round {first_round_num} vs Last Round {last_round_num}",
            height=800,
            width=1200,
            font=dict(size=10)
        )
        
        return fig
    
    def evaluate_program_test_combination(self, programs: List[str], tests: List[str]) -> List[List[int]]:
        """Evaluate a specific combination of programs against tests."""
        if not programs or not tests:
            return []
        
        print(f"🔄 Evaluating {len(programs)} programs against {len(tests)} tests...")
        
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).resolve().parent.parent))
            from code_eval import eval_codes
            
            # Prepare evaluation data correctly
            all_programs = []
            all_tests = []
            
            for program in programs:
                all_programs.append(program)
                all_tests.append(tests)  # Each program gets the full test list
            
            # Use eval_codes
            pass_rates, all_pass_rates = eval_codes(
                solution_strs=all_programs,
                test_cases=all_tests,
                return_test_cases_pass_status=True,
                num_processes=8  # Reduce load
            )
            
            # Build matrix from results
            matrix = []
            
            for i, program in enumerate(programs):
                if i < len(all_pass_rates):
                    test_results = all_pass_rates[i]
                    program_row = []
                    
                    if isinstance(test_results, list):
                        for j, test_result in enumerate(test_results):
                            if j < len(tests):
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
                        # Fallback
                        program_pass_rate = pass_rates[i] if i < len(pass_rates) else 0
                        program_row = [1 if program_pass_rate > 0.5 else 0] * len(tests)
                else:
                    # Error case
                    program_row = [-1] * len(tests)
                
                matrix.append(program_row)
            
            return matrix
            
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            # Return error matrix
            return [[-1 for _ in range(len(tests))] for _ in range(len(programs))]
    
    def extract_existing_eval_matrix(self, round_data: Dict) -> List[List[int]]:
        """Extract evaluation matrix from existing round data."""
        try:
            eval_results = round_data.get('gen_result', {}).get('eval_results', [])
            tests = round_data.get('synthesis_result', {}).get('tests', [])
            
            if not eval_results or not tests:
                return []
            
            matrix = []
            for eval_result in eval_results:
                test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
                row = []
                
                for test_idx in range(len(tests)):
                    if test_idx < len(test_cases_pass_status):
                        test_result = test_cases_pass_status[test_idx]
                        if isinstance(test_result, dict):
                            passed = 1 if test_result.get('pass', False) else 0
                        else:
                            passed = 1 if test_result else 0
                        row.append(passed)
                    else:
                        row.append(-1)  # Missing data
                
                matrix.append(row)
            
            return matrix if matrix else []
            
        except Exception as e:
            print(f"❌ Error extracting existing eval matrix: {e}")
            return []

    def create_all_rounds_matrices(self, qid: str) -> go.Figure:
        """Create subplots showing program-test matrices for all rounds."""
        if not qid or qid not in self.data_cache:
            fig = go.Figure()
            fig.add_annotation(text="No data available - run pipeline first", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        rounds_data = self.data_cache[qid]
        if not rounds_data:
            fig = go.Figure()
            fig.add_annotation(text="No rounds data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig

        # Get all available rounds
        available_rounds = sorted(set(item.get('round', 0) for item in rounds_data))
        num_rounds = len(available_rounds)
        
        if num_rounds == 0:
            fig = go.Figure()
            fig.add_annotation(text="No valid rounds found", 
                             xref="paper", yref="paper", x=0.5, y=0.5, 
                             showarrow=False, font_size=16)
            return fig
        
        # Calculate subplot layout
        cols = min(3, num_rounds)  # Max 3 columns
        rows = (num_rounds + cols - 1) // cols  # Ceiling division
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f"Round {r}" for r in available_rounds],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Get maximum test count across all rounds for consistent grid layout
        max_test_count = 0
        for round_data in rounds_data:
            tests = round_data.get('synthesis_result', {}).get('tests', [])
            max_test_count = max(max_test_count, len(tests))
        
        # Use max test count for consistent grid, but each round will show actual tests
        max_test_labels = [f"t{i+1}" for i in range(max_test_count)]
        
        # Create matrix for each round
        for idx, round_num in enumerate(available_rounds):
            row_pos = (idx // cols) + 1
            col_pos = (idx % cols) + 1
            
            # Find round data
            round_data = None
            for item in rounds_data:
                if item.get('round', 0) == round_num:
                    round_data = item
                    break
            
            if round_data:
                eval_results = round_data.get('gen_result', {}).get('eval_results', [])
                programs = round_data.get('programs', [round_data.get('program', '')])
                current_tests = round_data.get('synthesis_result', {}).get('tests', [])
                
                # Build matrix for this round using actual test count
                matrix = []
                program_labels = []
                test_labels = [f"R{round_num}_T{i+1}" for i in range(len(current_tests))]
                
                for prog_idx, eval_result in enumerate(eval_results):
                    program_labels.append(f"R{round_num}_P{prog_idx + 1}")
                    row = []
                    test_cases_pass_status = eval_result.get('test_cases_pass_status', [])
                    
                    # Use actual test count for this round
                    for test_idx in range(len(current_tests)):
                        if test_idx < len(test_cases_pass_status):
                            test_result = test_cases_pass_status[test_idx]
                            if isinstance(test_result, dict):
                                passed = 1 if test_result.get('pass', False) else 0
                            else:
                                passed = 1 if test_result else 0
                            row.append(passed)
                        else:
                            row.append(-1)  # Missing data
                    
                    matrix.append(row)
                
                if not matrix:  # No evaluation results
                    program_labels = [f"R{round_num}_P1"]
                    matrix = [[-1] * len(current_tests)]
                
                # Convert for display with clear color mapping
                display_matrix = []
                for row in matrix:
                    display_row = []
                    for val in row:
                        if val == -1:
                            display_row.append(-1)   # Gray (N/A)
                        elif val == 0:
                            display_row.append(0)    # Red (FAIL)
                        else:
                            display_row.append(1)    # Green (PASS)
                    display_matrix.append(display_row)
                
                fig.add_trace(
                    go.Heatmap(
                        z=display_matrix,
                        x=test_labels,
                        y=program_labels,
                        colorscale=[
                            [0.0, '#FF0000'],    # Red for FAIL (0)
                            [0.5, '#CCCCCC'],    # Gray for N/A (-1)  
                            [1.0, '#00FF00']     # Green for PASS (1)
                        ],
                        zmid=0,  # Center the colorscale at 0
                        showscale=False,
                        xgap=1,
                        ygap=1
                    ),
                    row=row_pos, col=col_pos
                )
        
        fig.update_layout(
            title=f"All Rounds Program-Test Matrices (QID: {qid[:8]}...)",
            height=300 * rows,
            width=1000,
            font=dict(size=10)
        )
        
        return fig



def create_integrated_interface():
    """Create the integrated Gradio interface."""
    
    system = IntegratedAdvancedSystem()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button-secondary {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        color: white !important;
    }
    .log-container {
        background-color: #1e1e1e !important;
        color: #00ff00 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
        padding: 10px !important;
        border-radius: 5px !important;
        max-height: 400px !important;
        overflow-y: auto !important;
    }
    .progress-bar {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%) !important;
        border-radius: 10px !important;
    }
    """
    
    with gr.Blocks(
        title="🚀 Integrated Adversarial Generation System", 
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🚀 Integrated Adversarial Generation System</h1>
            <p>Configure, run, and visualize multi-round adversarial generation in real-time</p>
        </div>
        """)
        
        with gr.Tab("🎯 Pipeline Control"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>⚙️ Configuration</h3>")
                    
                    rounds_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=1,
                        label="Number of Rounds",
                        info="How many adversarial rounds to run (1-50)"
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "gpt-4.1-mini",
                            "gpt-4o-mini",
                            "gpt-4o",
                            "gpt-4-turbo",
                            "gpt-4",
                            "gpt-3.5-turbo"
                        ],
                        value="gpt-4.1-mini",
                        label="Model Name",
                        info="OpenAI model to use for generation"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=1000,
                        maximum=10000,
                        step=500,
                        value=4000,
                        label="Max Tokens",
                        info="Maximum tokens per API call"
                    )
                    
                    max_samples_slider = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        step=10,
                        value=100,
                        label="Max Samples",
                        info="Maximum number of questions to process"
                    )
                    
                    seed_number = gr.Number(
                        value=42,
                        label="Random Seed",
                        info="Random seed for reproducible results",
                        precision=0
                    )
                    
                    output_dir_text = gr.Textbox(
                        value="outputs/acecoder_rounds",
                        label="Output Directory",
                        info="Where to save results"
                    )
                    
                    overwrite_checkbox = gr.Checkbox(
                        value=True,
                        label="Overwrite Existing Files",
                        info="Whether to overwrite existing outputs"
                    )
                    
                    gr.HTML("<h4>🔑 API Configuration</h4>")
                    
                    api_key_text = gr.Textbox(
                        value=os.getenv('OPENAI_API_KEY', ''),
                        label="OpenAI API Key",
                        placeholder="sk-proj-...",
                        type="password",
                        info="Your OpenAI API key (will be masked)"
                    )
                    
                    backend_logging_checkbox = gr.Checkbox(
                        value=True,
                        label="🖥️ Show Backend Logs",
                        info="Display logs in terminal (backend) in addition to frontend"
                    )
                    
                    skip_step4_checkbox = gr.Checkbox(
                        value=False,
                        label="⏭️ Skip Step 4 (Cross-Round Evaluation)",
                        info="Skip the computationally expensive cross-round evaluation. Recommended for large datasets to prevent hanging."
                    )
                    
                    with gr.Row():
                        start_btn = gr.Button("🚀 Start Pipeline", variant="primary", size="lg")
                        stop_btn = gr.Button("🛑 Stop Pipeline", variant="secondary", size="lg")
                
                with gr.Column(scale=2):
                    gr.HTML("<h3>📊 Real-time Status</h3>")
                    
                    status_text = gr.Textbox(
                        value="Ready to start",
                        label="Current Status",
                        interactive=False
                    )
                    

                    
                    gr.HTML("<h3>📋 Live Logs</h3>")
                    log_display = gr.Textbox(
                        value="",
                        label="Execution Logs",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["log-container"]
                    )
        
        with gr.Tab("📊 Real-time Visualization"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>🎮 Visualization Controls</h3>")
                    
                    qid_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select QID",
                        info="Choose a Question ID to visualize"
                    )
                    
                    viz_type_radio = gr.Radio(
                        choices=["All Rounds (Grid)", "Single Round", "Comprehensive Matrix", "Adversarial Evolution"],
                        value="All Rounds (Grid)",
                        label="Visualization Type",
                        info="Choose visualization type"
                    )
                    
                    round_viz_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        step=1,
                        value=0,
                        label="Round Number",
                        info="Select which round to visualize (for Single Round Matrix)",
                        visible=False
                    )
                    
                    # Visualization directory selector
                    viz_dir_text = gr.Textbox(
                        value="outputs/acecoder_rounds",
                        label="Visualization Data Directory",
                        info="Path to the output directory containing visualization data"
                    )
                    
                    refresh_viz_btn = gr.Button("🔄 Refresh Visualization", variant="primary")
                    set_viz_dir_btn = gr.Button("📂 Set Viz Directory", variant="secondary")
            
            with gr.Row():
                matrix_plot = gr.Plot(label="Program-Test Results Matrix")
        
        with gr.Tab("📈 Results Summary"):
            gr.HTML("<h3>📋 Pipeline Summary</h3>")
            summary_text = gr.Markdown("Run the pipeline to see results summary")
        
        # State variables
        log_state = gr.State("")
        
        # Event handlers
        def start_pipeline(rounds, model_name, max_tokens, max_samples, seed, output_dir, overwrite, api_key, backend_logging, skip_step4):
            if system.is_running:
                return "⚠️ Pipeline already running!", ""
            
            # Set API key in environment if provided
            if api_key.strip():
                os.environ['OPENAI_API_KEY'] = api_key.strip()
            elif not os.getenv('OPENAI_API_KEY'):
                return "❌ Please provide OpenAI API Key!", ""
            
            # Configure backend logging
            system.set_backend_logging(backend_logging)
            
            # Start the pipeline
            system.run_pipeline(rounds, model_name, max_tokens, max_samples, output_dir, overwrite, seed, skip_step4)
            
            return "🚀 Pipeline started...", ""
        
        def stop_pipeline():
            system.stop_pipeline()
            return "🛑 Pipeline stopped", ""
        
        def update_status():
            status = system.get_status()
            all_logs = system.get_all_logs()
            
            # Check if pipeline just completed and results are available
            if not status['is_running'] and system.results_updated:
                status_msg = f"✅ Pipeline completed! New visualization data available - Check Visualization tab"
            elif status['is_running']:
                # Add progress information to status when running
                progress_percent = status['progress']
                if progress_percent > 0:
                    status_msg = f"🔄 Running - {status['status_text']} ({progress_percent:.1f}%)"
                else:
                    status_msg = f"🔄 Running - {status['status_text']}"
            else:
                status_msg = f"✅ Ready - {status['status_text']}"
            
            progress = status['progress'] / 100.0 if status['progress'] > 0 else 0
            
            return status_msg, all_logs, progress
        
        def set_visualization_directory(viz_dir):
            """Set the visualization data directory and reload results."""
            try:
                system.set_visualization_directory(viz_dir)
                system.load_results()
                qid_choices = system.get_qid_list()
                
                if qid_choices:
                    updated_plot = system.create_all_rounds_matrices(qid_choices[0])
                    return (
                        gr.Dropdown(choices=qid_choices, value=qid_choices[0]),
                        updated_plot,
                        f"✅ Loaded visualization data from: {viz_dir}"
                    )
                else:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text=f"No data found in {viz_dir}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    return (
                        gr.Dropdown(choices=[], value=None),
                        empty_fig,
                        f"❌ No visualization data found in: {viz_dir}"
                    )
            except Exception as e:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error loading from {viz_dir}: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return (
                    gr.Dropdown(choices=[], value=None),
                    empty_fig,
                    f"❌ Error: {str(e)}"
                )
        
        def refresh_visualization():
            # Force reload results regardless of flag
            system.load_results()
            qid_choices = system.get_qid_list()
            
            # Return updated components - default to all rounds grid
            updated_dropdown = gr.Dropdown(choices=qid_choices, value=qid_choices[0] if qid_choices else None)
            
            if qid_choices:
                try:
                    updated_plot = system.create_all_rounds_matrices(qid_choices[0])
                except Exception as e:
                    print(f"❌ Error in refresh_visualization: {e}")
                    # Create error figure
                    updated_plot = go.Figure()
                    updated_plot.add_annotation(text=f"Error loading visualization: {str(e)}", 
                                              xref="paper", yref="paper", x=0.5, y=0.5, 
                                              showarrow=False, font_size=16)
            else:
                updated_plot = None
                
            updated_summary = get_summary()
            
            return updated_dropdown, updated_plot, updated_summary
        
        def update_visualization(qid, viz_type, round_num):
            if not qid:
                return None
            
            try:
                if viz_type == "All Rounds (Grid)":
                    return system.create_all_rounds_matrices(qid)
                elif viz_type == "Single Round":
                    return system.create_program_test_matrix(qid, round_num)
                elif viz_type == "Comprehensive Matrix":
                    return system.create_comprehensive_matrix(qid)
                elif viz_type == "Adversarial Evolution":
                    return system.create_adversarial_evolution_matrix(qid)
            except Exception as e:
                print(f"❌ Error in update_visualization: {e}")
                # Return empty figure on error
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {str(e)}", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, 
                                 showarrow=False, font_size=16)
                return fig
        
        def toggle_round_slider(viz_type):
            """Show/hide round slider based on visualization type."""
            if viz_type in ["All Rounds (Grid)", "Comprehensive Matrix", "Adversarial Evolution"]:
                return gr.Slider(visible=False)
            else:
                return gr.Slider(visible=True)
        
        def get_summary():
            if not system.data_cache or "error" in system.data_cache:
                return "No results available yet. Run the pipeline to generate data."
            
            summary = "## 📊 Pipeline Results Summary\n\n"
            summary += f"**Total QIDs processed:** {len(system.data_cache)}\n\n"
            
            for qid, items in system.data_cache.items():
                summary += f"### QID: {qid[:32]}...\n"
                summary += f"- **Rounds completed:** {len(items)}\n"
                final_item = items[-1] if items else {}
                tests = final_item.get('synthesis_result', {}).get('tests') or []
                programs = final_item.get('gen_result', {}).get('eval_results') or []
                summary += f"- **Final state:** {len(programs)} programs × {len(tests)} test cases\n\n"
            
            return summary
        
        # Wire up events
        start_btn.click(
            start_pipeline,
            inputs=[rounds_slider, model_dropdown, max_tokens_slider, max_samples_slider, seed_number, output_dir_text, overwrite_checkbox, api_key_text, backend_logging_checkbox, skip_step4_checkbox],
            outputs=[status_text, log_display]
        )
        
        stop_btn.click(
            stop_pipeline,
            outputs=[status_text, log_display]
        )
        
        refresh_viz_btn.click(
            refresh_visualization,
            outputs=[qid_dropdown, matrix_plot, summary_text]
        )
        
        # Visualization type toggle
        viz_type_radio.change(
            toggle_round_slider,
            inputs=[viz_type_radio],
            outputs=[round_viz_slider]
        )
        
        # Update visualization when any control changes
        qid_dropdown.change(
            update_visualization,
            inputs=[qid_dropdown, viz_type_radio, round_viz_slider],
            outputs=[matrix_plot]
        )
        
        viz_type_radio.change(
            update_visualization,
            inputs=[qid_dropdown, viz_type_radio, round_viz_slider],
            outputs=[matrix_plot]
        )
        
        round_viz_slider.change(
            update_visualization,
            inputs=[qid_dropdown, viz_type_radio, round_viz_slider],
            outputs=[matrix_plot]
        )
        
        # Periodic status updates
        def periodic_update():
            status_msg, new_logs, progress = update_status()
            
            # Progress bar is handled separately, just return status and logs
            return status_msg, new_logs
        
        # Set up automatic refresh every 1 second for better responsiveness
        status_timer = gr.Timer(1.0)
        status_timer.tick(
            periodic_update,
            outputs=[status_text, log_display]
        )
        
        # Manual visualization refresh after pipeline completion

        

        
        set_viz_dir_btn.click(
            set_visualization_directory,
            inputs=[viz_dir_text],
            outputs=[qid_dropdown, matrix_plot, summary_text]
        )
    
    return app

if __name__ == "__main__":
    app = create_integrated_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=False
    )
