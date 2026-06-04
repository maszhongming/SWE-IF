# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import sys
import io
import re
import ast
import tokenize
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# For functional evaluation
try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Warning: gradio_client not available. BigCodeBench functional evaluation will be skipped.")

# For LiveCodeBench evaluation
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available. LiveCodeBench functional evaluation will be skipped.")

# Import LiveCodeBench utilities
try:
    from .livecodebench_utils import (
        translate_private_test_cases, 
        has_test_type, 
        post_process_code, 
        lcb_run, 
        has_code
    )
    LIVECODEBENCH_UTILS_AVAILABLE = True
except ImportError:
    LIVECODEBENCH_UTILS_AVAILABLE = False
    print("Warning: livecodebench_utils not available. LiveCodeBench functional evaluation will be skipped.")


# ==============================================================================
#  IF EVALUATION CODE
# ==============================================================================

def _run_ruff_check(response: str, *ruff_args: str) -> bool:
    """
    A generic helper function to run Ruff with specific arguments.
    Uses stdin to pass code content to avoid file I/O overhead.
    """
    if not shutil.which("ruff"):
        raise RuntimeError(
            "Ruff is not installed or not in the system's PATH. "
            "Please install it with 'pip install ruff'."
        )

    try:
        command = [
            "ruff", "check", "-",  # "-" means read from stdin
            *ruff_args,
            "--no-fix",
            "--force-exclude"
        ]

        result = subprocess.run(
            command,
            input=response,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        return result.returncode == 0

    except Exception:
        return False


# Style tests
def test_pep8_compliance(response: str) -> bool:
    return _run_ruff_check(response, "--select", "E,W")

def test_isort_imports(response: str) -> bool:
    return _run_ruff_check(response, "--select", "I")

def test_line_length(response: str, line_length: int = 79) -> bool:
    return _run_ruff_check(response, "--select", "E501", "--line-length", str(line_length))

def test_trailing_commas(response: str) -> bool:
    return _run_ruff_check(response, "--select", "COM812")

def test_quote_consistency(response: str) -> bool:
    return _run_ruff_check(response, "--select", "Q")

def test_pep8_local_variable_naming(response: str) -> bool:
    return _run_ruff_check(response, "--select", "N806")

def test_no_redundant_open_modes(response: str) -> bool:
    return _run_ruff_check(response, "--select", "UP015")

def test_no_magic_values(response: str) -> bool:
    return _run_ruff_check(response, "--select", "PLR2004")

def test_no_unused_unpacked_variable(response: str) -> bool:
    return _run_ruff_check(response, "--select", "RUF059", "--preview")

# Logic tests
def test_pyflakes(response: str) -> bool:
    return _run_ruff_check(response, "--select", "F")

def test_nesting_depth(response: str, max_nested_blocks: int = 1) -> bool:
    return _run_ruff_check(response, "--select", "PLR1702", "--config", f"lint.pylint.max-nested-blocks={max_nested_blocks}", "--preview")

def test_max_branch(response: str, max_branches: int = 2) -> bool:
    return _run_ruff_check(response, "--select", "PLR0912", "--config", f"lint.pylint.max-branches={max_branches}")

def test_max_statements(response: str, max_statements: int = 6) -> bool:
    return _run_ruff_check(response, "--select", "PLR0915", "--config", f"lint.pylint.max-statements={max_statements}")

def test_max_locals(response: str, max_locals: int = 2) -> bool:
    return _run_ruff_check(response, "--select", "PLR0914", "--config", f"lint.pylint.max-locals={max_locals}", "--preview")

def test_redundant_control_flow(response: str) -> bool:
    return _run_ruff_check(response, "--select", "RET504,RET505,RET506,RET507,RET508")

def test_cyclomatic_complexity(response: str, max_complexity: int = 2) -> bool:
    return _run_ruff_check(response, "--select", "C901", "--config", f"lint.mccabe.max-complexity={max_complexity}")

def test_use_comprehensions(response: str) -> bool:
    return _run_ruff_check(response, "--select", "PERF401,PERF402,PERF403")

def test_boolean_complexity(response: str, max_bool_expr: int = 1) -> bool:
    return _run_ruff_check(response, "--select", "PLR0916", "--config", f"lint.pylint.max-bool-expr={max_bool_expr}", "--preview")

# Documentation tests
def test_type_annotations(response: str) -> bool:
    return _run_ruff_check(response, "--select", "ANN001,ANN002,ANN003,ANN201,ANN202,ANN204,ANN205,ANN206,ANN401")

def test_docstring_length(response: str, line_length: int = 72) -> bool:
    return _run_ruff_check(response, "--select", "W505", "--config", f"lint.pycodestyle.max-doc-length={str(line_length)}")

def test_docstring_convention(response: str, convention: str = "pep257") -> bool:
    return _run_ruff_check(response, "--select", "D", "--config", f"lint.pydocstyle.convention='{convention}'")

def test_pure_code_output(response: str) -> bool:
    try:
        ast.parse(response)
        return True
    except SyntaxError:
        return False

def test_no_comments_or_docstrings(response: str) -> bool:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(response).readline)
        if any(token.type == tokenize.COMMENT for token in tokens):
            return False
    except (tokenize.TokenError, IndentationError):
        return False

    try:
        tree = ast.parse(response)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is not None:
                    return False
    except SyntaxError:
        return False
    return True

def test_json_format(response: str) -> bool:
    pattern = r"^\s*```python\s*([\s\S]+?)\s*```\s*```json\s*([\s\S]+?)\s*```\s*$"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return False
    json_string = match.group(2)
    try:
        json.loads(json_string)
    except json.JSONDecodeError:
        return False
    return True

# Error handling tests
def test_exception_chaining(response: str) -> bool:
    return _run_ruff_check(response, "--select", "B904")

def test_exception_messages(response: str) -> bool:
    return _run_ruff_check(response, "--select", "EM101,EM102,EM103")

def test_os_error_alias(response: str) -> bool:
    return _run_ruff_check(response, "--select", "UP024")

def test_no_vanilla_raise_args(response: str) -> bool:
    return _run_ruff_check(response, "--select", "TRY003")

# Library tests
def test_use_pathlib(response: str) -> bool:
    return _run_ruff_check(response, "--select", "PTH")

def test_idiomatic_pandas(response: str) -> bool:
    return _run_ruff_check(response, "--select", "PD901,PD002,PD011")

# Test dispatcher
TEST_DISPATCHER = {
    'style_1': test_pep8_compliance,
    'style_2': test_isort_imports,
    'style_3': test_line_length,
    'style_4': test_trailing_commas,
    'style_5': test_quote_consistency,
    'style_6': test_pep8_local_variable_naming,
    'style_7': test_no_redundant_open_modes,
    'style_8': test_no_magic_values,
    'style_9': test_no_unused_unpacked_variable,
    'logic_1': test_pyflakes,
    'logic_2': test_nesting_depth,
    'logic_3': test_max_branch,
    'logic_4': test_max_statements,
    'logic_5': test_max_locals,
    'logic_6': test_redundant_control_flow,
    'logic_7': test_cyclomatic_complexity,
    'logic_8': test_use_comprehensions,
    'logic_9': test_boolean_complexity,
    'doc_1': test_type_annotations,
    'doc_2': test_docstring_length,
    'doc_3': test_docstring_convention,
    'doc_4': test_pure_code_output,
    'doc_5': test_no_comments_or_docstrings,
    'doc_6': test_json_format,
    'error_1': test_exception_chaining,
    'error_2': test_exception_messages,
    'error_3': test_os_error_alias,
    'error_4': test_no_vanilla_raise_args,
    'library_1': test_use_pathlib,
    'library_2': test_idiomatic_pandas,
}

# Category mapping for analysis
CATEGORY_MAPPING = {
    'style': ['style_1', 'style_2', 'style_3', 'style_4', 'style_5', 'style_6', 'style_7', 'style_8', 'style_9'],
    'logic': ['logic_1', 'logic_2', 'logic_3', 'logic_4', 'logic_5', 'logic_6', 'logic_7', 'logic_8', 'logic_9'],
    'doc': ['doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5', 'doc_6'],
    'error': ['error_1', 'error_2', 'error_3', 'error_4'],
    'library': ['library_1', 'library_2'],
}


def evaluate_if_constraint(if_id: str, response: str, **kwargs) -> bool:
    """Evaluates a code response using the dispatch dictionary."""
    if if_id not in TEST_DISPATCHER:
        print(f"Warning: Unknown IF constraint ID: '{if_id}'. Returning False.", file=sys.stderr)
        return False
    
    test_function = TEST_DISPATCHER[if_id]
    try:
        return test_function(response, **kwargs)
    except Exception as e:
        print(f"Warning: Evaluation for constraint '{if_id}' failed with error: {e}. Returning False.", file=sys.stderr)
        return False


# ==============================================================================
#  MAIN EVALUATOR CLASSES WITH MULTITHREADING
# ==============================================================================

class FunctionalEvaluator:
    """Evaluator for functional correctness using BigCodeBench API and LiveCodeBench local evaluation."""
    
    def __init__(self):
        # Lazy initialization for BigCodeBench API
        self.gradio_endpoint = "https://bigcode-bigcodebench-evaluator.hf.space/"
        self.client = None
        
        # LiveCodeBench dataset cache
        self._livecodebench_dataset = None
    
    def _load_livecodebench_dataset(self):
        """Load LiveCodeBench dataset with test cases."""
        if not DATASETS_AVAILABLE or not LIVECODEBENCH_UTILS_AVAILABLE:
            raise RuntimeError("datasets library and livecodebench_utils are required for LiveCodeBench evaluation")
        
        if self._livecodebench_dataset is None:
            print("Loading LiveCodeBench dataset...")
            # Load the dataset used in your generator
            from .data import load_benchmark
            dataset_items = load_benchmark('livecodebench')
            
            # Convert to dict for easy lookup by question_id
            self._livecodebench_dataset = {}
            for item in dataset_items:
                question_id = item['question_id']
                # Decode private test cases
                private_test_cases = translate_private_test_cases(item['private_test_cases'])
                # Check if stdin type
                is_stdin = has_test_type(item['public_test_cases'], 'stdin')
                
                self._livecodebench_dataset[question_id] = {
                    'question_id': question_id,
                    'test': private_test_cases,
                    'is_stdin': is_stdin,
                    'public_test_cases': item['public_test_cases'],
                    'difficulty': item.get('difficulty', 'unknown')
                }
            print(f"Loaded {len(self._livecodebench_dataset)} LiveCodeBench examples")
        
        return self._livecodebench_dataset
    
    def _check_correctness_livecodebench(self, problem: Dict, completion: str, timeout: float = 12.0) -> bool:
        """Check correctness for LiveCodeBench problem."""
        is_extracted = not problem['is_stdin']
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        return all(details)
    
    def _evaluate_single_livecodebench(self, args: Tuple) -> Dict[str, Any]:
        """Evaluate a single LiveCodeBench example."""
        task_id, solution_code, full_response = args
        
        # Get problem from dataset
        dataset = self._load_livecodebench_dataset()
        if task_id not in dataset:
            return {
                'task_id': task_id,
                'passed': False,
                'error': f'Task {task_id} not found in dataset'
            }
        
        problem = dataset[task_id]
        
        # Extract code from response if needed
        if not solution_code.strip():
            # Try to extract from full_response
            code_matches = has_code(full_response)
            if code_matches:
                solution_code = code_matches[-1]  # Use the last code block
            else:
                solution_code = full_response  # Use full response as code
        
        if not solution_code.strip():
            return {
                'task_id': task_id,
                'passed': False,
                'error': 'No code found in response'
            }
        
        try:
            # Post-process the code
            processed_code = post_process_code(solution_code)
            
            # Check correctness
            passed = self._check_correctness_livecodebench(problem, processed_code)
            
            return {
                'task_id': task_id,
                'passed': passed,
                'error': None if passed else 'Tests failed'
            }
            
        except Exception as e:
            return {
                'task_id': task_id,
                'passed': False,
                'error': f'Evaluation error: {str(e)}'
            }
    
    def evaluate_livecodebench(self, input_file: str) -> Dict[str, Any]:
        """
        Evaluate LiveCodeBench functional correctness for a given file.
        
        Args:
            input_file: Path to the JSONL file with solutions
            
        Returns:
            Dictionary with evaluation results including pass@1
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Pre-load dataset before threading to avoid multiple loading
        self._load_livecodebench_dataset()
        
        # Load solutions from JSONL file
        tasks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    task_id = data.get('task_id', 'unknown')
                    solution_code = data.get('solution', '')
                    full_response = data.get('full_response', '')
                    
                    tasks.append((task_id, solution_code, full_response))
                    
                except Exception as e:
                    print(f"Error processing line in LiveCodeBench evaluation: {e}")
                    continue
        
        if not tasks:
            return {'pass_at_1': 0.0, 'error': 'No valid tasks found'}
        
        print(f"Evaluating {len(tasks)} LiveCodeBench examples...")
        
        # Evaluate in parallel using threads
        results = []
        with ThreadPoolExecutor(max_workers=min(64, len(tasks))) as executor:
            # Submit all tasks
            futures = [executor.submit(self._evaluate_single_livecodebench, task) for task in tasks]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="LiveCodeBench evaluation"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in LiveCodeBench evaluation: {e}")
                    results.append({
                        'task_id': 'unknown',
                        'passed': False,
                        'error': f'Future error: {str(e)}'
                    })
        
        # Calculate metrics
        total_tasks = len(results)
        passed_tasks = sum(1 for r in results if r['passed'])
        pass_at_1 = passed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Count by difficulty if available
        difficulty_stats = defaultdict(lambda: {'total': 0, 'passed': 0})
        dataset = self._load_livecodebench_dataset()
        
        for result in results:
            task_id = result['task_id']
            if task_id in dataset:
                difficulty = dataset[task_id].get('difficulty', 'unknown')
                difficulty_stats[difficulty]['total'] += 1
                if result['passed']:
                    difficulty_stats[difficulty]['passed'] += 1
        
        return {
            'pass_at_1': pass_at_1,
            'total_tasks': total_tasks,
            'passed_tasks': passed_tasks,
            'failed_tasks': [r for r in results if not r['passed']],
            'difficulty_breakdown': {
                diff: {
                    'pass_rate': stats['passed'] / stats['total'] if stats['total'] > 0 else 0.0,
                    'total': stats['total'],
                    'passed': stats['passed']
                }
                for diff, stats in difficulty_stats.items()
            },
            'detailed_results': results
        }
    
    def evaluate(self, input_file: str, benchmark: str) -> Dict[str, Any]:
        """
        Evaluate functional correctness for a given file.
        
        Args:
            input_file: Path to the JSONL file with solutions
            benchmark: 'bigcodebench' or 'livecodebench'
            
        Returns:
            Dictionary with evaluation results including pass@1
        """
        if benchmark == 'livecodebench':
            return self.evaluate_livecodebench(input_file)
        elif benchmark == 'bigcodebench':
            return self.evaluate_bigcodebench(input_file, benchmark)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    def evaluate_bigcodebench(self, input_file: str, benchmark: str) -> Dict[str, Any]:
        """
        Evaluate BigCodeBench functional correctness using Gradio API.
        
        Args:
            input_file: Path to the JSONL file with solutions
            benchmark: 'bigcodebench'
            
        Returns:
            Dictionary with evaluation results including pass@1
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("gradio_client is required for BigCodeBench functional evaluation")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Lazy initialization of Gradio client
        if self.client is None:
            self.client = Client(self.gradio_endpoint)
        
        # Determine subset based on benchmark
        subset = 'full'
        
        try:
            results, pass_at_k = self.client.predict(
                split='instruct',
                subset=subset,
                samples=handle_file(input_file),
                pass_k="1",
                parallel=-1,
                min_time_limit=1,
                max_as_limit=30*1024,
                max_data_limit=30*1024,
                max_stack_limit=10,
                calibrated=True,
                check_gt_only=False,
                no_gt=False,
                selective_evaluate='',
                api_name="/predict"
            )
            
            return {
                'pass_at_1': pass_at_k.get('pass@1', 0.0),
                'detailed_results': results,
                'model': pass_at_k.get('model', ''),
                'split': pass_at_k.get('split', ''),
                'subset': pass_at_k.get('subset', ''),
                'calibrated': pass_at_k.get('calibrated', True),
                'gt_pass_rate': pass_at_k.get('gt_pass_rate', 1.0),
                'failed_tasks': pass_at_k.get('failed_tasks', [])
            }
            
        except Exception as e:
            print(f"Error during BigCodeBench functional evaluation: {e}")
            return {'pass_at_1': 0.0, 'error': str(e)}


class IFEvaluator:
    """Evaluator for instruction following constraints."""
    
    def __init__(self, taxonomy_path: str = None, n_threads: int = None):
        """
        Initialize IF evaluator.
        
        Args:
            taxonomy_path: Path to the taxonomy CSV file
            n_threads: Number of parallel threads (default: min(64, cpu_count * 4))
        """
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy()
        # Default thread count: 4x CPU count but capped at 64
        if n_threads is None:
            n_threads = min(64, (os.cpu_count() or 1) * 4)
        self.n_threads = n_threads
    
    def _load_taxonomy(self) -> pd.DataFrame:
        """Load the VeriCode taxonomy (from HF MingZhong/SWE-IF by default, or
        from a local CSV/JSONL if taxonomy_path was given)."""
        try:
            from .data import load_taxonomy
            return load_taxonomy(local_path=self.taxonomy_path)
        except Exception as e:
            print(f"Warning: Could not load taxonomy: {e}")
            return pd.DataFrame()
    
    def evaluate_instance(self, solution_code: str, full_response: str, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate IF constraints for a single instance.
        
        Args:
            solution_code: The extracted code solution
            full_response: The full LLM response
            constraints: List of constraint dictionaries with 'index' and 'parameters'
            
        Returns:
            Dictionary with IF evaluation results
        """
        if not constraints:
            return {
                'constraint_level_score': 1.0,
                'instance_level_score': 1.0,
                'constraint_details': []
            }
        
        constraint_results = []
        for constraint in constraints:
            constraint_id = constraint.get('index')
            parameters = constraint.get('parameters', {})
            
            # Use full_response for doc_4 and doc_6, solution_code for others
            response_to_check = full_response if constraint_id in ['doc_4', 'doc_6'] else solution_code
            
            # Evaluate constraint
            result = evaluate_if_constraint(constraint_id, response_to_check, **parameters)
            
            # Get constraint info from taxonomy
            constraint_info = {'category': 'unknown', 'description': 'unknown'}
            if not self.taxonomy.empty:
                taxonomy_row = self.taxonomy[self.taxonomy['ID'] == constraint_id]
                if not taxonomy_row.empty:
                    constraint_info = {
                        'category': taxonomy_row.iloc[0]['Category'],
                        'description': taxonomy_row.iloc[0]['Description']
                    }
            
            constraint_results.append({
                'constraint_id': constraint_id,
                'passed': result,
                'score': int(result),
                'parameters': parameters,
                'category': constraint_info['category'],
                'description': constraint_info['description']
            })
        
        # Calculate scores
        individual_scores = [r['score'] for r in constraint_results]
        constraint_level_score = sum(individual_scores) / len(individual_scores) if individual_scores else 1.0
        instance_level_score = 1.0 if constraint_level_score == 1.0 else 0.0
        
        return {
            'constraint_level_score': constraint_level_score,
            'instance_level_score': instance_level_score,
            'constraint_details': constraint_results
        }
    
    def _evaluate_instance_wrapper(self, args: Tuple) -> Optional[Dict[str, Any]]:
        """Wrapper for thread pool execution."""
        task_id, solution_code, full_response, constraints = args
        try:
            result = self.evaluate_instance(solution_code, full_response, constraints)
            return {
                'task_id': task_id,
                'result': result
            }
        except Exception as e:
            print(f"Error evaluating instance {task_id}: {e}")
            return None
    
    def evaluate_file(self, input_file: str, target_constraints: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate IF constraints for all instances in a file using multithreading.
        
        Args:
            input_file: Path to JSONL file with solutions and if_info
            target_constraints: Optional list of constraints to evaluate against
            
        Returns:
            Dictionary with aggregated IF evaluation results
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # First, collect all tasks
        tasks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    task_id = data.get('task_id', 'unknown')
                    solution_code = data.get('solution', '')
                    full_response = data.get('full_response', '')
                    if_info = data.get('if_info', {})
                    
                    # Use target_constraints if provided, otherwise use file's constraints
                    constraints = target_constraints if target_constraints is not None else if_info.get('constraints', [])
                    
                    tasks.append((task_id, solution_code, full_response, constraints))
                    
                except Exception as e:
                    print(f"Error processing line in IF evaluation: {e}")
                    continue
        
        # Process tasks in parallel using threads
        all_constraint_scores = []
        all_instance_scores = []
        category_scores = defaultdict(list)
        constraint_type_scores = defaultdict(list)
        detailed_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit all tasks
            futures = [executor.submit(self._evaluate_instance_wrapper, task) for task in tasks]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating IF constraints"):
                result_data = future.result()
                if result_data:
                    task_id = result_data['task_id']
                    result = result_data['result']
                    
                    # Store scores
                    all_constraint_scores.append(result['constraint_level_score'])
                    all_instance_scores.append(result['instance_level_score'])
                    
                    # Group by category and constraint type
                    for detail in result['constraint_details']:
                        category = detail['category']
                        constraint_id = detail['constraint_id']
                        score = detail['score']
                        
                        category_scores[category].append(score)
                        constraint_type_scores[constraint_id].append(score)
                    
                    # Store detailed result
                    detailed_results.append(result_data)
        
        # Calculate aggregated scores
        overall_constraint_score = sum(all_constraint_scores) / len(all_constraint_scores) if all_constraint_scores else 0.0
        overall_instance_score = sum(all_instance_scores) / len(all_instance_scores) if all_instance_scores else 0.0
        
        # Calculate category scores
        category_summary = {}
        for category, scores in category_scores.items():
            category_summary[category] = {
                'constraint_level_score': sum(scores) / len(scores) if scores else 0.0,
                'count': len(scores)
            }
        
        # Calculate constraint type scores
        constraint_type_summary = {}
        for constraint_id, scores in constraint_type_scores.items():
            constraint_type_summary[constraint_id] = {
                'constraint_level_score': sum(scores) / len(scores) if scores else 0.0,
                'count': len(scores)
            }
        
        return {
            'overall_constraint_score': overall_constraint_score,
            'overall_instance_score': overall_instance_score,
            'category_summary': category_summary,
            'constraint_type_summary': constraint_type_summary,
            'detailed_results': detailed_results,
            'total_instances': len(detailed_results)
        }


class CodeEvaluator:
    """Main evaluator for code generation benchmarks with functional and instruction following evaluation."""
    
    def __init__(self, taxonomy_path: str = "data/code_gym_taxonomy.csv", 
                 eval_functional: bool = True, eval_if: bool = True, n_threads: int = None):
        """
        Initialize comprehensive evaluator.
        
        Args:
            taxonomy_path: Path to the taxonomy CSV file
            eval_functional: Whether to evaluate functional correctness
            eval_if: Whether to evaluate instruction following
            n_threads: Number of threads for IF evaluation
        """
        self.eval_functional = eval_functional
        self.eval_if = eval_if
        
        # Initialize evaluators based on what's needed
        if self.eval_functional:
            self.functional_evaluator = FunctionalEvaluator()
        else:
            self.functional_evaluator = None
            
        if self.eval_if:
            self.if_evaluator = IFEvaluator(taxonomy_path, n_threads=n_threads)
        else:
            self.if_evaluator = None
        
    def evaluate_single(self, task: str, benchmark: str, model: str, n_instructions: int) -> Dict[str, Any]:
        """
        Evaluate a single configuration (task + benchmark + model + n_instructions).
        
        Args:
            task: 'generation' or 'editing'
            benchmark: 'bigcodebench' or 'livecodebench'
            model: Model name
            n_instructions: Number of instructions
            
        Returns:
            Dictionary with evaluation results
        """
        # Sanitize model name for filesystem paths ("openai/gpt-4o-mini" -> "openai__gpt-4o-mini"),
        # matching how code_generator names its output files.
        model = model.replace('/', '__')
        # Construct input file path
        # For editing task with 0-instructions, read from generation folder
        if task == 'editing' and n_instructions == 0:
            input_file = Path(f"results/{benchmark}/generation/{model}_{n_instructions}-instructions.jsonl")
        else:
            input_file = Path(f"results/{benchmark}/{task}/{model}_{n_instructions}-instructions.jsonl")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        results = {
            'task': task,
            'benchmark': benchmark,
            'model': model,
            'n_instructions': n_instructions,
            'input_file': str(input_file)
        }
        
        # Functional evaluation
        if self.eval_functional and self.functional_evaluator:
            print(f"Evaluating functional correctness for {n_instructions}-instructions...")
            try:
                func_results = self.functional_evaluator.evaluate(str(input_file), benchmark)
                results['functional'] = func_results
                pass_at_1 = func_results.get('pass_at_1', 0.0)
                print(f"✓ Functional Pass@1: {pass_at_1:.4f}")
            except Exception as e:
                print(f"✗ Functional evaluation failed: {e}")
                results['functional'] = {'pass_at_1': 0.0, 'error': str(e)}
        elif self.eval_functional:
            missing_deps = []
            if not GRADIO_AVAILABLE and benchmark == 'bigcodebench':
                missing_deps.append('gradio_client')
            if not DATASETS_AVAILABLE and benchmark == 'livecodebench':
                missing_deps.append('datasets')
            if not LIVECODEBENCH_UTILS_AVAILABLE and benchmark == 'livecodebench':
                missing_deps.append('livecodebench_utils')
            
            error_msg = f"Required dependencies missing: {', '.join(missing_deps)}" if missing_deps else "Functional evaluator not initialized"
            print(f"✗ Functional evaluation requested but not available ({error_msg})")
            results['functional'] = {'pass_at_1': 0.0, 'error': error_msg}
        
        # IF evaluation (only for instructions > 0)
        if self.eval_if and self.if_evaluator and n_instructions > 0:
            print(f"Evaluating IF constraints for {n_instructions}-instructions...")
            try:
                if_results = self.if_evaluator.evaluate_file(str(input_file))
                results['if_evaluation'] = if_results
                
                constraint_score = if_results.get('overall_constraint_score', 0.0)
                instance_score = if_results.get('overall_instance_score', 0.0)
                total_instances = if_results.get('total_instances', 0)
                print(f"✓ IF Constraint-Level: {constraint_score:.4f}")
                print(f"✓ IF Instance-Level: {instance_score:.4f}")
                print(f"✓ Total Instances: {total_instances}")
                    
            except Exception as e:
                print(f"✗ IF evaluation failed: {e}")
                results['if_evaluation'] = {'overall_constraint_score': 0.0, 'overall_instance_score': 0.0, 'error': str(e)}
        elif self.eval_if and n_instructions == 0:
            print("Skipping IF evaluation for 0-instructions")
        
        return results
    
    def evaluate_range(self, task: str, benchmark: str, model: str, n_instructions: int) -> Dict[str, Any]:
        """
        Evaluate a range from 0 to n_instructions.
        
        Args:
            task: 'generation' or 'editing'
            benchmark: 'bigcodebench' or 'livecodebench'
            model: Model name
            n_instructions: Maximum number of instructions
            
        Returns:
            Dictionary with evaluation results for all instruction counts
        """
        all_results = {}
        
        for i in range(n_instructions + 1):
            print(f"\n{'='*60}")
            print(f"Evaluating {i}-instructions")
            print(f"{'='*60}")
            
            try:
                result = self.evaluate_single(task, benchmark, model, i)
                all_results[f"{i}_instructions"] = result
            except Exception as e:
                print(f"Failed to evaluate {i}-instructions: {e}")
                all_results[f"{i}_instructions"] = {
                    'task': task,
                    'benchmark': benchmark,
                    'model': model,
                    'n_instructions': i,
                    'error': str(e)
                }
        
        return {
            'task': task,
            'benchmark': benchmark,
            'model': model,
            'max_instructions': n_instructions,
            'results': all_results
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save essential evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create simplified results with only essential information
        simplified_results = {
            'task': results.get('task'),
            'benchmark': results.get('benchmark'),
            'model': results.get('model'),
            'max_instructions': results.get('max_instructions'),
            'results': {}
        }
        
        evaluation_results = results.get('results', {})
        for instruction_key, result in evaluation_results.items():
            n_instructions = result.get('n_instructions', 0)
            
            # Extract overall scores
            overall_scores = {}
            if 'functional' in result:
                overall_scores['pass_at_1'] = result['functional'].get('pass_at_1', 0.0)
            
            if 'if_evaluation' in result:
                if_eval = result['if_evaluation']
                overall_scores['constraint_level_score'] = if_eval.get('overall_constraint_score', 0.0)
                overall_scores['instance_level_score'] = if_eval.get('overall_instance_score', 0.0)
            
            # Extract instance-level details - one record per task_id (not per constraint)
            instances = []
            if 'if_evaluation' in result and 'detailed_results' in result['if_evaluation']:
                for detail in result['if_evaluation']['detailed_results']:
                    task_id = detail['task_id']
                    instance_result = detail['result']
                    
                    # Get functional score for this instance (if available)
                    functional_score = 0.0  # Default, would need per-instance functional results
                    
                    # Collect all constraint information for this instance
                    if_indices = []
                    parameters_list = []
                    individual_scores = []
                    
                    constraint_details = instance_result.get('constraint_details', [])
                    if constraint_details:
                        for constraint_detail in constraint_details:
                            if_indices.append(constraint_detail['constraint_id'])
                            parameters_list.append(constraint_detail['parameters'])
                            individual_scores.append(constraint_detail['score'])
                    
                    # Create single record per task_id with all constraint information
                    instances.append({
                        'task_id': task_id,
                        'functional_score': functional_score,
                        'constraint_level_score': instance_result.get('constraint_level_score', 0.0),  # Average of all constraints
                        'instance_level_score': instance_result.get('instance_level_score', 0.0),  # 1.0 if all passed, 0.0 otherwise
                        'if_indices': if_indices,  # List of all constraint IDs
                        'parameters_list': parameters_list,  # List of all constraint parameters
                        'individual_scores': individual_scores,  # Score for each individual constraint
                        'n_constraints': len(constraint_details)  # Number of constraints for this instance
                    })
            
            simplified_results['results'][instruction_key] = {
                'n_instructions': n_instructions,
                'overall_scores': overall_scores,
                'instances': instances
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
    
    def _format_table(self, headers: List[str], rows: List[List[str]], title: str = None) -> List[str]:
        """Format a table using PrettyTable."""
        if not rows:
            return []
        
        table = PrettyTable()
        table.field_names = headers
        
        for row in rows:
            # Ensure row has same length as headers
            formatted_row = [str(cell) for cell in row]
            while len(formatted_row) < len(headers):
                formatted_row.append("")
            table.add_row(formatted_row[:len(headers)])
        
        table_lines = []
        
        if title:
            table_lines.append(title)
            table_lines.append("-" * len(title))
            table_lines.append("")
        
        # Convert table to string and split by lines
        table_str = str(table)
        table_lines.extend(table_str.split('\n'))
        
        return table_lines
    
    def generate_summary_text(self, results: Dict[str, Any], output_path: str, eval_functional: bool = True, eval_if: bool = True):
        """Generate a text summary of the evaluation results."""
        text_content = []
        
        # Header
        text_content.append("=" * 80)
        text_content.append("EVALUATION SUMMARY")
        text_content.append("=" * 80)
        text_content.append("")
        text_content.append(f"Task: {results.get('task', 'unknown')}")
        text_content.append(f"Benchmark: {results.get('benchmark', 'unknown')}")
        text_content.append(f"Model: {results.get('model', 'unknown')}")
        text_content.append(f"Max Instructions: {results.get('max_instructions', 0)}")
        
        # Show evaluation mode
        modes = []
        if eval_functional:
            modes.append("functional correctness")
        if eval_if:
            modes.append("instruction following")
        text_content.append(f"Evaluation mode: {' + '.join(modes)}")
        text_content.append("")
        
        # Overall summary table
        headers = ["Instructions"]
        if eval_functional:
            headers.append("Pass@1")
        if eval_if:
            headers.extend(["Constraint-IF", "Instance-IF"])
        
        summary_rows = []
        evaluation_results = results.get('results', {})
        
        for i in range(results.get('max_instructions', 0) + 1):
            key = f"{i}_instructions"
            if key in evaluation_results:
                result = evaluation_results[key]
                row = [str(i)]
                
                if eval_functional:
                    pass_at_1 = result.get('functional', {}).get('pass_at_1', 0.0)
                    row.append(f"{pass_at_1:.4f}")
                
                if eval_if:
                    if i == 0:
                        # Use "-" for 0-instruction IF scores
                        row.extend(["-", "-"])
                    else:
                        constraint_score = result.get('if_evaluation', {}).get('overall_constraint_score', 0.0)
                        instance_score = result.get('if_evaluation', {}).get('overall_instance_score', 0.0)
                        row.extend([f"{constraint_score:.4f}", f"{instance_score:.4f}"])
                
                summary_rows.append(row)
        
        table_lines = self._format_table(headers, summary_rows, "OVERALL PERFORMANCE SUMMARY")
        text_content.extend(table_lines)
        text_content.append("")
        text_content.append("")
        
        # Detailed analysis for each instruction count
        for i in range(results.get('max_instructions', 0) + 1):
            key = f"{i}_instructions"
            if key in evaluation_results:
                result = evaluation_results[key]
                text_content.append("=" * 60)
                text_content.append(f"{i}-INSTRUCTIONS ANALYSIS")
                text_content.append("=" * 60)
                text_content.append("")
                
                # Functional results
                if eval_functional:
                    func_result = result.get('functional', {})
                    text_content.append(f"Functional Correctness: {func_result.get('pass_at_1', 0.0):.4f}")
                    
                    # Add LiveCodeBench difficulty breakdown if available
                    if 'difficulty_breakdown' in func_result:
                        text_content.append("Difficulty Breakdown:")
                        for diff, stats in func_result['difficulty_breakdown'].items():
                            pass_rate = stats.get('pass_rate', 0.0)
                            total = stats.get('total', 0)
                            passed = stats.get('passed', 0)
                            text_content.append(f"  {diff}: {pass_rate:.4f} ({passed}/{total})")
                    
                    text_content.append("")
                
                # IF results (skip for 0-instructions)
                if eval_if and i > 0:
                    if_result = result.get('if_evaluation', {})
                    text_content.append(f"Constraint-Level Score: {if_result.get('overall_constraint_score', 0.0):.4f}")
                    text_content.append(f"Instance-Level Score: {if_result.get('overall_instance_score', 0.0):.4f}")
                    text_content.append("")
                    
                    # Category breakdown (sorted by score desc, then count desc)
                    if 'category_summary' in if_result:
                        category_data = []
                        for category, data in if_result['category_summary'].items():
                            score = data.get('constraint_level_score', 0.0)
                            count = data.get('count', 0)
                            category_data.append((category, score, count))
                        
                        # Sort by score desc, then count desc
                        category_data.sort(key=lambda x: (-x[1], -x[2]))
                        
                        cat_rows = []
                        for category, score, count in category_data:
                            cat_rows.append([category, f"{score:.4f}", str(count)])
                        
                        cat_table = self._format_table(["Category", "Score", "Count"], cat_rows, "Performance by Category")
                        text_content.extend(cat_table)
                        text_content.append("")
                    
                    # Constraint type breakdown (sorted by score desc, then count desc)
                    if 'constraint_type_summary' in if_result:
                        constraint_data = []
                        for constraint_id, data in if_result['constraint_type_summary'].items():
                            score = data.get('constraint_level_score', 0.0)
                            count = data.get('count', 0)
                            constraint_data.append((constraint_id, score, count))
                        
                        # Sort by score desc, then count desc
                        constraint_data.sort(key=lambda x: (-x[1], -x[2]))
                        
                        const_rows = []
                        for constraint_id, score, count in constraint_data:
                            const_rows.append([constraint_id, f"{score:.4f}", str(count)])
                        
                        const_table = self._format_table(["IF Index", "Score", "Count"], const_rows, "Performance by IF Index")
                        text_content.extend(const_table)
                        text_content.append("")
                
                text_content.append("")
        
        # Write text file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
        
        print(f"Summary saved to: {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation system for code generation/editing with instruction following",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate both functional correctness and instruction following (default)
  python3 -m swe_if.evaluator --task generation --benchmark bigcodebench --model gemini-3.1-pro-preview --n-instructions 5 --run-all
  
  # Evaluate only functional correctness
  python3 -m swe_if.evaluator --task generation --benchmark bigcodebench --model claude-sonnet-4 --n-instructions 3 --functional-only
  
  # Evaluate only instruction following
  python3 -m swe_if.evaluator --task generation --benchmark bigcodebench --model claude-sonnet-4 --n-instructions 3 --if-only
  
  # Evaluate single instruction count (both metrics)
  python3 -m swe_if.evaluator --task generation --benchmark bigcodebench --model claude-sonnet-4 --n-instructions 3
  
  # Evaluate editing task with custom taxonomy
  python3 -m swe_if.evaluator --task editing --benchmark bigcodebench --model gemini-3.1-pro-preview --n-instructions 3 --run-all --taxonomy-path custom_taxonomy.csv
  
  # Evaluate LiveCodeBench
  python3 -m swe_if.evaluator --task generation --benchmark livecodebench --model claude-sonnet-4 --n-instructions 3 --run-all
  
  # Overwrite existing output files
  python3 -m swe_if.evaluator --task generation --benchmark bigcodebench --model claude-sonnet-4 --n-instructions 3 --run-all --overwrite
  
  # Use custom thread count for faster IF evaluation
  python3 -m swe_if.evaluator --task generation --benchmark bigcodebench --model claude-sonnet-4 --n-instructions 3 --run-all --n-threads 64

Evaluation Modes:
  - Default: Both functional correctness and instruction following
  - --functional-only: Only functional correctness (Pass@1)
  - --if-only: Only instruction following (constraint-level and instance-level scores)
        """
    )
    
    parser.add_argument("--task", required=True, choices=['generation', 'editing'], help="Task type")
    parser.add_argument("--benchmark", required=True, choices=['bigcodebench', 'livecodebench'], help="Benchmark dataset")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--n-instructions", type=int, required=True, help="Number of instructions")
    parser.add_argument("--run-all", action="store_true", help="Evaluate from 0 to n-instructions")
    parser.add_argument("--taxonomy-path", default="data/code_gym_taxonomy.csv", help="Path to taxonomy CSV file")
    parser.add_argument("--output-dir", default="scores", help="Output directory for results")
    parser.add_argument("--n-threads", type=int, default=None, help="Number of threads for IF evaluation (default: min(64, cpu_count * 4))")
    
    # Metric selection arguments
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument("--functional-only", action="store_true", help="Evaluate functional correctness only")
    metric_group.add_argument("--if-only", action="store_true", help="Evaluate instruction following only")
    # Default: evaluate both (no flag needed)
    
    # File overwrite control
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    # Determine which metrics to evaluate
    eval_functional = not args.if_only  # True unless if-only is specified
    eval_if = not args.functional_only  # True unless functional-only is specified
    
    # Setup output paths
    # Sanitize model name for output filenames (matches code_generator and input paths)
    args.model = args.model.replace('/', '__')
    output_dir = Path(args.output_dir) / args.benchmark / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.functional_only:
        suffix = "_func"
    elif args.if_only:
        suffix = "_if"
    else:
        suffix = ""
    
    if args.run_all:
        json_path = output_dir / f"{args.model}_0-{args.n_instructions}-instructions{suffix}.json"
        txt_path = output_dir / f"{args.model}_0-{args.n_instructions}-instructions{suffix}_summary.txt"
    else:
        json_path = output_dir / f"{args.model}_{args.n_instructions}-instructions{suffix}.json"
        txt_path = output_dir / f"{args.model}_{args.n_instructions}-instructions{suffix}_summary.txt"
    
    # Check if output files already exist
    existing_files = []
    if json_path.exists():
        existing_files.append(str(json_path))
    if txt_path.exists():
        existing_files.append(str(txt_path))
    
    if existing_files and not args.overwrite:
        print("Error: Output files already exist:")
        for file_path in existing_files:
            print(f"  - {file_path}")
        print("\nUse --overwrite to overwrite existing files, or choose a different output location.")
        return 1
    
    # Initialize evaluator
    evaluator = CodeEvaluator(
        taxonomy_path=args.taxonomy_path,
        eval_functional=eval_functional,
        eval_if=eval_if,
        n_threads=args.n_threads
    )
    
    # Print evaluation mode
    modes = []
    if eval_functional:
        modes.append("functional correctness")
    if eval_if:
        modes.append("instruction following")
    print(f"Evaluation mode: {' + '.join(modes)}")
    if eval_if:
        thread_count = args.n_threads or min(64, (os.cpu_count() or 1) * 4)
        print(f"Using {thread_count} threads for IF evaluation")
    print()
    
    if args.run_all:
        # Evaluate range
        print(f"Evaluating {args.task} task on {args.benchmark} with {args.model} for 0 to {args.n_instructions} instructions")
        results = evaluator.evaluate_range(args.task, args.benchmark, args.model, args.n_instructions)
        
    else:
        # Evaluate single
        print(f"Evaluating {args.task} task on {args.benchmark} with {args.model} for {args.n_instructions} instructions")
        
        result = evaluator.evaluate_single(args.task, args.benchmark, args.model, args.n_instructions)
        
        # Wrap single result in range format for consistency
        results = {
            'task': args.task,
            'benchmark': args.benchmark,
            'model': args.model,
            'max_instructions': args.n_instructions,
            'results': {f"{args.n_instructions}_instructions": result}
        }
    
    # Save results and generate summary
    evaluator.save_results(results, str(json_path))
    evaluator.generate_summary_text(results, str(txt_path), eval_functional, eval_if)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    print(f"Evaluation mode: {' + '.join(modes)}")
    
    if args.run_all:
        print(f"Instructions evaluated: 0 to {args.n_instructions}")
        
        # Print summary table using PrettyTable
        table = PrettyTable()
        
        # Dynamic header based on evaluation mode
        header_parts = ['Instructions']
        if eval_functional:
            header_parts.append('Pass@1')
        if eval_if:
            header_parts.extend(['Constraint-IF', 'Instance-IF'])
        
        table.field_names = header_parts
        
        for i in range(args.n_instructions + 1):
            key = f"{i}_instructions"
            if key in results['results']:
                result = results['results'][key]
                row_parts = [str(i)]
                
                if eval_functional:
                    pass_at_1 = result.get('functional', {}).get('pass_at_1', 0.0)
                    row_parts.append(f"{pass_at_1:.4f}")
                
                if eval_if:
                    if i == 0:
                        # Use "-" for 0-instruction IF scores
                        row_parts.extend(["-", "-"])
                    else:
                        constraint_score = result.get('if_evaluation', {}).get('overall_constraint_score', 0.0)
                        instance_score = result.get('if_evaluation', {}).get('overall_instance_score', 0.0)
                        row_parts.extend([f"{constraint_score:.4f}", f"{instance_score:.4f}"])
                
                table.add_row(row_parts)
        
        print("\nSummary Table:")
        print(table)
        
    else:
        print(f"Instructions evaluated: {args.n_instructions}")
        result = results['results'][f"{args.n_instructions}_instructions"]
        
        # Create single-row table for consistency
        table = PrettyTable()
        header_parts = ['Instructions']
        row_parts = [str(args.n_instructions)]
        
        if eval_functional:
            pass_at_1 = result.get('functional', {}).get('pass_at_1', 0.0)
            header_parts.append('Pass@1')
            row_parts.append(f"{pass_at_1:.4f}")
        
        if eval_if:
            constraint_score = result.get('if_evaluation', {}).get('overall_constraint_score', 0.0)
            instance_score = result.get('if_evaluation', {}).get('overall_instance_score', 0.0)
            header_parts.extend(['Constraint-IF', 'Instance-IF'])
            if args.n_instructions == 0:
                row_parts.extend(["-", "-"])
            else:
                row_parts.extend([f"{constraint_score:.4f}", f"{instance_score:.4f}"])
        
        table.field_names = header_parts
        table.add_row(row_parts)
        print(table)
    
    print(f"\nDetailed results: {json_path}")
    print(f"Summary report: {txt_path}")


if __name__ == "__main__":
    main()
