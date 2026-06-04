#!/usr/bin/env python3
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

"""
Code Generator for Coding Benchmarks with Instruction Following.
Supports two task types: generation and editing, with optional concurrent processing.
"""

import json
import argparse
import re
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datasets import load_dataset

from .llm_interface import create_llm_interface
from .concurrent_processor import create_concurrent_processor


class CodeGenerator:
    """Code generator for coding benchmarks with instruction following capabilities.
    
    Supports two main tasks:
    - generation: Generate code with 0 to n instructions (use --run-all for complete range)
    - editing: Edit code step-by-step with instructions (edit-1, edit-2, ...)
    
    Features concurrent processing for improved performance.
    """
    
    SUPPORTED_DATASETS = {
        'bigcodebench': ('bigcode/bigcodebench', 'v0.1.4'),
        'livecodebench': ('livecodebench/code_generation_lite', 'release_v6'),  # LiveCodeBench v6
    }
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str = "bigcodebench",
        max_retries: int = 3,
        output_dir: str = "results",
        enable_concurrent: bool = False,
    ):
        """
        Initialize the code generator.
        
        Args:
            model_name: Name of the LLM model
            dataset_name: Name of the dataset
            max_retries: Maximum number of retries for LLM generation failures (used by _generate_with_retries)
            output_dir: Base output directory
            enable_concurrent: Whether to enable concurrent processing
            
        Note:
            - max_retries: Controls LLM-level retries (empty responses, parsing errors, API errors)
            - concurrent max_retries: Controls task-level retries (from model_configs.py, for unexpected errors)
            These serve different purposes and operate at different levels.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_retries = max_retries
        self.output_dir = Path(output_dir)
        self.enable_concurrent = enable_concurrent
        
        # Temperature follows the paper: Claude thinking models keep their config value
        # (1.0, required by the API); otherwise Big-SWE-IF uses 0.0, Live-SWE-IF uses 0.2.
        if 'thinking' in model_name:
            eval_temperature = None  # keep the model's config temperature (1.0)
        elif dataset_name == 'livecodebench':
            eval_temperature = 0.2
        else:
            eval_temperature = 0.0
        # Initialize LLM interface
        self.llm = create_llm_interface(
            model=model_name,
            system_instruction=self._get_generation_system_prompt(),
            temperature=eval_temperature,
        )
        
        # Get sanitized model name for file naming (handles models with slashes like "x-ai/grok-4")
        self.sanitized_model_name = self.llm.get_sanitized_model_name()
        
        # Initialize concurrent processor if enabled
        if enable_concurrent:
            self.concurrent_processor = create_concurrent_processor(
                model_name=model_name,
                enable_logging=True
            )
            print(f"Concurrent processing enabled for {model_name}")
        else:
            print(f"Using serial processing for {model_name}")
        
    def _get_generation_system_prompt(self) -> str:
        """Get system prompt for code generation."""
        if self.dataset_name == 'livecodebench':
            return (
                "**Objective:**\n"
                "You are an expert code generation assistant. Your primary objective is to provide functional Python solutions.\n\n"
                "**Output Formats:**\n"
                "You must strictly adhere to one of these two formats:\n\n"
                "1.  **Format 1: Markdown-Wrapped Code**\n"
                "    * **Use Case:** Use this format if your response contains ANY text or explanation in addition to the code.\n"
                "    * **Specification:** Place the entire code solution within the *first* Markdown code block (e.g., ```python ... ```).\n\n"
                "2.  **Format 2: Raw Code Only**\n"
                "    * **Use Case:** Use this format ONLY if your response consists *exclusively* of the code solution.\n"
                "    * **Specification:** Provide the raw code directly, with no Markdown or other text.\n\n"
                "**Requirements:**\n"
                "1. When generating or editing code, satisfy ALL user instructions throughout the entire conversation while keeping the functionality intact.\n"
                "2. Always include **complete Python functions** in every response, even if no changes are needed from the previous version."
            )
        else:
            # Original system prompt for BigCodeBench
            return (
                "**Objective:**\n"
                "You are an expert code generation assistant. Your primary objective is to provide a complete and runnable code solution for every request.\n\n"
                "**Output Formats:**\n"
                "You must strictly adhere to one of these two formats:\n\n"
                "1.  **Format 1: Markdown-Wrapped Code**\n"
                "    * **Use Case:** Use this format if your response contains ANY text or explanation in addition to the code.\n"
                "    * **Specification:** Place the entire code solution within the *first* Markdown code block (e.g., ```python ... ```).\n\n"
                "2.  **Format 2: Raw Code Only**\n"
                "    * **Use Case:** Use this format ONLY if your response consists *exclusively* of the code solution.\n"
                "    * **Specification:** Provide the raw code directly, with no Markdown or other text.\n\n"
                "**Requirements:**\n"
                "1. When generating or editing code, satisfy ALL user instructions throughout the entire conversation while keeping the functionality intact.\n"
                "2. Always include **complete, runnable code** in every response, even if no changes are needed from the previous version."
            )
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract code from LLM response."""
        if not response_text:
            return ""

        code_patterns = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)
        matches = code_patterns.findall(response_text)

        if matches:
            for match_content in matches:
                content = match_content
                stripped_content = content.strip()
                if stripped_content:
                    return content
            return ""
        else:
            return response_text
    
    def _get_instance_id(self, instance: Dict[str, Any]) -> str:
        """Get the appropriate ID field for the instance based on dataset."""
        if self.dataset_name == 'livecodebench':
            return instance.get('question_id', 'unknown')
        else:
            return instance.get('task_id', 'unknown')
    
    def _get_user_query(self, instance: Dict[str, Any]) -> str:
        """Get the appropriate user query field based on dataset."""
        if self.dataset_name == 'livecodebench':
            return instance.get('question_content', '')
        else:
            return instance.get('instruct_prompt', '')
    
    def _check_is_stdin(self, instance: Dict[str, Any]) -> bool:
        """Check if the instance is stdin type."""
        if self.dataset_name == 'livecodebench':
            # For LiveCodeBench, check public_test_cases for stdin type
            public_tests = instance.get('public_test_cases', '[]')
            if isinstance(public_tests, str):
                try:
                    test_list = json.loads(public_tests)
                    for test in test_list:
                        if test.get("testtype") == "stdin":
                            return True
                except:
                    pass
            return False
        else:
            # BigCodeBench doesn't have stdin type
            return False
    
    def _create_minimal_instance_info(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal instance info for storage optimization."""
        if self.dataset_name == 'livecodebench':
            # For LiveCodeBench, only store essential fields
            return {
                'question_id': instance.get('question_id'),
                'difficulty': instance.get('difficulty', 'unknown'),
                'is_stdin': self._check_is_stdin(instance)
            }
        else:
            # For BigCodeBench, keep full instance (smaller datasets)
            return dict(instance)
    
    def _create_livecodebench_prompt(self, original_prompt: str, is_stdin: bool) -> str:
        """Create LiveCodeBench specific prompt with stdin/functional prefix."""
        if is_stdin:
            instructions = (
                "**Problem Type: Standard Input/Output**\n\n"
                "**Your Task:**\n"
                "Write an executable Python function that solves the problem described in the prompt below.\n\n"
                "**Requirements:**\n"
                "- The function must read all necessary input from `stdin`.\n"
                "- The function must print the final output to `stdout`.\n"
                "- Simply call the function after the definition.\n\n"
                "**Evaluation:**\n"
                "We will evaluate your solution by running the code and comparing its standard output with the expected solution."
            )
            # instructions = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
        else:
            instructions = (
                "**Problem Type: Functional Implementation**\n\n"
                "**Your Task:**\n"
                "Implement a function to solve the problem described in the prompt below.\n\n"
                "**Requirements:**\n"
                "- The primary function, which takes arguments as input and returns the final result, **must be named `solve()`**.\n"
                "- The function **must NOT read from standard input** (e.g., using `input()` or `sys.stdin`). All required data will be passed in as function arguments.\n"
                "- Any helper functions are permitted.\n"
                "- Your code must only contain function definitions. **Strictly do not include a call to `solve()`**.\n\n"
                "**Evaluation:**\n"
                "We will evaluate your solution by first executing your code to load the function definitions, and then calling your `solve()` function directly with various test cases."
            )
            # instructions = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
        
        return f"{instructions}\n\n---\n**Problem:**\n{original_prompt}"
    
    def _load_dataset(self, input_data_file: str = None) -> List[Dict[str, Any]]:
        """Load augmented SWE-IF instances (original benchmark fields + an
        `instruction_list`). Loads from HF `MingZhong/SWE-IF` by default; if
        `input_data_file` names a local JSONL under `data/<dataset>/`, that file
        is used instead (e.g. after you re-run the augmentation yourself)."""
        from .data import load_benchmark
        local_file = None
        if input_data_file:
            candidate = self.output_dir.parent / "data" / self.dataset_name / input_data_file
            if candidate.exists():
                local_file = str(candidate)
        return load_benchmark(self.dataset_name, local_file=local_file)
    
    def _load_existing_task_ids(self, output_file: Path) -> set:
        """Load existing task IDs from output file."""
        existing_task_ids = set()
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if 'task_id' in data:
                                existing_task_ids.add(data['task_id'])
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON at line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error reading existing file: {e}")
        return existing_task_ids
    
    def _load_previous_edit_response(self, prev_edit_file: Path, task_id: str) -> str:
        """Load full_response from previous edit file for a specific task_id."""
        if not prev_edit_file.exists():
            print(f"Warning: Previous edit file not found: {prev_edit_file}")
            return ""
        
        try:
            with open(prev_edit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get('task_id') == task_id:
                            return data.get('full_response', '')
        except Exception as e:
            print(f"Warning: Error reading previous edit file {prev_edit_file}: {e}")
        
        print(f"Warning: Could not find task_id {task_id} in {prev_edit_file}")
        return ""
    
    def _create_generation_prompt(
        self, 
        original_prompt: str, 
        instructions: List[Dict[str, Any]],
        is_stdin: bool = False
    ) -> str:
        """Create prompt for code generation task."""
        # For LiveCodeBench, add the stdin/functional prefix
        if self.dataset_name == 'livecodebench':
            original_prompt = self._create_livecodebench_prompt(original_prompt, is_stdin)
        
        if not instructions:
            return original_prompt
        
        prompt = original_prompt + "\n\n"
        
        if len(instructions) == 1:
            prompt += f"Additionally, ensure your code meets the following requirement: {instructions[0]['generation_prompt']}"
        else:
            prompt += "Additionally, ensure your code meets the following requirements:"
            for i, instruction in enumerate(instructions, 1):
                prompt += f"\n{i}. {instruction['generation_prompt']}"
        
        return prompt
    
    def _create_editing_prompt(self, instruction: Dict[str, Any]) -> str:
        """Create prompt for code editing task."""
        return instruction['edit_prompt']
    
    def _build_if_info_generation(self, instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build IF_INFO for generation task."""
        if not instructions:
            return {}
        
        # Create combined prompt
        if len(instructions) == 1:
            prompt = f"Also, generate the code to meet the following requirement: {instructions[0]['generation_prompt']}"
        else:
            prompt = "Also, generate the code to meet the following requirements:"
            for i, instruction in enumerate(instructions, 1):
                prompt += f"\n{i}. {instruction['generation_prompt']}"
        
        constraints = []
        for instruction in instructions:
            constraints.append({
                "index": instruction['instruction_idx'],
                "parameters": instruction['parameters']
            })
        
        return {
            "prompt": prompt,
            "constraints": constraints
        }
    
    def _build_if_info_editing(
        self, 
        instructions: List[Dict[str, Any]], 
        current_edit: int
    ) -> Dict[str, Any]:
        """Build IF_INFO for editing task (cumulative)."""
        if not instructions or current_edit <= 0:
            return {}
        
        # Include instructions up to current edit
        relevant_instructions = instructions[:current_edit]
        
        prompts = []
        constraints = []
        
        for instruction in relevant_instructions:
            prompts.append(instruction['edit_prompt'])
            constraints.append({
                "index": instruction['instruction_idx'],
                "parameters": instruction['parameters']
            })
        
        return {
            "prompts": prompts,
            "constraints": constraints
        }
    
    def _generate_with_retries(
        self, 
        contents: List[Dict[str, Any]], 
        task_id: str
    ) -> Tuple[str, str, bool]:
        """
        Generate response with retries for LLM-related failures.
        
        This method handles all LLM-specific errors and retries at the generation level:
        - Empty responses from LLM
        - Code extraction failures  
        - Network/API errors
        - Response parsing issues
        
        Args:
            contents: Messages to send to LLM
            task_id: Task identifier for logging
            
        Returns:
            Tuple of (solution, full_response, success)
            - solution: Extracted code or empty string
            - full_response: Full LLM response or error message
            - success: True if generation succeeded, False if failed after all retries
        """
        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate_response(contents=contents)
                
                if not response:
                    raise ValueError(f"LLM returned empty response")
                
                solution = self._extract_code_from_response(response)
                
                if not solution:
                    raise ValueError(f"Could not extract code from response")
                
                return solution, response, True
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)[:150]}"
                if self.enable_concurrent:
                    print(f"Task {task_id}, {error_msg}")
                else:
                    tqdm.write(f"Task {task_id}, {error_msg}")
                
                if attempt < self.max_retries - 1:
                    # time.sleep(min(16, 2**(attempt + 1)))
                    time.sleep(16 * (attempt + 1))
                else:
                    final_error = f"Failed after {self.max_retries} attempts. Last error: {str(e)}"
                    return "", final_error, False
        
        return "", "Unknown error", False
    
    def _process_generation_instance(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single generation instance. Used for concurrent processing.
        
        This function handles data preparation and calls LLM generation.
        LLM failures are handled by _generate_with_retries and returned as failed results.
        Only unexpected errors (data processing, format issues) should raise exceptions.
        """
        instance = instance_data['instance']
        n_instructions = instance_data['n_instructions']
        
        task_id = self._get_instance_id(instance)
        original_prompt = self._get_user_query(instance)
        is_stdin = self._check_is_stdin(instance)
        
        if n_instructions == 0:
            # No instructions, just original prompt
            generation_prompt = self._create_generation_prompt(original_prompt, [], is_stdin)
            instruction_list = []
            actual_instruction_count = 0
        else:
            # Use first n_instructions instructions
            available_instructions = instance.get('instruction_list', [])
            instruction_list = available_instructions[:n_instructions]
            actual_instruction_count = len(instruction_list)
            
            # Safety check
            if actual_instruction_count < n_instructions:
                print(f"Warning: Task {task_id} only has {actual_instruction_count} instructions, expected {n_instructions}")
            
            generation_prompt = self._create_generation_prompt(original_prompt, instruction_list, is_stdin)
        
        contents = [{'role': 'user', 'parts': [{'text': generation_prompt}]}]
        
        # LLM generation with retries - this handles all LLM-related failures internally
        solution, full_response, success = self._generate_with_retries(contents, task_id)
        
        # Build IF_INFO
        if_info = self._build_if_info_generation(instruction_list)
        
        # Create minimal instance info (optimized for LiveCodeBench)
        minimal_instance = self._create_minimal_instance_info(instance)
        
        result = {
            "task_id": task_id,
            "solution": solution,
            "full_response": full_response,
            "original_instance": minimal_instance,
            "if_info": if_info
        }
        
        if not success:
            result["status"] = "failed_after_retries"
        
        # Always return result, don't raise exception for LLM failures
        # Only unexpected errors (data processing issues) should raise exceptions
        return result
    
    def run_generation_task(self, input_data_file: str, n_instructions: int, run_all: bool = False) -> List[str]:
        """Run code generation task with instructions."""
        if run_all:
            if self.enable_concurrent:
                return self._run_generation_concurrent_run_all(input_data_file, n_instructions)
            else:
                return self._run_generation_serial_run_all(input_data_file, n_instructions)
        else:
            if self.enable_concurrent:
                return [self._run_generation_concurrent_single(input_data_file, n_instructions)]
            else:
                return [self._run_generation_serial_single(input_data_file, n_instructions)]
    
    def _run_generation_concurrent_run_all(self, input_data_file: str, n_instructions: int) -> List[str]:
        """Run generation for all instruction counts (0 to n) with concurrent processing."""
        print(f"Running GENERATION task for ALL instruction counts from 0 to {n_instructions} with concurrent processing")
        
        output_files = []
        
        # Process each instruction count sequentially, but with concurrent processing within each
        for current_n in range(0, n_instructions + 1):
            print(f"\n=== Processing {current_n}-instructions with concurrent processing ===")
            
            output_file = self._run_generation_concurrent_single(input_data_file, current_n)
            output_files.append(output_file)
            
            print(f"Completed {current_n}-instructions, moving to next...")
        
        print(f"Run-all generation task completed. Generated {len(output_files)} files (0 to {n_instructions} instructions)")
        return output_files
    
    def _run_generation_serial_run_all(self, input_data_file: str, n_instructions: int) -> List[str]:
        """Run generation for all instruction counts (0 to n) with serial processing."""
        print(f"Running GENERATION task for ALL instruction counts from 0 to {n_instructions}")
        
        output_files = []
        
        for current_n in range(0, n_instructions + 1):
            print(f"\n=== Processing {current_n}-instructions ===")
            
            output_file = self._run_generation_serial_single(input_data_file, current_n)
            output_files.append(output_file)
        
        print(f"Run-all generation task completed. Generated {len(output_files)} files (0 to {n_instructions} instructions)")
        return output_files
    
    def _run_generation_concurrent_single(self, input_data_file: str, n_instructions: int) -> str:
        """Run generation for a single instruction count with concurrent processing."""
        print(f"Running GENERATION task with exactly {n_instructions} instructions (concurrent)")
        
        # Load augmented instances (HF MingZhong/SWE-IF by default; local via --input-data)
        if n_instructions > 0:
            instances_with_instructions = self._load_dataset(input_data_file)
            print(f"Loaded {len(instances_with_instructions)} augmented instances")
        else:
            instances_with_instructions = []
        
        # Setup output directory
        output_dir = self.output_dir / self.dataset_name / "generation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.sanitized_model_name}_{n_instructions}-instructions.jsonl"
        
        # Check existing
        existing_task_ids = self._load_existing_task_ids(output_file)
        
        # Prepare instances
        if n_instructions == 0:
            # Load original dataset
            instances = self._load_dataset()
            instances_to_process = [
                {'instance': inst, 'n_instructions': n_instructions}
                for inst in instances 
                if self._get_instance_id(inst) not in existing_task_ids and self._get_user_query(inst)
            ]
            print(f"Loaded {len(instances)} instances from original dataset")
        else:
            # Use instances with sufficient instructions
            instances_to_process = [
                {'instance': inst, 'n_instructions': n_instructions}
                for inst in instances_with_instructions 
                if (self._get_instance_id(inst) not in existing_task_ids and 
                    len(inst.get('instruction_list', [])) >= n_instructions)
            ]
            
            total_with_instructions = sum(1 for inst in instances_with_instructions 
                                        if self._get_instance_id(inst) not in existing_task_ids)
            insufficient_count = total_with_instructions - len(instances_to_process)
            
            if insufficient_count > 0:
                print(f"  (Skipping {insufficient_count} instances with < {n_instructions} instructions)")
        
        print(f"Processing {len(instances_to_process)} new instances for {n_instructions}-instructions")
        
        if not instances_to_process:
            print(f"No new instances to process for {n_instructions}-instructions")
            return str(output_file)
        
        # Process concurrently
        successful_count = self.concurrent_processor.process_and_append_jsonl(
            items=instances_to_process,
            process_func=self._process_generation_instance,
            output_file=output_file,
            get_item_id=lambda x: self._get_instance_id(x['instance']),
            progress_desc=f"Generation {n_instructions}-instructions"
        )
        
        print(f"Concurrent generation completed: {successful_count}/{len(instances_to_process)} succeeded")
        return str(output_file)
    
    def _run_generation_serial_single(self, input_data_file: str, n_instructions: int) -> str:
        """Run generation for a single instruction count with serial processing."""
        print(f"Running GENERATION task with exactly {n_instructions} instructions")
        
        # Load augmented instances (HF MingZhong/SWE-IF by default; local via --input-data)
        if n_instructions > 0:
            instances_with_instructions = self._load_dataset(input_data_file)
            print(f"Loaded {len(instances_with_instructions)} augmented instances")
        else:
            instances_with_instructions = []
        
        # Setup output directory
        output_dir = self.output_dir / self.dataset_name / "generation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.sanitized_model_name}_{n_instructions}-instructions.jsonl"
        
        # Check existing
        existing_task_ids = self._load_existing_task_ids(output_file)
        
        # Prepare instances
        if n_instructions == 0:
            # Load original dataset
            instances = self._load_dataset()
            instances_to_process = [
                inst for inst in instances 
                if self._get_instance_id(inst) not in existing_task_ids and self._get_user_query(inst)
            ]
            print(f"Loaded {len(instances)} instances from original dataset")
        else:
            # Use instances with sufficient instructions
            instances_to_process = [
                inst for inst in instances_with_instructions 
                if (self._get_instance_id(inst) not in existing_task_ids and 
                    len(inst.get('instruction_list', [])) >= n_instructions)
            ]
            
            total_with_instructions = sum(1 for inst in instances_with_instructions 
                                        if self._get_instance_id(inst) not in existing_task_ids)
            insufficient_count = total_with_instructions - len(instances_to_process)
            
            if insufficient_count > 0:
                print(f"  (Skipping {insufficient_count} instances with < {n_instructions} instructions)")
        
        print(f"Processing {len(instances_to_process)} new instances for {n_instructions}-instructions")
        
        if not instances_to_process:
            print(f"No new instances to process for {n_instructions}-instructions")
            return str(output_file)
        
        succeeded = 0
        failed = 0
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for instance in tqdm(instances_to_process, desc=f"Generation {n_instructions}-inst"):
                instance_data = {'instance': instance, 'n_instructions': n_instructions}
                
                try:
                    result = self._process_generation_instance(instance_data)
                    
                    # Check if LLM generation was successful
                    if result.get('status') == 'failed_after_retries':
                        failed += 1
                    else:
                        succeeded += 1
                    
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    
                except Exception as e:
                    # Handle unexpected errors (data processing, format issues, etc.)
                    # These are different from LLM failures which are handled internally
                    failed += 1
                    minimal_instance = self._create_minimal_instance_info(instance)
                    failed_result = {
                        "task_id": self._get_instance_id(instance),
                        "solution": "",
                        "full_response": "",
                        "original_instance": minimal_instance,
                        "if_info": {},
                        "status": "processing_error",
                        "processing_error": f"Unexpected error during processing: {str(e)}"
                    }
                    f.write(json.dumps(failed_result) + '\n')
                    f.flush()
        
        print(f"{n_instructions}-instructions completed: {succeeded} succeeded, {failed} failed")
        return str(output_file)
    
    def run_editing_task(self, input_data_file: str, n_instructions: int) -> List[str]:
        """Run code editing task with instructions."""
        print(f"Running EDITING task with {n_instructions} edit instructions")
        
        # Load baseline results (0-instructions)
        baseline_file = self.output_dir / self.dataset_name / "generation" / f"{self.sanitized_model_name}_0-instructions.jsonl"
        
        # Load baseline results
        baseline_results = {}
        successful_baselines = 0
        with open(baseline_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    baseline_results[item['task_id']] = item
                    if item.get('solution'):
                        successful_baselines += 1
        
        print(f"Loaded {len(baseline_results)} baseline results from {baseline_file}")
        print(f"  - {successful_baselines} with non-empty solutions")
        print(f"  - {len(baseline_results) - successful_baselines} with empty solutions (will still attempt editing)")
        
        # Load augmented instances (HF MingZhong/SWE-IF by default; local via --input-data)
        instructions_data = {}
        instructions_with_list = 0
        for item in self._load_dataset(input_data_file):
            task_id = self._get_instance_id(item)
            instructions_data[task_id] = item
            if item.get('instruction_list'):
                instructions_with_list += 1

        print(f"Loaded {len(instructions_data)} augmented instances")
        print(f"  - {instructions_with_list} with non-empty instruction lists")

        # Hot-patch baseline results with missing question_content
        if self.dataset_name == 'livecodebench':
            print("Loading full LiveCodeBench dataset to patch missing question content...")
            full_dataset_instances = self._load_dataset()
            content_map = {self._get_instance_id(inst): self._get_user_query(inst) for inst in full_dataset_instances}
            
            patched_count = 0
            for task_id, baseline in baseline_results.items():
                if 'original_instance' in baseline and task_id in content_map:
                    # Inject the missing question_content back into the minimal instance
                    baseline['original_instance']['question_content'] = content_map[task_id]
                    patched_count += 1
            print(f"Patching complete. Injected content for {patched_count} instances.")

        # Find common task_ids
        common_task_ids = set(baseline_results.keys()) & set(instructions_data.keys())
        print(f"Found {len(common_task_ids)} instances with both baseline and instruction data")
        
        output_files = []
        
        for edit_num in range(1, n_instructions + 1):
            print(f"\nProcessing edit-{edit_num}/{n_instructions}")
            
            # Setup output for this edit
            output_dir = self.output_dir / self.dataset_name / "editing"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{self.sanitized_model_name}_{edit_num}-instructions.jsonl"
            output_files.append(str(output_file))
            
            # Check existing
            existing_task_ids = self._load_existing_task_ids(output_file)
            
            # Find task_ids to process for this edit
            task_ids_to_process = []
            for task_id in common_task_ids:
                if task_id in existing_task_ids:
                    continue
                
                instruction_list = instructions_data[task_id].get('instruction_list', [])
                if len(instruction_list) >= edit_num:
                    task_ids_to_process.append(task_id)
            
            total_available = len([tid for tid in common_task_ids if tid not in existing_task_ids])
            skipped_insufficient = total_available - len(task_ids_to_process)
            
            print(f"Processing {len(task_ids_to_process)} new instances for edit-{edit_num}")
            if skipped_insufficient > 0:
                print(f"  (Skipping {skipped_insufficient} instances with < {edit_num} instructions)")
            
            if not task_ids_to_process:
                print(f"No new instances to process for edit-{edit_num}")
                continue
            
            # Process editing instances
            if self.enable_concurrent:
                # Prepare data for concurrent processing
                editing_items = []
                for task_id in task_ids_to_process:
                    editing_items.append({
                        'task_id': task_id,
                        'edit_num': edit_num,
                        'baseline_result': baseline_results[task_id],
                        'instruction_item': instructions_data[task_id]
                    })
                
                successful_count = self.concurrent_processor.process_and_append_jsonl(
                    items=editing_items,
                    process_func=self._process_editing_instance,
                    output_file=output_file,
                    get_item_id=lambda x: x['task_id'],
                    progress_desc=f"Edit-{edit_num}"
                )
                
                print(f"Edit-{edit_num} completed: {successful_count}/{len(task_ids_to_process)} succeeded")
            else:
                # Serial processing
                succeeded = 0
                failed = 0
                
                with open(output_file, 'a', encoding='utf-8') as f:
                    for task_id in tqdm(task_ids_to_process, desc=f"Edit-{edit_num}"):
                        editing_item = {
                            'task_id': task_id,
                            'edit_num': edit_num,
                            'baseline_result': baseline_results[task_id],
                            'instruction_item': instructions_data[task_id]
                        }
                        
                        try:
                            result = self._process_editing_instance(editing_item)
                            
                            # Check if LLM editing was successful
                            if result.get('status') == 'failed_after_retries':
                                failed += 1
                            else:
                                succeeded += 1
                            
                            f.write(json.dumps(result) + '\n')
                            f.flush()
                            
                        except Exception as e:
                            # Handle unexpected errors (data processing, format issues, etc.)
                            # These are different from LLM failures which are handled internally
                            failed += 1
                            failed_result = {
                                "task_id": task_id,
                                "solution": "",
                                "full_response": "",
                                "original_instance": baseline_results[task_id]['original_instance'],
                                "messages": [],
                                "current_edit_prompt": "",
                                "edit_num": edit_num,
                                "if_info": {},
                                "status": "processing_error",
                                "processing_error": f"Unexpected error during processing: {str(e)}"
                            }
                            f.write(json.dumps(failed_result) + '\n')
                            f.flush()
                
                print(f"Edit-{edit_num} completed: {succeeded} succeeded, {failed} failed")
        
        print(f"Editing task completed. Generated {len(output_files)} edit files")
        return output_files
    
    def _process_editing_instance(self, editing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single editing instance. Used for both serial and concurrent processing.
        
        This function handles conversation history building and calls LLM editing.
        LLM failures are handled by _generate_with_retries and returned as failed results.
        Only unexpected errors (data processing, format issues) should raise exceptions.
        """
        task_id = editing_data['task_id']
        edit_num = editing_data['edit_num']
        baseline_result = editing_data['baseline_result']
        instruction_item = editing_data['instruction_item']
        
        # Build conversation history
        original_prompt = self._get_user_query(baseline_result['original_instance'])
        
        # Use the stored 'is_stdin' value instead of re-calculating it
        # The original `_check_is_stdin` would fail because 'public_test_cases' is not in the minimal instance.
        # This now correctly uses the value that was calculated and stored in the 0-instructions file.
        is_stdin = baseline_result['original_instance'].get('is_stdin', False)
        
        # For LiveCodeBench, add the prefix to the original prompt
        if self.dataset_name == 'livecodebench':
            original_prompt = self._create_livecodebench_prompt(original_prompt, is_stdin)
        
        instruction_list = instruction_item['instruction_list'][:edit_num]
        
        # Start with original prompt and baseline response
        messages = [
            {'role': 'user', 'parts': [{'text': original_prompt}]},
            {'role': 'model', 'parts': [{'text': baseline_result['full_response']}]}
        ]
        
        # Add previous editing instructions and responses
        for i in range(edit_num - 1):
            edit_prompt = self._create_editing_prompt(instruction_list[i])
            messages.append({'role': 'user', 'parts': [{'text': edit_prompt}]})
            
            # Load previous edit response
            prev_edit_file = self.output_dir / self.dataset_name / "editing" / f"{self.sanitized_model_name}_{i+1}-instructions.jsonl"
            prev_full_response = self._load_previous_edit_response(prev_edit_file, task_id)
            
            messages.append({'role': 'model', 'parts': [{'text': prev_full_response}]})
        
        # Add current editing instruction
        current_edit_prompt = self._create_editing_prompt(instruction_list[edit_num - 1])
        messages.append({'role': 'user', 'parts': [{'text': current_edit_prompt}]})
        
        # LLM editing with retries - this handles all LLM-related failures internally
        solution, full_response, success = self._generate_with_retries(messages, task_id)
        
        # Add current response to complete history
        messages.append({'role': 'model', 'parts': [{'text': full_response}]})
        
        # Build IF_INFO (cumulative)
        if_info = self._build_if_info_editing(instruction_list, edit_num)
        
        result = {
            "task_id": task_id,
            "solution": solution,
            "full_response": full_response,
            "original_instance": baseline_result['original_instance'],
            "messages": messages,
            "current_edit_prompt": current_edit_prompt,
            "edit_num": edit_num,
            "if_info": if_info
        }
        
        if not success:
            result["status"] = "failed_after_retries"
        
        # Always return result, don't raise exception for LLM failures
        # Only unexpected errors (data processing issues) should raise exceptions
        return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate code for benchmarks with optional instruction following",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model", required=True, help="LLM model name")
    parser.add_argument("--dataset", default="bigcodebench", 
                       choices=['bigcodebench', 'livecodebench'], 
                       help="Dataset name")
    parser.add_argument("--task", required=True, choices=['generation', 'editing'], help="Task type")
    parser.add_argument("--input-data", help="Input data filename (required for generation with n>0 and editing)")
    parser.add_argument("--n-instructions", type=int, help="Number of instructions to use")
    parser.add_argument(
        "--run-all", 
        action="store_true", 
        help="[GENERATION ONLY] Run ALL instruction counts from 0 to n-instructions. "
             "This will generate separate files for 0-instructions.jsonl, 1-instructions.jsonl, ..., n-instructions.jsonl. "
             "Without this flag, only the specific n-instructions count will be generated."
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for failed requests")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent processing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == 'generation':
        if args.n_instructions is None:
            parser.error("--n-instructions is required for generation task")
        if args.n_instructions > 0 and not args.input_data:
            parser.error("--input-data is required for generation task when n-instructions > 0")
    
    elif args.task == 'editing':
        if not args.n_instructions:
            parser.error("--n-instructions is required for editing task")
        if not args.input_data:
            parser.error("--input-data is required for editing task")
    
    if args.run_all and args.task != 'generation':
        parser.error("--run-all can only be used with generation task")
    
    # Create code generator
    generator = CodeGenerator(
        model_name=args.model,
        dataset_name=args.dataset,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
        enable_concurrent=args.concurrent
    )
    
    # Run task
    try:
        if args.task == 'generation':
            output_files = generator.run_generation_task(
                args.input_data, 
                args.n_instructions, 
                args.run_all
            )
            if args.run_all:
                print(f"Run-all generation task completed. Generated {len(output_files)} files:")
                for file in output_files:
                    print(f"  {file}")
            else:
                print(f"Generation task completed. Output: {output_files[0]}")
            
        elif args.task == 'editing':
            output_files = generator.run_editing_task(args.input_data, args.n_instructions)
            print(f"Editing task completed. Generated {len(output_files)} edit files:")
            for file in output_files:
                print(f"  {file}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())