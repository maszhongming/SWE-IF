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

import os
import json
import random
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from pathlib import Path
import logging

from .llm_interface import create_llm_interface
from .concurrent_processor import create_concurrent_processor


class InstructionPlugger:
    """
    Class to plug coding instructions into benchmark instances using LLM-based relevance judgment.
    Supports concurrent processing for improved performance.
    """
    
    SUPPORTED_BENCHMARKS = {
        'bigcodebench': ('bigcode/bigcodebench', 'v0.1.4'),
        'livecodebench': ('livecodebench/code_generation_lite', 'release_v6')
    }
    
    def __init__(
        self,
        model_name: str,
        taxonomy_path: str = "data/code_gym_taxonomy.csv",
        output_dir: str = "data",
        enable_logging: bool = True,
        enable_concurrent: bool = False
    ):
        """
        Initialize the InstructionPlugger.
        
        Args:
            model_name: Name of the LLM model to use for relevance judgment
            taxonomy_path: Path to the taxonomy CSV file
            output_dir: Directory to save output files
            enable_logging: Whether to enable detailed logging
            enable_concurrent: Whether to enable concurrent processing
        """
        self.model_name = model_name
        self.sanitized_model_name = model_name.replace('/', '__')  # for filesystem paths
        self.taxonomy_path = taxonomy_path
        self.output_dir = Path(output_dir)
        self.enable_logging = enable_logging
        self.enable_concurrent = enable_concurrent
        
        # Setup logging first
        self.logger = self._setup_logger()
        
        # Initialize LLM for relevance judgment
        self.llm = create_llm_interface(
            model=model_name,
            system_instruction=self._get_system_instruction()
        )
        
        # Initialize concurrent processor if enabled
        if enable_concurrent:
            self.concurrent_processor = create_concurrent_processor(
                model_name=model_name,
                enable_logging=True
            )
            self.logger.info(f"Concurrent processing enabled for {model_name}")
        
        # Load taxonomy
        self.taxonomy = self._load_taxonomy()
    
    def _get_system_instruction(self) -> str:
        """Get system instruction for the LLM."""
        return """You are an expert code analysis assistant. Your task is to determine whether a coding instruction is relevant to a given user query and doesn't conflict with existing instructions.

If the instruction is relevant and non-conflicting, output the instruction in the required JSON format with appropriate parameters. If not, output a rejection JSON.

Always wrap your JSON response in ```json code blocks for easy extraction.

Be precise and analytical in your judgments. Consider the specific requirements, coding style, and technical aspects mentioned in both the query and instructions."""
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the plugger."""
        logger = logging.getLogger('InstructionPlugger')
        logger.setLevel(logging.INFO if self.enable_logging else logging.WARNING)
        
        # Remove existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _substitute_parameters(self, text: str, parameters: Dict[str, Any]) -> str:
        """
        Substitute parameter placeholders in text with actual values.
        
        Args:
            text: Text containing placeholders like {line_length}
            parameters: Dictionary of parameter values
            
        Returns:
            Text with placeholders replaced
        """
        result = text
        for param_name, param_value in parameters.items():
            placeholder = "{" + param_name + "}"
            result = result.replace(placeholder, str(param_value))
        return result
    
    def _validate_prompt_consistency(
        self, 
        original_prompt: str, 
        returned_prompt: str, 
        parameters: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate that the returned prompt matches the original (ignoring parameter substitutions).
        
        Args:
            original_prompt: The original prompt from CSV
            returned_prompt: The prompt returned by LLM
            parameters: Parameters that should be substituted
            
        Returns:
            Tuple of (is_consistent, error_message)
        """
        # Create expected prompt by substituting parameters in original
        expected_prompt = self._substitute_parameters(original_prompt, parameters)
        
        # Check if returned prompt matches expected
        if returned_prompt.strip() == expected_prompt.strip():
            return True, ""
        
        # If not exact match, check if it's the original without substitution
        if returned_prompt.strip() == original_prompt.strip():
            return True, "needs_substitution"
        
        # Check if it's structurally similar (allowing for minor formatting differences)
        # Remove extra whitespace and compare
        expected_normalized = re.sub(r'\s+', ' ', expected_prompt.strip())
        returned_normalized = re.sub(r'\s+', ' ', returned_prompt.strip())
        
        if expected_normalized == returned_normalized:
            return True, "minor_formatting_difference"
        
        return False, f"Prompt mismatch. Expected: '{expected_prompt}', Got: '{returned_prompt}'"
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON object from ```json code blocks in LLM response.
        Handle cases where JSON content itself contains ```json strings.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            JSON string if found, None otherwise
        """
        # Look for ```json blocks
        json_start = response.find('```json')
        if json_start == -1:
            # Try without language specification
            json_start = response.find('```')
            if json_start == -1:
                return None
            start_pos = json_start + 3
        else:
            start_pos = json_start + 7  # len('```json')
        
        # Skip any whitespace/newlines after ```json
        while start_pos < len(response) and response[start_pos] in ' \t\n\r':
            start_pos += 1
        
        # Find the closing ``` by looking for a ``` that's at the start of a line or after whitespace
        current_pos = start_pos
        while current_pos < len(response):
            next_triple = response.find('```', current_pos)
            if next_triple == -1:
                return None
            
            # Check if this ``` is at the start of a line or after whitespace (not inside quoted content)
            if next_triple == 0 or response[next_triple - 1] in ' \t\n\r':
                # This looks like a real closing marker
                json_str = response[start_pos:next_triple].strip()
                
                # Validate it's actually valid JSON
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    # This wasn't the right closing marker, continue searching
                    current_pos = next_triple + 3
                    continue
            else:
                # This ``` is inside quoted content, skip it
                current_pos = next_triple + 3
                continue
        
        return None
    
    def _load_taxonomy(self) -> pd.DataFrame:
        """Load the instruction taxonomy from CSV."""
        try:
            taxonomy = pd.read_csv(self.taxonomy_path)
            self.logger.info(f"Loaded taxonomy with {len(taxonomy)} instructions")
            return taxonomy
        except Exception as e:
            raise ValueError(f"Failed to load taxonomy from {self.taxonomy_path}: {e}")
    
    def _get_instance_id(self, instance: Dict[str, Any], benchmark_name: str) -> str:
        """Get the appropriate ID field for the instance based on benchmark."""
        if benchmark_name == 'livecodebench':
            return instance.get('question_id', 'unknown')
        else:
            return instance.get('task_id', 'unknown')
    
    def _get_user_query(self, instance: Dict[str, Any], benchmark_name: str) -> str:
        """Get the appropriate user query field based on benchmark."""
        if benchmark_name == 'livecodebench':
            return instance.get('question_content', '')
        else:
            return instance.get('instruct_prompt', instance.get('prompt', ''))
    
    def _load_benchmark_dataset(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """Load benchmark dataset."""
        if benchmark_name not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}. Supported: {list(self.SUPPORTED_BENCHMARKS.keys())}")
        
        dataset_info = self.SUPPORTED_BENCHMARKS[benchmark_name]
        dataset_name, version_or_split = dataset_info
        
        try:
            if benchmark_name == 'livecodebench':
                # LiveCodeBench uses version_tag parameter
                dataset = load_dataset(dataset_name, version_tag=version_or_split)
                # LiveCodeBench has a 'test' split
                instances = []
                for item in dataset['test']:
                    instances.append(dict(item))
            else:
                # BigCodeBench uses split parameter
                dataset = load_dataset(dataset_name, split=version_or_split)
                instances = []
                for item in dataset:
                    instances.append(dict(item))
            
            self.logger.info(f"Loaded {len(instances)} instances from {benchmark_name}")
            return instances
        
        except Exception as e:
            raise ValueError(f"Failed to load benchmark {benchmark_name}: {e}")
    
    def _create_instruction_evaluation_prompt(
        self,
        user_query: str,
        instruction: Dict[str, Any],
        existing_instructions: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for LLM to evaluate and potentially format instruction."""
        
        existing_summary = ""
        if existing_instructions:
            existing_list = []
            for idx, inst in enumerate(existing_instructions, 1):
                existing_list.append(f"{idx}.\nGeneration Prompt: {inst['generation_prompt']}\n")
                existing_list.append(f"Edit Prompt: {inst['edit_prompt']}\n")
                existing_list.append(f"Parameters: {inst['parameters']}\n")
            existing_summary = f"EXISTING INSTRUCTIONS ALREADY SELECTED:\n" + "\n".join(existing_list) + "\n\n"
        
        prompt = f"""You are evaluating coding instructions for a real-world user query. We are trying to simulate more complex coding scenarios by adding relevant constraints that would require code or response modifications.

{existing_summary}USER QUERY:
{user_query}

PROPOSED INSTRUCTION:
Category:
{instruction['Category']}

Description:
{instruction['Description']}

Generation Prompt (This is used when generating new code from scratch):
{instruction['Generation_Prompt']}

Edit Prompt (This is used when modifying existing code):
{instruction['Edit_Prompt']}

Parameters:
{instruction['Parameters']}

Notes:
{instruction['Notes']}

Please evaluate:
1. RELEVANCE: Would this proposed instruction add meaningful constraints that require code or response modifications for this specific user query? Consider if it addresses coding practices, style requirements, documentation standards, explanation formats, or technical constraints relevant to the query.

2. CONFLICT: Does this instruction contradict or conflict with any existing instructions? Look for contradictory requirements, incompatible coding styles, or mutually exclusive constraints.

If BOTH conditions are met (relevant AND no conflict), respond with:
```json
{{
    "accepted": true,
    "instruction_idx": "{instruction['ID']}",
    "generation_prompt": "{instruction['Generation_Prompt']}",
    "edit_prompt": "{instruction['Edit_Prompt']}",
    "parameters": {{actual_parameters_based_on_notes}}
}}
```

If the instruction is NOT relevant OR has conflict, respond with:
```json
{{
    "accepted": false,
    "reason": "brief explanation"
}}
```

IMPORTANT: You MUST copy the generation_prompt and edit_prompt EXACTLY as shown above. Do not modify, rephrase, or change them in any way. Only the parameters object should contain your assigned values. Always wrap your JSON response in ```json code blocks.

For parameter selection:
- If this is a docstring convention instruction, randomly choose one of the three available values
- For all other parameters, use the default value mentioned in Notes first
- If the default is not possible, choose from recommended value ranges in Notes
- If Parameters is "None", use empty object: {{"parameters": {{}}}}

If the instruction is NOT relevant OR has conflict, respond with:
{{
    "accepted": false,
    "reason": "brief explanation"
}}

Be conservative - only accept if the instruction would meaningfully impact the code generation or response format for this specific query."""
        
        return prompt
    
    def _evaluate_instruction(
        self,
        user_query: str,
        instruction: Dict[str, Any],
        existing_instructions: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Use LLM to evaluate instruction and return formatted instruction if accepted.
        
        Returns:
            Tuple of (instruction_dict_if_accepted_or_None, reason)
        """
        prompt = self._create_instruction_evaluation_prompt(user_query, instruction, existing_instructions)
        
        try:
            response = self.llm.generate_response([
                {'role': 'user', 'parts': [{'text': prompt}]}
            ])
            
            # Extract JSON from ```json code blocks
            json_str = self._extract_json_from_response(response)
            if not json_str:
                raise ValueError("No valid JSON code block found in response")
            
            result = json.loads(json_str)
            
            if result.get('accepted', False):
                # Extract returned prompts and parameters
                returned_gen_prompt = result['generation_prompt']
                returned_edit_prompt = result['edit_prompt']
                parameters = result.get('parameters', {})
                
                # Validate and fix generation prompt
                gen_consistent, gen_error = self._validate_prompt_consistency(
                    instruction['Generation_Prompt'], returned_gen_prompt, parameters
                )
                
                # Validate and fix edit prompt  
                edit_consistent, edit_error = self._validate_prompt_consistency(
                    instruction['Edit_Prompt'], returned_edit_prompt, parameters
                )
                
                # Log validation results
                if not gen_consistent:
                    self.logger.warning(f"Generation prompt inconsistency for {instruction['ID']}: {gen_error}")
                if not edit_consistent:
                    self.logger.warning(f"Edit prompt inconsistency for {instruction['ID']}: {edit_error}")
                
                # Use original prompts with parameter substitution (more reliable)
                final_gen_prompt = self._substitute_parameters(instruction['Generation_Prompt'], parameters)
                final_edit_prompt = self._substitute_parameters(instruction['Edit_Prompt'], parameters)
                
                # Create final instruction
                formatted_inst = {
                    "instruction_idx": result['instruction_idx'],
                    "generation_prompt": final_gen_prompt,
                    "edit_prompt": final_edit_prompt,
                    "parameters": parameters
                }
                
                # Log validation status
                validation_status = []
                if gen_error == "needs_substitution" or edit_error == "needs_substitution":
                    validation_status.append("parameter_substitution_applied")
                if gen_error == "minor_formatting_difference" or edit_error == "minor_formatting_difference":
                    validation_status.append("minor_formatting_fixed")
                if not gen_consistent or not edit_consistent:
                    validation_status.append("prompt_corrected")
                
                validation_msg = f"accepted ({', '.join(validation_status)})" if validation_status else "accepted"
                return formatted_inst, validation_msg
            else:
                # Log rejection reason
                reason = result.get('reason', 'No reason provided')
                self.logger.info(f"Instruction {instruction['ID']} rejected: {reason}")
                return None, reason
            
        except Exception as e:
            # Log the raw response for debugging
            self.logger.error(f"Failed to parse LLM response for instruction {instruction['ID']}: {e}")
            self.logger.debug(f"Raw response: {response[:500]}...")
            return None, f"Parsing error: {str(e)}"
    
    def _process_plugging_instance(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single instance for instruction plugging. Used for concurrent processing.
        
        Args:
            instance_data: Dictionary containing:
                - instance: The benchmark instance
                - n_instructions: Target number of instructions
                - benchmark_name: Name of the benchmark
                - taxonomy_shuffled: Shuffled taxonomy DataFrame
                - existing_instructions: List of already selected instructions (for resume)
                
        Returns:
            Updated instance with instruction_list
        """
        instance = instance_data['instance']
        n_instructions = instance_data['n_instructions']
        benchmark_name = instance_data['benchmark_name']
        taxonomy_shuffled = instance_data['taxonomy_shuffled']
        existing_instructions = instance_data.get('existing_instructions', [])
        
        instance_id = self._get_instance_id(instance, benchmark_name)
        user_query = self._get_user_query(instance, benchmark_name)
        
        if not user_query:
            self.logger.warning(f"No user query found for instance {instance_id}")
            instance_updated = instance.copy()
            instance_updated['instruction_list'] = existing_instructions
            return instance_updated
        
        # Initialize instruction list with existing instructions
        selected_instructions = existing_instructions.copy()
        
        # Try each instruction in the provided shuffled order
        for _, instruction in taxonomy_shuffled.iterrows():
            if len(selected_instructions) >= n_instructions:
                break
            
            try:
                formatted_instruction, reason = self._evaluate_instruction(
                    user_query, instruction, selected_instructions
                )
                
                if formatted_instruction:
                    selected_instructions.append(formatted_instruction)
                    self.logger.info(f"Instance {instance_id}: Instruction {instruction['ID']} accepted - {reason}")
                else:
                    self.logger.info(f"Instance {instance_id}: Instruction {instruction['ID']} rejected - {reason}")
                    
            except Exception as e:
                self.logger.error(f"Error processing instruction {instruction['ID']} for instance {instance_id}: {e}")
        
        # Update instance
        instance_updated = instance.copy()
        instance_updated['instruction_list'] = selected_instructions
        
        self.logger.info(f"Instance {instance_id}: Selected {len(selected_instructions)}/{n_instructions} instructions")
        
        return instance_updated
    
    def _plug_instructions_for_instance(
        self,
        instance: Dict[str, Any],
        n_instructions: int,
        benchmark_name: str,
        log_file: Optional[object] = None,
        existing_instructions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Plug instructions for a single instance (serial version, kept for compatibility).
        
        Args:
            instance: The benchmark instance
            n_instructions: Number of instructions to plug
            benchmark_name: Name of the benchmark
            log_file: Optional file object for logging
            existing_instructions: Existing instructions to continue from
            
        Returns:
            Updated instance with instruction_list
        """
        # Get user query based on benchmark
        user_query = self._get_user_query(instance, benchmark_name)
        instance_id = self._get_instance_id(instance, benchmark_name)
        
        if not user_query:
            self.logger.warning(f"No user query found for instance {instance_id}")
            return instance
        
        # Initialize instruction list with existing instructions
        selected_instructions = existing_instructions or []
        initial_count = len(selected_instructions)
        
        # Get IDs of already selected instructions to avoid duplicates
        existing_instruction_ids = set()
        if selected_instructions:
            for inst in selected_instructions:
                existing_instruction_ids.add(inst.get('instruction_idx', ''))
        
        # Create shuffled copy of taxonomy, excluding already selected instructions
        available_taxonomy = self.taxonomy[~self.taxonomy['ID'].isin(existing_instruction_ids)]
        
        if len(available_taxonomy) == 0:
            self.logger.warning(f"No available instructions left for instance {instance_id}")
            instance_updated = instance.copy()
            instance_updated['instruction_list'] = selected_instructions
            return instance_updated
        
        taxonomy_shuffled = available_taxonomy.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        
        # Log filtering info
        if existing_instruction_ids:
            self.logger.info(f"Filtered out {len(existing_instruction_ids)} already selected instructions, {len(taxonomy_shuffled)} candidates remaining")
        
        # Log instance start
        if log_file:
            if initial_count > 0:
                log_file.write(f"Continuing instance: {instance_id}\n")
                log_file.write(f"User query: {user_query[:200]}...\n")
                log_file.write(f"Existing instructions: {initial_count}\n")
                log_file.write(f"Target instructions: {n_instructions}\n\n")
            else:
                log_file.write(f"Processing instance: {instance_id}\n")
                log_file.write(f"User query: {user_query[:200]}...\n")
                log_file.write(f"Target instructions: {n_instructions}\n\n")
        
        # Try each instruction in random order
        for _, instruction in taxonomy_shuffled.iterrows():
            if len(selected_instructions) >= n_instructions:
                break
            
            try:
                formatted_instruction, reason = self._evaluate_instruction(
                    user_query, instruction, selected_instructions
                )
                
                if formatted_instruction:
                    selected_instructions.append(formatted_instruction)
                    decision = "ADDED"
                    log_msg = f"Instruction {instruction['ID']} ({instruction['Category']}): {decision}"
                    log_reason = f"  Status: {reason}"
                    
                    self.logger.info(f"{log_msg} - {reason}")
                    if log_file:
                        log_file.write(f"{log_msg}\n")
                        log_file.write(f"{log_reason}\n")
                        log_file.write(f"  Generated instruction: {json.dumps(formatted_instruction, indent=2)}\n\n")
                else:
                    decision = "SKIPPED"
                    log_msg = f"Instruction {instruction['ID']} ({instruction['Category']}): {decision}"
                    log_reason = f"  Reason: {reason}"
                    
                    self.logger.info(f"{log_msg} - {reason}")
                    if log_file:
                        log_file.write(f"{log_msg}\n")
                        log_file.write(f"{log_reason}\n\n")
                
            except Exception as e:
                error_msg = f"Error processing instruction {instruction['ID']}: {e}"
                self.logger.error(error_msg)
                if log_file:
                    log_file.write(f"ERROR: {error_msg}\n\n")
        
        # Update instance
        instance_updated = instance.copy()
        instance_updated['instruction_list'] = selected_instructions
        
        # Final log
        added_count = len(selected_instructions) - initial_count
        final_msg = f"Final: Selected {len(selected_instructions)}/{n_instructions} instructions (added {added_count} new)"
        self.logger.info(final_msg)
        if log_file:
            log_file.write(f"{final_msg}\n")
            log_file.write("-" * 50 + "\n\n")
        
        return instance_updated
    
    def _get_processed_instances(self, output_file: Path, target_instructions: int, benchmark_name: str) -> Tuple[Dict[str, Any], set]:
        """
        Get processed instances and their instruction counts from existing output file.
        
        Args:
            output_file: Path to the output JSONL file
            target_instructions: Target number of instructions
            benchmark_name: Name of the benchmark
            
        Returns:
            Tuple of (instances_with_partial_instructions_dict, fully_processed_ids_set)
        """
        processed_instances = {}  # id -> instance_data
        fully_processed_ids = set()
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            instance = json.loads(line.strip())
                            instance_id = self._get_instance_id(instance, benchmark_name)
                            if instance_id and instance_id != 'unknown':
                                instruction_count = len(instance.get('instruction_list', []))
                                processed_instances[instance_id] = instance
                                
                                if instruction_count >= target_instructions:
                                    fully_processed_ids.add(instance_id)
                                
                self.logger.info(f"Found {len(processed_instances)} existing instances in {output_file}")
                self.logger.info(f"  - {len(fully_processed_ids)} fully processed (>= {target_instructions} instructions)")
                self.logger.info(f"  - {len(processed_instances) - len(fully_processed_ids)} need more instructions")
                
            except Exception as e:
                self.logger.warning(f"Error reading existing output file {output_file}: {e}")
                
        return processed_instances, fully_processed_ids
    
    def plug_instructions(
        self,
        benchmark_name: str,
        n_instructions: int
    ) -> str:
        """
        Main function to plug instructions into benchmark instances.
        
        Args:
            benchmark_name: Name of the benchmark dataset
            n_instructions: Number of instructions to plug per instance
            
        Returns:
            Path to the output JSONL file
        """
        # Load dataset
        instances = self._load_benchmark_dataset(benchmark_name)
        
        # Setup output directory
        output_dir = self.output_dir / benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files
        output_file = output_dir / f"{self.sanitized_model_name}_{n_instructions}-instructions.jsonl"
        log_file_path = output_dir / f"{self.sanitized_model_name}_{n_instructions}-instructions.txt"
        
        # Check for already processed instances
        processed_instances, fully_processed_ids = self._get_processed_instances(output_file, n_instructions, benchmark_name)
        
        # Categorize instances
        new_instances = []        # Completely new instances
        partial_instances = []    # Instances with some but not enough instructions
        
        for instance in instances:
            instance_id = self._get_instance_id(instance, benchmark_name)
            if instance_id in fully_processed_ids:
                continue  # Skip fully processed instances
            elif instance_id in processed_instances:
                # This instance exists but needs more instructions
                partial_instances.append((instance, processed_instances[instance_id]))
            else:
                # Brand new instance
                new_instances.append(instance)
        
        total_to_process = len(new_instances) + len(partial_instances)
        
        self.logger.info(f"Processing status:")
        self.logger.info(f"  - Total instances: {len(instances)}")
        self.logger.info(f"  - Fully processed: {len(fully_processed_ids)}")
        self.logger.info(f"  - Need completion: {len(partial_instances)}")
        self.logger.info(f"  - Brand new: {len(new_instances)}")
        self.logger.info(f"  - Total to process: {total_to_process}")
        
        if total_to_process == 0:
            self.logger.info("All instances have already been fully processed!")
            return str(output_file)
        
        if self.enable_concurrent:
            return self._plug_instructions_concurrent(
                instances, new_instances, partial_instances, processed_instances,
                fully_processed_ids, n_instructions, benchmark_name, output_file, log_file_path
            )
        else:
            return self._plug_instructions_serial(
                instances, new_instances, partial_instances, processed_instances,
                fully_processed_ids, n_instructions, benchmark_name, output_file, log_file_path
            )
    
    def _plug_instructions_concurrent(
        self,
        all_instances: List[Dict[str, Any]],
        new_instances: List[Dict[str, Any]],
        partial_instances: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        processed_instances: Dict[str, Any],
        fully_processed_ids: set,
        n_instructions: int,
        benchmark_name: str,
        output_file: Path,
        log_file_path: Path
    ) -> str:
        """Concurrent version of instruction plugging."""
        self.logger.info("Using concurrent processing for instruction plugging")
        
        # Prepare items for concurrent processing
        items_to_process = []
        
        # Add partial instances (need to complete)
        for original_instance, existing_instance in partial_instances:
            existing_instructions = existing_instance.get('instruction_list', [])
            existing_instruction_ids = {inst.get('instruction_idx', '') for inst in existing_instructions}
            
            # Create shuffled taxonomy excluding already selected instructions
            available_taxonomy = self.taxonomy[~self.taxonomy['ID'].isin(existing_instruction_ids)]
            taxonomy_shuffled = available_taxonomy.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
            
            items_to_process.append({
                'instance': original_instance,
                'n_instructions': n_instructions,
                'benchmark_name': benchmark_name,
                'taxonomy_shuffled': taxonomy_shuffled,
                'existing_instructions': existing_instructions
            })
        
        # Add new instances
        for instance in new_instances:
            # Create shuffled taxonomy for this instance
            taxonomy_shuffled = self.taxonomy.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
            
            items_to_process.append({
                'instance': instance,
                'n_instructions': n_instructions,
                'benchmark_name': benchmark_name,
                'taxonomy_shuffled': taxonomy_shuffled,
                'existing_instructions': []
            })
        
        # First, write all fully processed instances to the file
        with open(output_file, 'w', encoding='utf-8') as f:
            for instance in all_instances:
                instance_id = self._get_instance_id(instance, benchmark_name)
                if instance_id in fully_processed_ids:
                    f.write(json.dumps(processed_instances[instance_id], ensure_ascii=False) + '\n')
        
        # Process all items concurrently and append results as they complete
        successful_count = self.concurrent_processor.process_and_append_jsonl(
            items=items_to_process,
            process_func=self._process_plugging_instance,
            output_file=output_file,
            get_item_id=lambda x: self._get_instance_id(x['instance'], x['benchmark_name']),
            progress_desc=f"Plugging {n_instructions} instructions"
        )
        
        self.logger.info(f"Concurrent processing completed! Processed {successful_count}/{len(items_to_process)} instances successfully")
        self.logger.info(f"Output saved to: {output_file}")
        
        return str(output_file)
    
    def _plug_instructions_serial(
        self,
        all_instances: List[Dict[str, Any]],
        new_instances: List[Dict[str, Any]],
        partial_instances: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        processed_instances: Dict[str, Any],
        fully_processed_ids: set,
        n_instructions: int,
        benchmark_name: str,
        output_file: Path,
        log_file_path: Path
    ) -> str:
        """Serial version of instruction plugging (original implementation)."""
        # Setup log file
        log_file_handle = None
        if self.enable_logging:
            log_file_handle = open(log_file_path, 'a' if processed_instances else 'w', encoding='utf-8')
            
            if not processed_instances:
                # Write header only if starting fresh
                log_file_handle.write(f"Instruction Plugging Log\n")
                log_file_handle.write(f"Benchmark: {benchmark_name}\n")
                log_file_handle.write(f"Model: {self.model_name}\n")
                log_file_handle.write(f"Target instructions per instance: {n_instructions}\n")
                log_file_handle.write(f"Total instances: {len(all_instances)}\n")
                log_file_handle.write("=" * 50 + "\n\n")
            else:
                log_file_handle.write(f"\n{'='*50}\n")
                log_file_handle.write(f"RESUMING/COMPLETING PROCESSING\n")
                log_file_handle.write(f"New instances: {len(new_instances)}\n")
                log_file_handle.write(f"Partial instances to complete: {len(partial_instances)}\n")
                log_file_handle.write(f"{'='*50}\n\n")
            
            log_file_handle.flush()
        
        # Process partial instances (continue from existing instructions)
        with open(output_file, 'a', encoding='utf-8') as f:
            for i, (original_instance, existing_instance) in enumerate(partial_instances):
                instance_id = self._get_instance_id(original_instance, benchmark_name)
                existing_instructions = existing_instance.get('instruction_list', [])
                
                self.logger.info(f"Completing instance {instance_id} ({len(existing_instructions)}/{n_instructions} instructions)")
                
                try:
                    updated_instance = self._plug_instructions_for_instance(
                        original_instance, n_instructions, benchmark_name, log_file_handle, existing_instructions
                    )
                    f.write(json.dumps(updated_instance, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    if log_file_handle:
                        log_file_handle.flush()
                    
                    self.logger.info(f"Successfully completed instance {instance_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to complete instance {instance_id}: {e}"
                    self.logger.error(error_msg)
                    if log_file_handle:
                        log_file_handle.write(f"ERROR: {error_msg}\n")
                        log_file_handle.write("-" * 50 + "\n\n")
                        log_file_handle.flush()
                    
                    # Keep existing instance even if completion failed
                    f.write(json.dumps(existing_instance, ensure_ascii=False) + '\n')
                    f.flush()
            
            # Process new instances
            for i, instance in enumerate(new_instances):
                instance_id = self._get_instance_id(instance, benchmark_name)
                self.logger.info(f"Processing new instance {instance_id}")
                
                try:
                    updated_instance = self._plug_instructions_for_instance(
                        instance, n_instructions, benchmark_name, log_file_handle
                    )
                    f.write(json.dumps(updated_instance, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    if log_file_handle:
                        log_file_handle.flush()
                    
                    self.logger.info(f"Successfully processed instance {instance_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to process instance {instance_id}: {e}"
                    self.logger.error(error_msg)
                    if log_file_handle:
                        log_file_handle.write(f"ERROR: {error_msg}\n")
                        log_file_handle.write("-" * 50 + "\n\n")
                        log_file_handle.flush()
                    
                    # Add instance without instructions
                    instance_copy = instance.copy()
                    instance_copy['instruction_list'] = []
                    f.write(json.dumps(instance_copy, ensure_ascii=False) + '\n')
                    f.flush()
        # Close the log file
        if log_file_handle:
            log_file_handle.close()
        
        self.logger.info(f"Processing completed! Processed {len(partial_instances) + len(new_instances)} instances")
        self.logger.info(f"Output saved to: {output_file}")
        if self.enable_logging:
            self.logger.info(f"Detailed log saved to: {log_file_path}")
        
        return str(output_file)


def main():
    """Example usage of InstructionPlugger."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plug coding instructions into benchmark instances")
    parser.add_argument("--benchmark", required=True, choices=['bigcodebench', 'livecodebench'], 
                        help="Benchmark dataset name")
    parser.add_argument("--model", required=True, help="LLM model name for relevance judgment")
    parser.add_argument("--n-instructions", type=int, required=True, help="Number of instructions to plug per instance")
    parser.add_argument("--taxonomy-path", default="data/code_gym_taxonomy.csv", help="Path to taxonomy CSV file")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed logging")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent processing")
    
    args = parser.parse_args()
    
    # Create plugger
    plugger = InstructionPlugger(
        model_name=args.model,
        taxonomy_path=args.taxonomy_path,
        output_dir=args.output_dir,
        enable_logging=not args.no_logging,
        enable_concurrent=args.concurrent
    )
    
    # Run plugging
    output_file = plugger.plug_instructions(
        benchmark_name=args.benchmark,
        n_instructions=args.n_instructions
    )
    
    print(f"Instruction plugging completed. Output saved to: {output_file}")


if __name__ == "__main__":
    main()