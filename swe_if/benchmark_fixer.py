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
Tool to detect and fix parameter mismatches in augmented benchmark files.
Compares instruction parameters against the taxonomy specification.
"""

import json
import pandas as pd
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


class BenchmarkParameterFixer:
    """Tool to detect and fix parameter issues in benchmark instruction lists."""
    
    def __init__(self, taxonomy_path: str):
        """
        Initialize the fixer with taxonomy.
        
        Args:
            taxonomy_path: Path to the taxonomy CSV file
        """
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy()
        self.parameter_specs = self._parse_parameter_specs()
    
    def _load_taxonomy(self) -> pd.DataFrame:
        """Load taxonomy from CSV file."""
        try:
            taxonomy = pd.read_csv(self.taxonomy_path)
            print(f"Loaded taxonomy with {len(taxonomy)} instructions")
            return taxonomy
        except Exception as e:
            raise ValueError(f"Failed to load taxonomy from {self.taxonomy_path}: {e}")
    
    def _parse_parameter_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse parameter specifications from taxonomy.
        
        Returns:
            Dictionary mapping instruction ID to parameter specifications
        """
        specs = {}
        
        for _, row in self.taxonomy.iterrows():
            instruction_id = row['ID']
            parameters_str = str(row['Parameters']).strip()
            notes_str = str(row['Notes']).strip() if 'Notes' in row else ""
            
            # Parse parameter specification
            if parameters_str.lower() in ['none', 'nan', '', 'null']:
                # No parameters expected
                specs[instruction_id] = {
                    'expected_params': set(),
                    'param_types': {},
                    'notes': notes_str,
                    'raw_parameters': parameters_str
                }
            else:
                # Try multiple parsing strategies
                expected_params = set()
                param_types = {}
                
                # Strategy 1: Look for patterns like "line_length (int)", "convention (str)"
                pattern1 = r'(\w+)\s*\(([^)]+)\)'
                matches1 = re.findall(pattern1, parameters_str)
                
                if matches1:
                    for param_name, param_type in matches1:
                        expected_params.add(param_name)
                        param_types[param_name] = param_type.strip()
                else:
                    # Strategy 2: Look for patterns like "line_length: int", "convention: str"  
                    pattern2 = r'(\w+)\s*:\s*(\w+)'
                    matches2 = re.findall(pattern2, parameters_str)
                    
                    if matches2:
                        for param_name, param_type in matches2:
                            expected_params.add(param_name)
                            param_types[param_name] = param_type.strip()
                    else:
                        # Strategy 3: Look for parameter names in curly braces like "{line_length}"
                        pattern3 = r'\{(\w+)\}'
                        matches3 = re.findall(pattern3, parameters_str)
                        
                        if matches3:
                            for param_name in matches3:
                                expected_params.add(param_name)
                                param_types[param_name] = 'inferred_from_braces'
                        else:
                            # Strategy 4: Look in Notes field for parameter information
                            notes_params = self._extract_params_from_notes(notes_str)
                            if notes_params:
                                for param_name, param_type in notes_params.items():
                                    expected_params.add(param_name)
                                    param_types[param_name] = param_type
                            else:
                                # Strategy 5: Simple word extraction (fallback)
                                simple_pattern = r'\b(\w+)\b'
                                potential_params = re.findall(simple_pattern, parameters_str)
                                # Filter out common words and very short words
                                stop_words = {'int', 'str', 'bool', 'float', 'or', 'and', 'the', 'a', 'an', 'is', 'are', 'with', 'for', 'to', 'of', 'in', 'type', 'parameter', 'param', 'value', 'default'}
                                
                                for param in potential_params:
                                    if (param.lower() not in stop_words and 
                                        len(param) > 2 and 
                                        not param.isdigit() and
                                        param.islower()):  # Likely parameter names are lowercase
                                        expected_params.add(param)
                                        param_types[param] = 'inferred_simple'
                
                specs[instruction_id] = {
                    'expected_params': expected_params,
                    'param_types': param_types,
                    'notes': notes_str,
                    'raw_parameters': parameters_str
                }
        
        return specs
    
    def _extract_params_from_notes(self, notes: str) -> Dict[str, str]:
        """
        Extract parameter information from Notes field.
        
        Args:
            notes: Notes string from taxonomy
            
        Returns:
            Dictionary mapping parameter names to types
        """
        params = {}
        
        # Look for patterns like "line_length parameter", "{line_length}", "line_length value"
        # Also look for type hints
        
        # Pattern 1: {parameter_name} with optional type info
        pattern1 = r'\{(\w+)\}'
        matches1 = re.findall(pattern1, notes)
        for match in matches1:
            params[match] = 'from_notes_braces'
        
        # Pattern 2: "parameter_name (type)" or "parameter_name parameter"
        pattern2 = r'(\w+)\s+(?:parameter|\(([^)]+)\))'
        matches2 = re.findall(pattern2, notes)
        for param_name, param_type in matches2:
            if len(param_name) > 2 and param_name.islower():
                params[param_name] = param_type.strip() if param_type else 'from_notes_param'
        
        # Pattern 3: Look for common parameter names
        common_params = [
            'line_length', 'max_length', 'convention', 'style', 'format', 
            'max_complexity', 'max_branches', 'max_statements', 'max_locals',
            'max_nested_blocks', 'max_bool_expr', 'quote_type', 'quote_style'
        ]
        
        for param in common_params:
            if param in notes.lower() and param not in params:
                params[param] = 'from_notes_common'
        
        return params
    
    def _validate_instruction_parameters(self, instruction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters for a single instruction.
        
        Args:
            instruction: Instruction dictionary with 'instruction_idx' and 'parameters'
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        instruction_id = instruction.get('instruction_idx', '')
        parameters = instruction.get('parameters', {})
        
        if instruction_id not in self.parameter_specs:
            return False, [f"Unknown instruction ID: {instruction_id}"]
        
        spec = self.parameter_specs[instruction_id]
        expected_params = spec['expected_params']
        actual_params = set(parameters.keys())
        
        issues = []
        
        # Check for unexpected parameters
        unexpected = actual_params - expected_params
        if unexpected:
            issues.append(f"Unexpected parameters: {sorted(unexpected)}")
        
        # Check for missing required parameters
        missing = expected_params - actual_params
        if missing:
            issues.append(f"Missing required parameters: {sorted(missing)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def detect_issues(self, benchmark_file: str) -> Dict[str, Any]:
        """
        Detect parameter issues in a benchmark file.
        
        Args:
            benchmark_file: Path to the benchmark JSONL file
            
        Returns:
            Dictionary with detected issues
        """
        if not Path(benchmark_file).exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
        
        print(f"Analyzing benchmark file: {benchmark_file}")
        
        issues_by_task = {}
        instruction_issue_counts = defaultdict(int)
        total_tasks = 0
        total_instructions = 0
        tasks_with_issues = 0
        instructions_with_issues = 0
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    task_id = data.get('task_id', f'line_{line_num}')
                    instruction_list = data.get('instruction_list', [])
                    
                    total_tasks += 1
                    total_instructions += len(instruction_list)
                    
                    task_issues = []
                    
                    for i, instruction in enumerate(instruction_list):
                        is_valid, issues = self._validate_instruction_parameters(instruction)
                        
                        if not is_valid:
                            instructions_with_issues += 1
                            instruction_id = instruction.get('instruction_idx', 'unknown')
                            instruction_issue_counts[instruction_id] += 1
                            
                            task_issues.append({
                                'instruction_index': i,
                                'instruction_id': instruction_id,
                                'issues': issues,
                                'current_parameters': instruction.get('parameters', {}),
                                'expected_parameters': list(self.parameter_specs.get(instruction_id, {}).get('expected_params', set()))
                            })
                    
                    if task_issues:
                        tasks_with_issues += 1
                        issues_by_task[task_id] = task_issues
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        return {
            'issues_by_task': issues_by_task,
            'instruction_issue_counts': dict(instruction_issue_counts),
            'summary': {
                'total_tasks': total_tasks,
                'total_instructions': total_instructions,
                'tasks_with_issues': tasks_with_issues,
                'instructions_with_issues': instructions_with_issues,
                'error_rate_tasks': tasks_with_issues / total_tasks if total_tasks > 0 else 0,
                'error_rate_instructions': instructions_with_issues / total_instructions if total_instructions > 0 else 0
            }
        }
    
    def fix_benchmark(self, benchmark_file: str, output_file: str = None) -> str:
        """
        Fix parameter issues in a benchmark file.
        
        Args:
            benchmark_file: Path to the input benchmark JSONL file
            output_file: Path to the output file (default: add '_fixed' suffix to input file)
            
        Returns:
            Path to the fixed file
        """
        if output_file is None:
            input_path = Path(benchmark_file)
            output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
        
        print(f"Fixing benchmark file: {benchmark_file}")
        print(f"Output will be saved to: {output_file}")
        print(f"Default naming: If no --output specified, adds '_fixed' suffix to input filename")
        print()
        
        fixed_tasks = 0
        fixed_instructions = 0
        
        with open(benchmark_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    instruction_list = data.get('instruction_list', [])
                    
                    task_had_issues = False
                    fixed_instruction_list = []
                    
                    for instruction in instruction_list:
                        instruction_id = instruction.get('instruction_idx', '')
                        current_params = instruction.get('parameters', {})
                        
                        # Get expected parameters
                        if instruction_id in self.parameter_specs:
                            expected_params = self.parameter_specs[instruction_id]['expected_params']
                            
                            # Create fixed parameters by keeping only expected ones
                            fixed_params = {}
                            for param_name in expected_params:
                                if param_name in current_params:
                                    fixed_params[param_name] = current_params[param_name]
                            
                            # Check if we made changes
                            if set(current_params.keys()) != set(fixed_params.keys()):
                                task_had_issues = True
                                fixed_instructions += 1
                                
                                removed_params = set(current_params.keys()) - set(fixed_params.keys())
                                if removed_params:
                                    print(f"Task {data.get('task_id', line_num)}, Instruction {instruction_id}: Removed parameters {sorted(removed_params)}")
                            
                            # Update instruction with fixed parameters
                            fixed_instruction = instruction.copy()
                            fixed_instruction['parameters'] = fixed_params
                            fixed_instruction_list.append(fixed_instruction)
                        else:
                            # Unknown instruction, keep as is but warn
                            print(f"Warning: Unknown instruction {instruction_id} in task {data.get('task_id', line_num)}")
                            fixed_instruction_list.append(instruction)
                    
                    if task_had_issues:
                        fixed_tasks += 1
                    
                    # Update data with fixed instruction list
                    data['instruction_list'] = fixed_instruction_list
                    
                    # Write fixed data
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    # Write original line in case of error
                    outfile.write(line + '\n')
                    continue
        
        print(f"Fixed {fixed_tasks} tasks with {fixed_instructions} instruction parameter issues")
        return output_file
    
    def print_parameter_specs(self):
        """Print parameter specifications for all instructions."""
        print("\nPARAMETER SPECIFICATIONS FROM TAXONOMY:")
        print("=" * 50)
        
        for instruction_id, spec in sorted(self.parameter_specs.items()):
            expected_params = spec['expected_params']
            param_types = spec['param_types']
            raw_parameters = spec.get('raw_parameters', 'unknown')
            
            print(f"\n{instruction_id}:")
            print(f"  Raw Parameters field: '{raw_parameters}'")
            
            if expected_params:
                print(f"  Expected parameters:")
                for param in sorted(expected_params):
                    param_type = param_types.get(param, 'unknown')
                    print(f"    - {param} ({param_type})")
            else:
                print(f"  No parameters expected")
            
            if spec['notes'] and len(spec['notes']) > 10:
                print(f"  Notes (first 100 chars): {spec['notes'][:100]}...")
    
    def print_issues_report(self, issues_data: Dict[str, Any]):
        """Print a detailed issues report."""
        issues_by_task = issues_data['issues_by_task']
        instruction_issue_counts = issues_data['instruction_issue_counts']
        summary = issues_data['summary']
        
        print("\n" + "=" * 80)
        print("BENCHMARK PARAMETER ISSUES REPORT")
        print("=" * 80)
        
        if issues_by_task:
            print(f"\nDETAILED ISSUES (all {len(issues_by_task)} tasks):")
            print("-" * 40)
            
            for task_id, task_issues in issues_by_task.items():
                print(f"\nTask: {task_id}")
                for issue in task_issues:
                    instruction_id = issue['instruction_id']
                    current_params = issue['current_parameters']
                    expected_params = issue['expected_parameters']
                    issues_list = issue['issues']
                    
                    print(f"  Instruction {instruction_id}:")
                    print(f"    Current parameters: {current_params}")
                    print(f"    Expected parameters: {expected_params}")
                    for issue_desc in issues_list:
                        print(f"    Issue: {issue_desc}")
        
        # Summary at the end
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Total tasks: {summary['total_tasks']}")
        print(f"  Total instructions: {summary['total_instructions']}")
        print(f"  Tasks with issues: {summary['tasks_with_issues']} ({summary['error_rate_tasks']:.2%})")
        print(f"  Instructions with issues: {summary['instructions_with_issues']} ({summary['error_rate_instructions']:.2%})")
        
        if instruction_issue_counts:
            print(f"\n  Issues by instruction type:")
            for instruction_id, count in sorted(instruction_issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {instruction_id}: {count} issues")
        
        if summary['tasks_with_issues'] == 0:
            print("\n✓ No parameter issues found!")
        else:
            print(f"\n⚠ Found issues in {summary['tasks_with_issues']} tasks")
            print(f"💡 Use --fix to automatically resolve parameter mismatches")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Detect and fix parameter issues in augmented benchmark files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show parameter specifications
  python -m swe_if.benchmark_fixer --show-specs
  
  # Debug parameter parsing (show raw fields)
  python -m swe_if.benchmark_fixer --debug-params
  
  # Detect issues only
  python -m swe_if.benchmark_fixer --detect data/bigcodebench/gemini-3.1-pro-preview_5-instructions.jsonl
  
  # Fix issues (saves to same directory with '_fixed' suffix)
  python -m swe_if.benchmark_fixer --fix data/bigcodebench/gemini-3.1-pro-preview_5-instructions.jsonl
  # Output: data/bigcodebench/gemini-3.1-pro-preview_5-instructions_fixed.jsonl
  
  # Fix with custom output path
  python -m swe_if.benchmark_fixer --fix data/bigcodebench/gemini-3.1-pro-preview_5-instructions.jsonl --output fixed_benchmark.jsonl
  
  # Both detect and fix
  python -m swe_if.benchmark_fixer --detect --fix data/bigcodebench/gemini-3.1-pro-preview_5-instructions.jsonl

File Output Locations:
  - Default fix output: {input_directory}/{input_name}_fixed.jsonl
  - Custom fix output: Use --output to specify exact path
        """
    )
    
    parser.add_argument("benchmark_file", nargs='?', help="Path to benchmark JSONL file")
    parser.add_argument("--taxonomy-path", default="data/code_gym_taxonomy.csv", help="Path to taxonomy CSV file")
    parser.add_argument("--detect", action="store_true", help="Detect parameter issues")
    parser.add_argument("--fix", action="store_true", help="Fix parameter issues")
    parser.add_argument("--output", help="Output file for fixed benchmark (default: add '_fixed' suffix)")
    parser.add_argument("--show-specs", action="store_true", help="Show parameter specifications from taxonomy")
    parser.add_argument("--debug-params", action="store_true", help="Show raw parameter fields for debugging")
    
    args = parser.parse_args()
    
    # Initialize fixer
    try:
        fixer = BenchmarkParameterFixer(args.taxonomy_path)
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return 1
    
    # Show specifications if requested
    if args.show_specs:
        fixer.print_parameter_specs()
        if not args.benchmark_file:
            return 0
    
    # Show debug parameter info if requested  
    if args.debug_params:
        print("\nRAW PARAMETER FIELDS FROM TAXONOMY:")
        print("=" * 40)
        for _, row in fixer.taxonomy.iterrows():
            print(f"{row['ID']}: '{row['Parameters']}'")
        if not args.benchmark_file:
            return 0
    
    # Check if benchmark file is provided when needed
    if (args.detect or args.fix) and not args.benchmark_file:
        print("Error: benchmark_file is required when using --detect or --fix")
        return 1
    
    # Detect issues
    if args.detect:
        try:
            issues_data = fixer.detect_issues(args.benchmark_file)
            fixer.print_issues_report(issues_data)
        except Exception as e:
            print(f"Error detecting issues: {e}")
            return 1
    
    # Fix issues
    if args.fix:
        try:
            fixed_file = fixer.fix_benchmark(args.benchmark_file, args.output)
            print(f"\nFixed benchmark saved to: {fixed_file}")
            
            # Detect issues in fixed file to verify
            print("\nVerifying fixed file...")
            issues_data = fixer.detect_issues(fixed_file)
            if issues_data['summary']['instructions_with_issues'] == 0:
                print("✓ All parameter issues have been fixed!")
            else:
                print(f"⚠ Still {issues_data['summary']['instructions_with_issues']} instructions with issues")
        except Exception as e:
            print(f"Error fixing benchmark: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())