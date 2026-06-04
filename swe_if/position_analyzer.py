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
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from prettytable import PrettyTable

# Import IF evaluation components from evaluator
from .evaluator import IFEvaluator


class PositionAnalyzer:
    """Analyzer for instruction position analysis."""
    
    def __init__(self, taxonomy_path: str = "data/code_gym_taxonomy.csv"):
        """
        Initialize position analyzer.
        
        Args:
            taxonomy_path: Path to the taxonomy CSV file
        """
        self.if_evaluator = IFEvaluator(taxonomy_path)
    
    def analyze_code_positions(self, 
                             code_file: str, 
                             instruction_definition_file: str) -> Dict[str, Any]:
        """
        Analyze how code performs on each instruction position.
        
        Args:
            code_file: Path to the code file to analyze (e.g., 2-instructions.jsonl)
            instruction_definition_file: Path to file with complete instruction definitions
            
        Returns:
            Dictionary with position analysis results
        """
        if not os.path.exists(code_file):
            raise FileNotFoundError(f"Code file not found: {code_file}")
        
        if not os.path.exists(instruction_definition_file):
            raise FileNotFoundError(f"Instruction definition file not found: {instruction_definition_file}")
        
        # Load complete instruction definitions from the definition file
        instruction_definitions = {}
        max_positions = 0
        
        print("Loading instruction definitions...")
        with open(instruction_definition_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading instructions"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    task_id = data.get('task_id')
                    if_info = data.get('if_info', {})
                    constraints = if_info.get('constraints', [])
                    
                    if task_id and constraints:
                        # Add category info to each constraint
                        enriched_constraints = []
                        for constraint in constraints:
                            constraint_id = constraint.get('index')
                            category = 'unknown'
                            if not self.if_evaluator.taxonomy.empty:
                                taxonomy_row = self.if_evaluator.taxonomy[self.if_evaluator.taxonomy['ID'] == constraint_id]
                                if not taxonomy_row.empty:
                                    category = taxonomy_row.iloc[0]['Category']
                            
                            enriched_constraints.append({
                                'constraint': constraint,
                                'constraint_id': constraint_id,
                                'category': category
                            })
                        
                        instruction_definitions[task_id] = enriched_constraints
                        max_positions = max(max_positions, len(constraints))
                
                except Exception as e:
                    continue
        
        if not instruction_definitions:
            raise ValueError("No instruction definitions found in the definition file")
        
        print(f"Loaded {len(instruction_definitions)} instruction definitions with max {max_positions} positions")
        
        # Analyze each position
        position_results = {}
        
        for position in range(1, max_positions + 1):
            print(f"\nAnalyzing position {position}...")
            
            # Collect results for this position
            position_scores = []
            category_scores = defaultdict(list)
            constraint_scores = defaultdict(list)
            
            # Process each instance in the code file
            with open(code_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Evaluating position {position}"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        task_id = data.get('task_id')
                        solution_code = data.get('solution', '')
                        full_response = data.get('full_response', '')
                        
                        # Check if we have instruction definition for this task_id and position
                        if (task_id in instruction_definitions and 
                            len(instruction_definitions[task_id]) >= position):
                            
                            # Get the constraint at this position from the complete definition
                            constraint_info = instruction_definitions[task_id][position - 1]
                            constraint = constraint_info['constraint']
                            constraint_id = constraint_info['constraint_id']
                            category = constraint_info['category']
                            
                            # Evaluate this single instance with its constraint at this position
                            result = self.if_evaluator.evaluate_instance(
                                solution_code, full_response, [constraint]
                            )
                            
                            score = result.get('constraint_level_score', 0.0)
                            
                            # Collect scores
                            position_scores.append(score)
                            category_scores[category].append(score)
                            constraint_scores[constraint_id].append(score)
                        
                    except Exception as e:
                        continue
            
            # Calculate averages for this position
            if position_scores:
                overall_score = sum(position_scores) / len(position_scores)
                
                # Calculate category averages
                category_averages = {}
                for cat, scores in category_scores.items():
                    category_averages[cat] = sum(scores) / len(scores)
                
                # Calculate constraint type averages
                constraint_averages = {}
                for const_id, scores in constraint_scores.items():
                    constraint_averages[const_id] = sum(scores) / len(scores)
                
                position_results[position] = {
                    'overall_score': overall_score,
                    'category_scores': category_averages,
                    'constraint_scores': constraint_averages,
                    'n_instances': len(position_scores)
                }
                
                print(f"Position {position}: Overall Score = {overall_score:.4f} ({len(position_scores)} instances)")
                
            else:
                position_results[position] = {
                    'overall_score': 0.0,
                    'category_scores': {},
                    'constraint_scores': {},
                    'n_instances': 0
                }
                
                print(f"Position {position}: No data available")
        
        return {
            'code_file': code_file,
            'instruction_definition_file': instruction_definition_file,
            'n_positions': max_positions,
            'position_results': position_results
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis results saved to: {output_path}")
    
    def generate_summary_text(self, results: Dict[str, Any], output_path: str):
        """Generate a text summary of the position analysis results."""
        text_content = []
        
        # Header
        text_content.append("=" * 80)
        text_content.append("POSITION ANALYSIS SUMMARY")
        text_content.append("=" * 80)
        text_content.append("")
        text_content.append(f"Code File: {results.get('code_file', 'unknown')}")
        text_content.append(f"Instruction Definition File: {results.get('instruction_definition_file', 'unknown')}")
        text_content.append(f"Number of Positions: {results.get('n_positions', 0)}")
        text_content.append("")
        
        position_results = results.get('position_results', {})
        n_positions = results.get('n_positions', 0)
        
        # Overall Position Analysis
        text_content.append("OVERALL POSITION ANALYSIS")
        text_content.append("=" * 40)
        text_content.append("")
        
        table = PrettyTable()
        table.field_names = ["Position", "Overall Score", "Instances"]
        
        for position in range(1, n_positions + 1):
            if position in position_results:
                pos_data = position_results[position]
                overall_score = pos_data.get('overall_score', 0.0)
                n_instances = pos_data.get('n_instances', 0)
                table.add_row([f"Position {position}", f"{overall_score:.4f}", str(n_instances)])
            else:
                table.add_row([f"Position {position}", "0.0000", "0"])
        
        text_content.append(str(table))
        text_content.append("")
        text_content.append("")
        
        # Category Analysis
        text_content.append("CATEGORY ANALYSIS")
        text_content.append("=" * 40)
        text_content.append("")
        
        # Collect all categories across all positions
        all_categories = set()
        for position_data in position_results.values():
            all_categories.update(position_data.get('category_scores', {}).keys())
        
        if all_categories:
            cat_table = PrettyTable()
            cat_headers = ["Category"] + [f"Position {i}" for i in range(1, n_positions + 1)]
            cat_table.field_names = cat_headers
            
            # Sort categories by name
            for category in sorted(all_categories):
                row = [category]
                for position in range(1, n_positions + 1):
                    if (position in position_results and 
                        category in position_results[position].get('category_scores', {})):
                        score = position_results[position]['category_scores'][category]
                        row.append(f"{score:.4f}")
                    else:
                        row.append("-")
                cat_table.add_row(row)
            
            text_content.append(str(cat_table))
        else:
            text_content.append("No category data available.")
        
        text_content.append("")
        text_content.append("")
        
        # IF Type Analysis
        text_content.append("IF TYPE ANALYSIS")
        text_content.append("=" * 40)
        text_content.append("")
        
        # Collect all constraint types across all positions
        all_constraint_types = set()
        for position_data in position_results.values():
            all_constraint_types.update(position_data.get('constraint_scores', {}).keys())
        
        if all_constraint_types:
            if_table = PrettyTable()
            if_headers = ["IF Index"] + [f"Position {i}" for i in range(1, n_positions + 1)]
            if_table.field_names = if_headers
            
            # Sort constraint types by name
            for constraint_id in sorted(all_constraint_types):
                row = [constraint_id]
                for position in range(1, n_positions + 1):
                    if (position in position_results and 
                        constraint_id in position_results[position].get('constraint_scores', {})):
                        score = position_results[position]['constraint_scores'][constraint_id]
                        row.append(f"{score:.4f}")
                    else:
                        row.append("-")
                if_table.add_row(row)
            
            text_content.append(str(if_table))
        else:
            text_content.append("No constraint type data available.")
        
        text_content.append("")
        text_content.append("")
        
        # Detailed Position Analysis
        text_content.append("DETAILED POSITION ANALYSIS")
        text_content.append("=" * 40)
        text_content.append("")
        
        for position in range(1, n_positions + 1):
            text_content.append(f"Position {position} Analysis:")
            text_content.append("-" * 30)
            
            if position in position_results:
                pos_data = position_results[position]
                overall_score = pos_data.get('overall_score', 0.0)
                n_instances = pos_data.get('n_instances', 0)
                category_scores = pos_data.get('category_scores', {})
                constraint_scores = pos_data.get('constraint_scores', {})
                
                text_content.append(f"Overall Score: {overall_score:.4f} ({n_instances} instances)")
                text_content.append("")
                
                # Category breakdown for this position
                if category_scores:
                    text_content.append("Category Breakdown:")
                    cat_pos_table = PrettyTable()
                    cat_pos_table.field_names = ["Category", "Score"]
                    
                    # Sort by score descending
                    sorted_categories = sorted(category_scores.items(), key=lambda x: -x[1])
                    for category, score in sorted_categories:
                        cat_pos_table.add_row([category, f"{score:.4f}"])
                    
                    text_content.append(str(cat_pos_table))
                    text_content.append("")
                
                # Constraint type breakdown for this position
                if constraint_scores:
                    text_content.append("IF Type Breakdown:")
                    if_pos_table = PrettyTable()
                    if_pos_table.field_names = ["IF Index", "Score"]
                    
                    # Sort by score descending
                    sorted_constraints = sorted(constraint_scores.items(), key=lambda x: -x[1])
                    for constraint_id, score in sorted_constraints:
                        if_pos_table.add_row([constraint_id, f"{score:.4f}"])
                    
                    text_content.append(str(if_pos_table))
                    text_content.append("")
                
            else:
                text_content.append("No data available for this position.")
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
        description="Position analysis system for instruction following evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze bigcodebench
  python -m swe_if.position_analyzer --task generation --benchmark bigcodebench --model claude-sonnet-4 --code-instructions 2 --max-instructions 5
  
  # Analyze livecodebench
  python -m swe_if.position_analyzer --task generation --benchmark livecodebench --model gemini-3.1-pro-preview --code-instructions 3 --max-instructions 5

  # Analyze editing task
  python -m swe_if.position_analyzer --task editing --benchmark bigcodebench --model claude-sonnet-4 --code-instructions 1 --max-instructions 5
  
  # Analyze 0-instructions code with custom paths
  python -m swe_if.position_analyzer --task generation --benchmark bigcodebench --model gemini-3.1-pro-preview --code-instructions 0 --max-instructions 3 --taxonomy-path custom_taxonomy.csv
  
  # Specify custom output directory and overwrite
  python -m swe_if.position_analyzer --task generation --benchmark bigcodebench --model claude-sonnet-4 --code-instructions 2 --max-instructions 5 --output-dir custom_analysis --overwrite

Description:
  This tool analyzes how code from a specific instruction level performs on each
  instruction position. It loads the complete instruction definitions from the
  max-instructions file and tests the code against each position individually.
  
  The analysis shows:
  - Overall constraint-level IF score for each position
  - Category-wise breakdown (style, logic, doc, error, library)
  - IF type breakdown (individual constraint performance)
        """
    )
    
    parser.add_argument("--task", required=True, choices=['generation', 'editing'], help="Task type")
    parser.add_argument("--benchmark", required=True, choices=['bigcodebench', 'livecodebench'], help="Benchmark dataset")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--code-instructions", type=int, required=True, help="Number of instructions in the code file to analyze")
    parser.add_argument("--max-instructions", type=int, required=True, help="Maximum number of instructions (for instruction definitions)")
    parser.add_argument("--taxonomy-path", default="data/code_gym_taxonomy.csv", help="Path to taxonomy CSV file")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for analysis results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()
    args.model = args.model.replace('/', '__')  # sanitize model name for filesystem paths

    # Validate arguments
    if args.code_instructions > args.max_instructions:
        print(f"Error: code-instructions ({args.code_instructions}) cannot be greater than max-instructions ({args.max_instructions})")
        return 1
    
    # Construct file paths
    # Code file to analyze
    if args.task == 'editing' and args.code_instructions == 0:
        code_file = Path(f"results/{args.benchmark}/generation/{args.model}_{args.code_instructions}-instructions.jsonl")
    else:
        code_file = Path(f"results/{args.benchmark}/{args.task}/{args.model}_{args.code_instructions}-instructions.jsonl")
    
    # Instruction definition file (always use max-instructions from the same task)
    if args.task == 'editing':
        instruction_definition_file = Path(f"results/{args.benchmark}/generation/{args.model}_{args.max_instructions}-instructions.jsonl")
    else:
        instruction_definition_file = Path(f"results/{args.benchmark}/{args.task}/{args.model}_{args.max_instructions}-instructions.jsonl")
    
    # Check if input files exist
    if not code_file.exists():
        print(f"Error: Code file not found: {code_file}")
        return 1
    
    if not instruction_definition_file.exists():
        print(f"Error: Instruction definition file not found: {instruction_definition_file}")
        return 1
    
    # Setup output paths
    output_dir = Path(args.output_dir) / args.benchmark / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{args.model}_{args.code_instructions}-instructions_position-analysis.json"
    txt_path = output_dir / f"{args.model}_{args.code_instructions}-instructions_position-analysis.txt"
    
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
    
    # Initialize analyzer
    analyzer = PositionAnalyzer(taxonomy_path=args.taxonomy_path)
    
    # Print analysis info
    print(f"{'='*80}")
    print("POSITION ANALYSIS")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    print(f"Code file: {code_file}")
    print(f"Instruction definition file: {instruction_definition_file}")
    print(f"Code instructions: {args.code_instructions}")
    print(f"Max instructions: {args.max_instructions}")
    print()
    
    # Run analysis
    try:
        results = analyzer.analyze_code_positions(str(code_file), str(instruction_definition_file))
        
        # Save results
        analyzer.save_results(results, str(json_path))
        analyzer.generate_summary_text(results, str(txt_path))
        
        # Print final summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED")
        print(f"{'='*80}")
        
        position_results = results.get('position_results', {})
        n_positions = results.get('n_positions', 0)
        
        # Print overall position scores
        print("Position Analysis Summary:")
        
        table = PrettyTable()
        table.field_names = ["Position", "Overall Score", "Instances"]
        
        for position in range(1, n_positions + 1):
            if position in position_results:
                pos_data = position_results[position]
                overall_score = pos_data.get('overall_score', 0.0)
                n_instances = pos_data.get('n_instances', 0)
                table.add_row([f"Position {position}", f"{overall_score:.4f}", str(n_instances)])
            else:
                table.add_row([f"Position {position}", "0.0000", "0"])
        
        print(table)
        
        print(f"\nDetailed analysis results: {json_path}")
        print(f"Summary report: {txt_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())