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
Analyze instruction distribution in plugged benchmark instances.
This module provides functions to analyze the distribution of instruction_idx.
"""

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_instances(file_path: str) -> List[Dict[str, Any]]:
    """Load instances from JSONL file."""
    instances = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        instance = json.loads(line)
                        instances.append(instance)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
        
        print(f"Loaded {len(instances)} instances from {file_path}")
        return instances
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []


def analyze_instruction_distribution(instances: List[Dict[str, Any]], n_instructions: int) -> Dict[str, Any]:
    """
    Analyze distribution of instruction_idx in the first n_instructions.
    
    Args:
        instances: List of benchmark instances
        n_instructions: Number of first instructions to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    if not instances:
        return {"error": "No instances to analyze"}
    
    # Collect instruction statistics
    total_instances = len(instances)
    instances_with_instructions = 0
    instruction_counts = Counter()
    position_counts = defaultdict(Counter)  # position -> {instruction_idx: count}
    position_category_counts = defaultdict(Counter)  # position -> {category: count}
    category_counts = Counter()  # category -> count
    category_instance_counts = Counter()  # category -> number of instances containing this category
    instruction_instance_counts = Counter()  # instruction_idx -> number of instances containing this instruction
    
    # Track instruction_list lengths
    length_distribution = Counter()
    
    for instance in instances:
        instruction_list = instance.get('instruction_list', [])
        length_distribution[len(instruction_list)] += 1
        
        if instruction_list:
            instances_with_instructions += 1
            
            # Track which categories and instructions appear in this instance
            categories_in_instance = set()
            instructions_in_instance = set()
            
            # Analyze first n_instructions
            for pos, instruction in enumerate(instruction_list[:n_instructions]):
                instruction_idx = instruction.get('instruction_idx', 'unknown')
                instruction_counts[instruction_idx] += 1
                position_counts[pos][instruction_idx] += 1
                instructions_in_instance.add(instruction_idx)
                
                # Extract category (prefix before underscore)
                category = instruction_idx.split('_')[0] if '_' in instruction_idx else instruction_idx
                category_counts[category] += 1
                categories_in_instance.add(category)
                position_category_counts[pos][category] += 1
            
            # Count instance-level occurrences
            for category in categories_in_instance:
                category_instance_counts[category] += 1
            for instruction_idx in instructions_in_instance:
                instruction_instance_counts[instruction_idx] += 1
    
    # Prepare results
    results = {
        "total_instances": total_instances,
        "instances_with_instructions": instances_with_instructions,
        "instances_without_instructions": total_instances - instances_with_instructions,
        "coverage_rate": instances_with_instructions / total_instances * 100 if total_instances > 0 else 0,
        "length_distribution": dict(length_distribution),
        "instruction_counts": dict(instruction_counts),
        "category_counts": dict(category_counts),
        "category_instance_counts": dict(category_instance_counts),
        "instruction_instance_counts": dict(instruction_instance_counts),
        "position_counts": {pos: dict(counts) for pos, counts in position_counts.items()},
        "position_category_counts": {pos: dict(counts) for pos, counts in position_category_counts.items()},
        "n_instructions_analyzed": n_instructions
    }
    
    return results


def print_analysis_results(results: Dict[str, Any]) -> None:
    """
    Pretty print analysis results.
    
    Args:
        results: Analysis results from analyze_instruction_distribution
    """
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    n_instructions = results["n_instructions_analyzed"]
    total_instances = results["total_instances"]
    instances_with_instructions = results["instances_with_instructions"]
    instruction_counts = results["instruction_counts"]
    category_counts = results["category_counts"]
    category_instance_counts = results["category_instance_counts"]
    instruction_instance_counts = results["instruction_instance_counts"]
    position_counts = results["position_counts"]
    position_category_counts = results.get("position_category_counts", {})
    length_distribution = results["length_distribution"]
    
    # Print analysis results
    print("=" * 80)
    print("INSTRUCTION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print(f"\nGeneral Statistics:")
    print(f"  Total instances: {total_instances}")
    print(f"  Instances with instructions: {instances_with_instructions}")
    print(f"  Instances without instructions: {results['instances_without_instructions']}")
    print(f"  Coverage rate: {results['coverage_rate']:.1f}%")
    
    # Instruction list length distribution
    print(f"\nInstruction List Length Distribution:")
    for length in sorted(length_distribution.keys()):
        count = length_distribution[length]
        percentage = count / total_instances * 100
        print(f"  {length} instructions: {count} instances ({percentage:.1f}%)")
    
    if not instruction_counts:
        print("\nNo instructions found in any instance.")
        return
    
    # Category distribution (UPDATED)
    print(f"\nCategory Distribution (First {n_instructions} Instructions):")
    print(f"Total instruction occurrences: {sum(category_counts.values())}")
    print(f"Unique categories: {len(category_counts)}")
    print()
    
    # Sort categories by frequency
    sorted_categories = Counter(category_counts).most_common()
    
    for category, count in sorted_categories:
        percentage = count / sum(category_counts.values()) * 100
        true_coverage = category_instance_counts[category] / instances_with_instructions * 100
        avg_per_instance = count / instances_with_instructions
        print(f"  {category:<12}: {count:>4} occurrences ({percentage:>5.1f}% of all)")
        print(f"               Coverage: {true_coverage:>5.1f}% of instances, Avg: {avg_per_instance:.2f} per instance")
    
    # Overall instruction distribution (first n_instructions)
    print(f"\nDetailed Instruction Distribution (First {n_instructions} Instructions):")
    print(f"Total instruction occurrences: {sum(instruction_counts.values())}")
    print(f"Unique instruction types: {len(instruction_counts)}")
    print()
    
    # Sort by frequency (most common first)
    sorted_instructions = Counter(instruction_counts).most_common()
    
    for instruction_idx, count in sorted_instructions:
        percentage = count / sum(instruction_counts.values()) * 100
        true_coverage = instruction_instance_counts[instruction_idx] / instances_with_instructions * 100
        print(f"  {instruction_idx:<15}: {count:>4} occurrences ({percentage:>5.1f}% of all, {true_coverage:>5.1f}% coverage)")
    
    # Position-wise analysis
    max_positions = min(n_instructions, max(position_counts.keys()) + 1 if position_counts else 0)
    print(f"\nPosition-wise Distribution (First {max_positions} Positions):")
    
    for pos in sorted(position_counts.keys()):
        if pos >= n_instructions:
            break
            
        print(f"\n  Position {pos + 1}:")
        pos_total = sum(position_counts[pos].values())
        unique_instructions = len(position_counts[pos])
        unique_categories = len(position_category_counts.get(pos, {}))
        
        print(f"    Total: {pos_total} instructions, {unique_instructions} unique instructions, {unique_categories} unique categories")
        
        # Sort by frequency within this position
        sorted_pos_instructions = Counter(position_counts[pos]).most_common()
        
        print(f"    Instruction distribution:")
        for instruction_idx, count in sorted_pos_instructions:
            percentage = count / pos_total * 100
            print(f"      {instruction_idx:<15}: {count:>4} ({percentage:>5.1f}%)")
        
        # Add category distribution for this position
        if pos in position_category_counts:
            print(f"    Category distribution:")
            sorted_pos_categories = Counter(position_category_counts[pos]).most_common()
            for category, count in sorted_pos_categories:
                percentage = count / pos_total * 100
                print(f"      {category:<12}: {count:>4} ({percentage:>5.1f}%)")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    avg_instructions_per_instance = sum(instruction_counts.values()) / instances_with_instructions if instances_with_instructions > 0 else 0
    print(f"  Average instructions per instance (with instructions): {avg_instructions_per_instance:.2f}")
    
    # Most and least common instructions
    if sorted_instructions:
        most_common = sorted_instructions[0]
        least_common = sorted_instructions[-1]
        print(f"  Most common instruction: {most_common[0]} ({most_common[1]} occurrences)")
        print(f"  Least common instruction: {least_common[0]} ({least_common[1]} occurrences)")
        
        # Category diversity
        if sorted_categories:
            most_common_category = sorted_categories[0]
            least_common_category = sorted_categories[-1]
            print(f"  Most common category: {most_common_category[0]} ({most_common_category[1]} occurrences)")
            print(f"  Least common category: {least_common_category[0]} ({least_common_category[1]} occurrences)")
        
        # Diversity measure
        diversity = len(instruction_counts) / max(instruction_counts.values()) if instruction_counts else 0
        category_diversity = len(category_counts) / max(category_counts.values()) if category_counts else 0
        print(f"  Instruction distribution diversity: {diversity:.2f} (higher = more evenly distributed)")
        print(f"  Category distribution diversity: {category_diversity:.2f} (higher = more evenly distributed)")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze instruction distribution in plugged benchmark instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze first 3 instructions
  python -m swe_if.analyze_instruction_distribution \\
    data/bigcodebench/claude-sonnet-4@20250514-thinking_4-instructions.jsonl \\
    --n-instructions 3

  # Analyze all instructions (set n-instructions high)
  python -m swe_if.analyze_instruction_distribution \\
    data/bigcodebench/gemini-2.5-flash-preview-05-20_5-instructions.jsonl \\
    --n-instructions 10
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the JSONL file containing plugged instructions"
    )
    
    parser.add_argument(
        "--n-instructions",
        type=int,
        required=True,
        help="Number of first instructions to analyze"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.n_instructions <= 0:
        print("Error: n-instructions must be positive")
        return
    
    if not Path(args.file_path).exists():
        print(f"Error: File does not exist: {args.file_path}")
        return
    
    # Load and analyze
    instances = load_instances(args.file_path)
    if instances:
        results = analyze_instruction_distribution(instances, args.n_instructions)
        print_analysis_results(results)


if __name__ == "__main__":
    main()