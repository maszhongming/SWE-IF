#!/bin/bash

# Show parameter specifications
# python3 -m swe_if.benchmark_fixer --show-specs

# Debug parameter parsing (show raw fields)
# python3 -m swe_if.benchmark_fixer --debug-params

# Detect issues only
python3 -m swe_if.benchmark_fixer --detect data/livecodebench/claude-opus-4@20250514-thinking_5-instructions_fixed.jsonl

# Detect and fix issues (saves to same directory with '_fixed' suffix)
# python3 -m swe_if.benchmark_fixer --detect --fix data/livecodebench/claude-opus-4@20250514-thinking_5-instructions.jsonl

# Fix with custom output path
# python3 -m swe_if.benchmark_fixer --detect --fix data/bigcodebench/gemini-3.1-pro-preview_5-instructions.jsonl --output fixed_benchmark.jsonl