#!/bin/bash

# Generate code for benchmarks with optional instruction following

# RECOMMENDED: Run ALL instruction counts from 0 to n
python3 -m swe_if.code_generator \
    --model gemini-3.1-pro-preview \
    --dataset bigcodebench \
    --task generation \
    --n-instructions 5 \
    --run-all \
    --concurrent \