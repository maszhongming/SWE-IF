#!/bin/bash

# Edit code for benchmarks with follow-up instructions

# Example: Run editing task
python3 -m swe_if.code_generator \
    --model gemini-3.1-pro-preview \
    --dataset bigcodebench \
    --task editing \
    --n-instructions 5 \
    --concurrent \