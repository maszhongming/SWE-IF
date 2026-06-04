#!/bin/bash

# EVALUATING GENERATION AND EDITING TASK


python3 -m swe_if.evaluator \
    --task editing \
    --benchmark bigcodebench \
    --model gemini-3.1-pro-preview \
    --n-instructions 5 \
    --overwrite