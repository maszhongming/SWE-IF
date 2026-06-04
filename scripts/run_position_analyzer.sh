#!/bin/bash

python3 -m swe_if.position_analyzer \
    --task editing \
    --benchmark bigcodebench \
    --model gemini-3.1-pro-preview \
    --code-instructions 5 \
    --max-instructions 5 \