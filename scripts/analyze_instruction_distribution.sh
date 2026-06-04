# MODEL_NAME="gemini-3.1-pro-preview"
MODEL_NAME="claude-opus-4@20250514-thinking"

python3 -m swe_if.analyze_instruction_distribution \
    data/bigcodebench/${MODEL_NAME}_5-instructions_fixed.jsonl \
    --n-instructions 5 \