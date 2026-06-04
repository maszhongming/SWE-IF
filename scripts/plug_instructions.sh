# gemini
python3 -m swe_if.instruction_plugger \
    --benchmark livecodebench \
    --model gemini-3.1-pro-preview \
    --n-instructions 5 \
    --concurrent

# claude
# python3 -m swe_if.instruction_plugger \
#     --benchmark livecodebench \
#     --model claude-opus-4@20250514-thinking \
#     --n-instructions 5 \
#     --concurrent