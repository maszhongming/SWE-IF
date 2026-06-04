# SWE-IF: Aligning Code Evaluation with Human Preference

<p align="center">
  <a href="https://arxiv.org/abs/2510.07315"><img src="https://img.shields.io/badge/📝-Paper-2f7df6" height="24"></a>
  <a href="https://huggingface.co/datasets/MingZhong/SWE-IF"><img src="https://img.shields.io/badge/🤗-Dataset%20%26%20Taxonomy-f59122" height="24"></a>
  <a href="https://huggingface.co/datasets/MingZhong/SWE-IF-model-outputs"><img src="https://img.shields.io/badge/📦-Model%20Outputs-17b85a" height="24"></a>
</p>

**SWE-IF** augments standard code benchmarks with verifiable software-engineering
instructions, and evaluates LLMs on both functional correctness and instruction
following (IF), across single-turn generation and multi-turn editing (ICML 2026).

It has three parts:

- **VeriCode**: a taxonomy of 30 verifiable code instructions in 5 categories, each with a deterministic verifier (26/30 via `ruff`).
- **Big-SWE-IF** / **Live-SWE-IF**: BigCodeBench / LiveCodeBench augmented with 5 instructions per task.
- A **pipeline** to augment, generate/edit, evaluate, and analyze.

## 📁 Repository Structure

```
swe_if/
├── data.py                       # load benchmarks + taxonomy from HuggingFace
├── instruction_plugger.py        # augment a benchmark with taxonomy instructions
├── code_generator.py             # single-turn generation & multi-turn editing
├── evaluator.py                  # functional correctness + IF (the 30 verifiers live here)
├── position_analyzer.py          # per-position IF analysis
├── correlations.py               # IF vs. functional score correlations
├── generate_comparison_plots.py  # reproduce the paper's main-text figures
├── model_configs.py              # model configs (providers + decoding settings)
└── llm_interface.py              # provider wrappers (Vertex AI / OpenRouter)
scripts/                          # a runnable example for each pipeline step
scores/
├── bigcodebench/                 # func / IF / position results for Big-SWE-IF
└── livecodebench/                # func / IF / position results for Live-SWE-IF
```

## ⚙️ Installation

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Credentials are only needed to run models, not to load data:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project"   # Gemini / Claude / Mistral via Vertex AI
export OPENROUTER_API_KEY="your-openrouter-key"  # other models via OpenRouter
```

## 📥 Data

Everything loads from the HuggingFace dataset by default, so a fresh clone works out of the box:

```python
from datasets import load_dataset

big  = load_dataset("MingZhong/SWE-IF", "big_swe_if")    # 1,140 tasks (real-world)
live = load_dataset("MingZhong/SWE-IF", "live_swe_if")   # 1,055 tasks (algorithmic)
tax  = load_dataset("MingZhong/SWE-IF", "vericode")      # 30 instructions + verifiers
```

Each augmented task keeps its original benchmark fields and adds an `instruction_list`
(the instructions selected for it); see the
[dataset card](https://huggingface.co/datasets/MingZhong/SWE-IF) for a full field
reference. To run on your own augmented data, pass a local JSONL via `--input-data <file>`.

## 🚀 Pipeline

The four steps chain together, each reading the previous one's output. The
`--benchmark` / `--dataset` flag selects which benchmark to run on.

### 1 · Benchmark Augmentation (optional)

```bash
python -m swe_if.instruction_plugger --benchmark livecodebench --model gemini-3.1-pro-preview \
    --n-instructions 5 --concurrent
```

Attaches taxonomy instructions to a base benchmark and writes an augmented JSONL under
`data/<benchmark>/`. The released data is already augmented, so this step is only needed
if you want to regenerate it.

### 2 · Code Generation / Editing

```bash
# single-turn generation
python -m swe_if.code_generator --model gemini-3.1-pro-preview --dataset livecodebench \
    --task generation --n-instructions 5 --run-all --concurrent

# multi-turn editing
python -m swe_if.code_generator --model gemini-3.1-pro-preview --dataset livecodebench \
    --task editing --n-instructions 5 --concurrent
```

Runs a model on the augmented tasks and saves its responses under
`results/<benchmark>/{generation,editing}/`.

### 3 · Evaluation

```bash
python -m swe_if.evaluator --task editing --benchmark livecodebench \
    --model gemini-3.1-pro-preview --n-instructions 5 --run-all
```

Scores those responses for functional correctness and instruction following; the numbers
are printed and cached for later analysis.

### 4 · Analysis

```bash
python -m swe_if.position_analyzer --task editing --benchmark bigcodebench \
    --model gemini-3.1-pro-preview --max-instructions 5
python -m swe_if.correlations --benchmark livecodebench --format pdf --visualization
```

Breaks IF down by instruction position and correlates IF against functional scores,
producing the per-position tables and correlation plots. See `scripts/` for a
ready-to-run example of each step.

## 🧩 Adding a Model

Models live in `swe_if/model_configs.py`. To add one, add an entry to
`MODEL_CONFIGS` keyed by the model id:

```python
'gemini-3.1-pro-preview': {
    'provider': 'gemini',        # gemini | claude | mistral | openrouter
    'max_tokens': 32768,
    'temperature': 0.0,
},
```

`provider` selects the backend: `gemini` / `claude` / `mistral` go through Vertex AI
(need `GOOGLE_CLOUD_PROJECT`); `openrouter` uses the OpenRouter REST API (need
`OPENROUTER_API_KEY`, with the id being the OpenRouter model id, e.g. `x-ai/grok-4`).
Append `-thinking` to a Claude id to turn on thinking mode.

A model id **not** found in `MODEL_CONFIGS` falls back to a generic OpenRouter config
(`provider=openrouter`, `max_tokens=32768`, `temperature=0.0`), so most OpenRouter
models run without any edit; add an explicit entry for precise control.

## 📈 Visualization

The aggregated scores behind the paper's figures live under `scores/`. Reproduce the
main-text figures (functional regression, task-level IF, and position bias) with one
command, no API or network needed:

```bash
python -m swe_if.generate_comparison_plots
```

## ⚠️ Safety

Functional evaluation runs model-generated code:

- **Live-SWE-IF** is executed locally via `exec` with only soft guards
  (`reliability_guard` is *not* a security sandbox), so run it inside an isolated
  container or VM.
- **Big-SWE-IF** is sent to the public BigCodeBench evaluator Space
  (`bigcode-bigcodebench-evaluator.hf.space`), i.e. your solutions are uploaded
  there to run.

## 📄 License & Citation

Code is under Apache-2.0 (Copyright Google LLC); the dataset (VeriCode taxonomy +
benchmarks) is on HuggingFace under CC BY 4.0. Big-SWE-IF builds on BigCodeBench
(Apache-2.0) and Live-SWE-IF on LiveCodeBench (problems from LeetCode / AtCoder /
Codeforces, which retain their original terms). If you find SWE-IF useful, please cite us:

```bibtex
@inproceedings{zhong2026sweif,
  title     = {SWE-IF: Aligning Code Evaluation with Human Preference},
  author    = {Zhong, Ming and Zhou, Xiang and Chang, Ting-Yun and Wang, Qingze and Xu, Nan and Si, Xiance and Garrette, Dan and Upadhyay, Shyam and Liu, Jeremiah and Han, Jiawei and Schillings, Benoit and Sun, Jiao},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

`swe_if/livecodebench_utils.py` is adapted from
[SkyThought](https://github.com/NovaSky-AI/SkyThought) (Apache-2.0).
