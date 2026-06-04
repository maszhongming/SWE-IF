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

"""Unified data loading for SWE-IF.

By default everything is loaded from the public HuggingFace dataset
`MingZhong/SWE-IF` (so a fresh clone works without any local data). Pass an
explicit local path to use your own files instead (e.g. after re-running the
augmentation pipeline yourself).

Configs on the Hub:
  - "vericode"     : the VeriCode taxonomy (30 verifiable instructions)
  - "big_swe_if"   : Big-SWE-IF  (BigCodeBench augmented, full incl. tests)
  - "live_swe_if"  : Live-SWE-IF (LiveCodeBench augmented, full incl. tests)
"""
import json

HF_REPO = "MingZhong/SWE-IF"

# Map the internal benchmark name to the HuggingFace config name.
BENCHMARK_TO_CONFIG = {
    "bigcodebench": "big_swe_if",
    "livecodebench": "live_swe_if",
}


def load_benchmark(dataset_name, local_file=None):
    """Return a list of augmented instances (original benchmark fields + an
    `instruction_list`). Loads from HF `MingZhong/SWE-IF` by default, or from a
    local JSONL file if `local_file` is given."""
    if local_file:
        with open(local_file) as f:
            return [json.loads(line) for line in f if line.strip()]
    if dataset_name not in BENCHMARK_TO_CONFIG:
        raise ValueError(
            f"Unknown benchmark '{dataset_name}'. "
            f"Expected one of {list(BENCHMARK_TO_CONFIG)}."
        )
    from datasets import load_dataset
    config = BENCHMARK_TO_CONFIG[dataset_name]
    return list(load_dataset(HF_REPO, config, split="train"))


def load_taxonomy(local_path=None):
    """Return the VeriCode taxonomy as a pandas DataFrame. Loads from HF
    `MingZhong/SWE-IF` (config `vericode`) by default, or from a local CSV/JSONL
    if `local_path` is given."""
    import pandas as pd
    if local_path:
        if str(local_path).endswith(".jsonl"):
            return pd.read_json(local_path, lines=True)
        return pd.read_csv(local_path)
    from datasets import load_dataset
    return load_dataset(HF_REPO, "vericode", split="train").to_pandas()
