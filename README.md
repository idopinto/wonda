# Not All Invariants Are Equal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Not All Invariants Are Equal: Curating Training Data to Accelerate Program Verification with SLMs"** ([arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX)).

## Abstract

The synthesis of inductive loop invariants is a critical bottleneck in automated program verification. While Large Language Models (LLMs) show promise in mitigating this issue, they often fail on hard instances, generating invariants that are invalid or computationally ineffective. While fine-tuning is a natural route to mitigate this limitation, obtaining high-quality training data for invariant generation remains an open challenge. We present a rigorous data curation pipeline designed to extract high-quality training signals from raw verifier-generated invariants. First, we formalize the properties required for a high-quality training invariant. Second, we propose **WONDA**, a pipeline that refines noisy data via AST-based normalization, followed by LLM-driven semantic rewriting and augmentation with provable quality guarantees. We demonstrate that fine-tuning Small Language Models (SLMs) on this curated dataset result in consistent and significant performance gain. In particular, a fine-tuned 4B parameter model matches the utility of a GPT-OSS-120B baseline and approaches the state-of-the-art GPT-5.2, without incurring reasoning-time overhead. On challenging instances from the recent InvBench evaluation suite, our approach doubles the invariant correctness rate of base models; and improves their Virtual Best Performance (VBP) rates on the verification task by up to 14.2%.

## Overview

This repository contains the code for **WONDA** (data curation pipeline) and the full training and evaluation framework for fine-tuning Small Language Models (SLMs) on loop invariant generation for C programs.

**Released models and datasets:** All fine-tuned models and datasets are available in the [Wonda HuggingFace collection](https://huggingface.co/collections/idopinto/wonda).

![WONDA evaluation overview](figures/vbs_figure.png)


## Installation

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (we used L40S or H200)
- [uv](https://github.com/astral-sh/uv) package manager
- [UAutomizer](https://zenodo.org/records/14209043) (for verification)
- [runlim](https://github.com/arminbiere/runlim) (for time/memory limits during verification)
- [Weights & Biases (W&B)](https://wandb.ai) account (optional but recommended): we use **wandb** for experiment tracking when training fine-tuned models, and [W&B Weave](https://docs.wandb.ai/weave) for evaluation observability; a W&B account gives you the full experience.

### Setup

```bash
# Clone the repository
git clone https://github.com/idopinto/wonda.git
cd wonda

# Install dependencies
uv sync

# Install dev dependencies (optional)
uv sync --extra dev
```

### Environment variables

Copy `.env.example` to `.env` and fill in any credentials you need:

```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, Together, OpenRouter, HuggingFace, W&B, etc.)
```

Variables are optional and depend on what you run: API keys for external model backends, `HF_TOKEN` for private models or pushing to the Hub, `WANDB_API_KEY` for experiment tracking. See `.env.example` for the full list.

### UAutomizer Setup

Download and extract [UAutomizer (0.3.0-dev-d790fec)](https://zenodo.org/records/14209043) to `tools/UAutomizer25/`. The framework expects the following structure:

```
tools/
└── UAutomizer25/
    └── Ultimate.py
```
Other releases can be found here:
https://fm-tools.sosy-lab.org/#tool-uautomizer

### Runlim Setup

Build or install [runlim](https://github.com/arminbiere/runlim) and place the `runlim` binary at `tools/runlim/runlim`. It is used to enforce time and memory limits when running the verifier (e.g. for evaluation and dataset building).

## Quick Start

Run a quick multi-run evaluation on a small sample (e.g. 5 instances). For a fast local smoke test, use `multi_run.num_runs=1` together with `weave.skip_weave=true` and `weave.test_mode=true`:

```bash
# GPT-5.2 (OpenRouter API; set OPENROUTER_API_KEY in .env)
uv run -m wonda.eval.run_multi_eval multi_run.num_runs=1 dataset.limit=5 weave.skip_weave=true weave.test_mode=true models=gpt_5.2_config

# Qwen3-0.6B base (no fine-tuning)
uv run -m wonda.eval.run_multi_eval multi_run.num_runs=1 dataset.limit=5 weave.skip_weave=true weave.test_mode=true models=qwen3_0.6b_nt_config

# Qwen3-0.6B fine-tuned v2.2 (WONDA SFT)
uv run -m wonda.eval.run_multi_eval multi_run.num_runs=1 dataset.limit=5 weave.skip_weave=true weave.test_mode=true models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
```

Evaluation requires UAutomizer and runlim (see Installation). We recommend using [W&B Weave](https://docs.wandb.ai/weave) for observability (traces, prompts, and model outputs); for quick local tests, disable it with `weave.skip_weave=true` and keep outputs isolated with `weave.test_mode=true`. Evaluation runs model inference with Weave parallelism (`WEAVE_PARALLELISM`; single-run default 8, multi-run default 1); UAutomizer verification is throttled via a semaphore (`VERIFIER_MAX_CONCURRENT`, default 1) to avoid resource contention. More models and full benchmark runs are described in [wonda/eval/README.md](wonda/eval/README.md).

## Reproducing Paper Results

### 1. Data Preparation

All datasets are on HuggingFace (see the [Wonda collection](https://huggingface.co/collections/idopinto/wonda) for the full list). See [wonda/preprocess/README.md](wonda/preprocess/README.md) for dataset details and how to rebuild them.

### 2. Training

We release fine-tuned Qwen3 (non-think) models on HuggingFace; training uses LoRA and the WONDA SFT datasets. See [wonda/train/README.md](wonda/train/README.md) for released models and how to train your own.

### 3. Evaluation

We run multi-trial evaluation and report the metrics described in the paper. The metrics (validation, correctness, usefulness, VBS, speedup, etc.) are implemented and explained in [wonda/eval/inv_gen_scorer.py](wonda/eval/inv_gen_scorer.py). See [wonda/eval/README.md](wonda/eval/README.md) for the evaluation process, models used in the paper, and how to run evaluations.

## Project Structure

```
wonda/
├── configs/
│   ├── eval/           # Evaluation configs (Hydra), incl. model configs in models/
│   ├── train/          # Training configs (qwen3_0.6b … qwen3_14b)
│   └── preprocess/     # Preprocessing configs (build_eval_dataset, build_sft_dataset, wonda_data_pipeline)
├── scripts/
│   ├── eval/           # SLURM scripts: run_evaluation.sbatch, run_multi_eval.sbatch
│   ├── train/          # SLURM scripts: train_qwen.sbatch, train_and_eval_qwen.sbatch
│   └── preprocess/     # SLURM/shell: build_eval_dataset.sbatch, build_raw_train_dataset.sbatch, wonda_pipeline.sbatch, build_sft_dataset.sh
├── wonda/              # Main package
│   ├── core/           # AST and property utilities (ast_program.py, property.py)
│   ├── eval/           # Evaluation pipeline
│   │   ├── evaluate.py, run_multi_eval.py, inv_gen_scorer.py
│   │   ├── decision_procedure.py, eval_data.py, validate.py, aggregate_results.py
│   │   └── models/     # Model backends (qwen_model, oss_model, open_router_model, vllm_model, etc.)
│   ├── preprocess/     # WONDA data curation pipeline
│   │   ├── build_eval_dataset.py, build_raw_train_dataset.py, build_sft_dataset.py
│   │   ├── wonda_pipeline.py, baseline_dataset_common.py
│   │   └── gt_invariant_normalization.py, gt_invariant_simplification.py
│   ├── train/          # Training: train_qwen.py, data_utils.py, train_utils.py
│   └── verifiers/      # UAutomizer wrapper (uautomizer.py)
├── tests/              # Test suite (see [tests/README.md](tests/README.md))
└── tools/              # UAutomizer and runlim (see Installation)
```

## Testing

See **[tests/README.md](tests/README.md)** for how to run the test suite, what to run, and a summary of test modules. Tests use no external tools (gcc, UAutomizer, runlim) or network and are suitable for CI.

## Citation

```bibtex
@article{pinto2026invariants,
  title={Not All Invariants Are Equal: Curating Training Data to Accelerate Program Verification with SLMs},
  author={Pinto, Ido},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
