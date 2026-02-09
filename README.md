# Not All Invariants Are Equal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Not All Invariants Are Equal: Curating Training Data to Accelerate Program Verification with SLMs"** ([arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX)).

## Abstract

The synthesis of inductive loop invariants is a critical bottleneck in automated program verification. While Large Language Models (LLMs) show promise in mitigating this issue, they often fail on hard instances, generating invariants that are invalid or computationally ineffective. While fine-tuning is a natural route to mitigate this limitation, obtaining high-quality training data for invariant generation remains an open challenge. We present a rigorous data curation pipeline designed to extract high-quality training signals from raw verifier-generated invariants. First, we formalize the properties required for a high-quality training invariant. Second, we propose **WONDA**, a pipeline that refines noisy data via AST-based normalization, followed by LLM-driven semantic rewriting and augmentation with provable quality guarantees. We demonstrate that fine-tuning Small Language Models (SLMs) on this curated dataset result in consistent and significant performance gain. In particular, a fine-tuned 4B parameter model matches the utility of a GPT-OSS-120B baseline and approaches the state-of-the-art GPT-5.2, without incurring reasoning-time overhead. On challenging instances from the recent InvBench evaluation suite, our approach doubles the invariant correctness rate of base models; and improves their Virtual Best Performance (VBP) rates on the verification task by up to 14.2%.

## Overview

This repository contains the code for **WONDA** (data curation pipeline) and the full training and evaluation framework for fine-tuning Small Language Models (SLMs) on loop invariant generation for C programs.

```
C Program → Fine-tuned SLM → Candidate Invariant → Decision Procedure
                                                        ├── Correct & Useful → Accelerated Verification
                                                        └── Invalid → Fallback to Baseline
```

## Installation

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (L40S or H200 recommended)
- [uv](https://github.com/astral-sh/uv) package manager
- [UAutomizer](https://zenodo.org/records/14209043) (for verification)

### Setup

```bash
# Clone the repository
git clone https://github.com/idopinto/inv-gen.git
cd inv-gen

# Install dependencies
uv sync

# Install dev dependencies (optional)
uv sync --extra dev
```

### UAutomizer Setup

Download and extract [UAutomizer (0.3.0-dev-d790fec)](https://zenodo.org/records/14209043) to `tools/UAutomizer25/`. The framework expects the following structure:

```
tools/
└── UAutomizer25/
    └── Ultimate.py
```

## Quick Start

Run a quick evaluation with a small sample:

```bash
# Evaluate base model on 5 samples
uv run -m wonda.eval.evaluate dataset.limit=5

# Evaluate fine-tuned model
uv run -m wonda.eval.evaluate dataset.limit=5 models.eval_ft_model=true
```

## Reproducing Paper Results

### 1. Data Preparation

The preprocessed datasets are available on HuggingFace:

- **InvBench (evaluation)**: [`idopinto/invbench-eval-uautomizer25-k3-l40s-full-runlim`](https://huggingface.co/datasets/idopinto/invbench-eval-uautomizer25-k3-l40s-full-runlim)
- **Training (raw)**: [`invbench-train-uautomizer25-k1-full-raw.json`](data/invbench-train-uautomizer25-k1-full-raw.json) — raw verifier-generated invariants before WONDA curation
- **Training (curated)**: [`invbench-train-uautomizer25-k1-full-parallel-latest.json`](data/invbench-train-uautomizer25-k1-full-parallel-latest.json) — training data after the WONDA pipeline (AST normalization + LLM-driven semantic rewriting)
- **Training (HuggingFace)**: versioned splits based on quality grade — [v2.1](https://huggingface.co/datasets/idopinto/invbench-train-uautomizer25-k1-v2.1), [v2.2](https://huggingface.co/datasets/idopinto/invbench-train-uautomizer25-k1-v2.2), [v2.3](https://huggingface.co/datasets/idopinto/invbench-train-uautomizer25-k1-v2.3)

> **Note:** The evaluation dataset above was built on an NVIDIA L40S GPU to match the hardware used for local model inference (0.6B--8B parameters; up to ~14B fits on an L40S for inference). If you plan to evaluate larger local models requiring a more powerful GPU, we recommend rebuilding the dataset from scratch on your hardware infrastructure, as baseline timings and easy/hard splits may differ across machines.

To rebuild the datasets from scratch:

```bash
# Build evaluation dataset (requires UAutomizer)
sbatch scripts/preprocess/build_eval_dataset.sh

# Build training dataset
sbatch scripts/preprocess/build_train_dataset.sh

# Run WONDA pipeline (clean, normalize, and grade training data)
sbatch scripts/preprocess/build_v1_train_dataset.sbatch
```

### 2. Training

Fine-tune a Qwen3 model with LoRA:

```bash
# Train Qwen3-0.6B
sbatch scripts/train/train_qwen3_0.6b.sbatch

# Train larger models
sbatch scripts/train/train_qwen3_4b.sbatch
sbatch scripts/train/train_qwen3_8b.sbatch
```

Training configuration is in `configs/train/`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sft.num_train_epochs` | 3 | Number of training epochs |
| `sft.learning_rate` | 0.0002 | Learning rate |
| `lora.r` | 8 | LoRA rank |
| `lora.lora_alpha` | 16 | LoRA alpha |

### 3. Evaluation

Run multi-trial evaluation with confidence intervals:

```bash
# Run 3 trials with statistical aggregation
sbatch scripts/eval/run_multi_eval.sbatch --num_runs=3 models=qwen3_0.6b_nt_config

# Evaluate specific fine-tuned model
sbatch scripts/eval/run_multi_eval.sbatch --num_runs=3 \
    models=qwen3_0.6b_nt_config \
    models.eval_ft_model=true \
    models.ft_model.sft_version="v2.3"
```

Results are saved to `experiments/` with aggregated statistics.

## Project Structure

```
inv-gen/
├── configs/
│   ├── eval/           # Evaluation configs (Hydra)
│   ├── train/          # Training configs
│   └── preprocess/     # Data preprocessing configs
├── scripts/
│   ├── eval/           # SLURM evaluation scripts
│   ├── train/          # SLURM training scripts
│   └── preprocess/     # SLURM preprocessing scripts
├── src/
│   ├── eval/           # Evaluation pipeline
│   │   ├── evaluate.py         # Main evaluation entry point
│   │   ├── run_multi_eval.py   # Multi-trial evaluation
│   │   ├── decision_procedure.py
│   │   └── models/             # Model implementations
│   ├── preprocess/     # WONDA data curation pipeline
│   │   ├── build_eval_dataset.py
│   │   ├── build_v1_train_dataset.py
│   │   └── clean_invariants.py
│   ├── train/          # Training code
│   │   └── sft/        # Supervised fine-tuning
│   ├── utils/          # Utilities
│   └── verifiers/      # UAutomizer wrapper
├── tests/              # Test suite
└── docs/               # Documentation
```

## Configuration

Configuration uses [Hydra](https://hydra.cc/). Override any parameter from the command line:

```bash
# Common overrides
uv run -m wonda.eval.evaluate \
    dataset.split=easy \
    dataset.limit=10 \
    models=qwen3_0.6b_nt_config \
    scorer.verifier.timeout_seconds=300
```

See `configs/eval/config.yaml` for all available options.

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `invbench-eval-*` | InvBench evaluation benchmark | 219 programs |
| `invbench-train-*-raw` | Raw verifier-generated invariants (before WONDA) | — |
| `invbench-train-*-v2.1` | WONDA-curated training data (quality grade 1--3) | — |
| `invbench-train-*-v2.2` | WONDA-curated training data (quality grade 2--3) | — |
| `invbench-train-*-v2.3` | WONDA-curated training data (quality grade 3 only) | — |

All datasets are available on [HuggingFace](https://huggingface.co/idopinto).

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
