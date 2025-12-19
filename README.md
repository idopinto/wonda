# Loop Invariant Generation

A framework for generating loop invariants using LLMs to accelerate traditional program verifiers.

## Overview

This project fine-tunes LLMs (GPT-OSS) to generate candidate loop invariants for C programs. The generated invariants help traditional verifiers like UAutomizer prove or disprove program correctness faster.

## Project Structure

```
inv-gen/
├── configs/
│   ├── eval/config.yaml      # Evaluation configuration
│   ├── train/config.yaml     # Training configuration
│   └── global_config.py      # Global paths and settings
├── scripts/
│   └── eval/
│       ├── eval_hf.sbatch    # Evaluation with HuggingFace client
│       └── eval_vllm.sbatch  # Evaluation with vLLM server
├── src/
│   ├── eval/                 # Evaluation pipeline
│   ├── train/sft/            # SFT training scripts
│   ├── preprocess/           # Data preprocessing
│   ├── utils/                # Utility classes (Program, Predicate)
│   └── verifiers/            # UAutomizer wrapper
└── nbs/                      # Jupyter notebooks for analysis
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (H200 recommended for vLLM, 2x L40S for HF)
- Access to SLURM cluster

### Installation

```bash
# Clone the repository
cd /cs/labs/guykatz/idopinto12/projects/inv-gen

# Install dependencies with uv
uv sync
```

## Evaluation

### Option 1: vLLM Server (Recommended for H200)

Uses vLLM with LoRA adapters for fast inference via OpenAI-compatible API.

```bash
# Basic usage
sbatch scripts/eval/eval_vllm.sbatch

# With Hydra overrides
sbatch scripts/eval/eval_vllm.sbatch dataset.split=easy dataset.limit=5
sbatch scripts/eval/eval_vllm.sbatch dataset.split=hard
```

### Option 2: HuggingFace Direct (For L40S or when H200 unavailable)

Loads the model directly using HuggingFace Transformers.

```bash
# Basic usage
sbatch scripts/eval/eval_hf.sbatch

# With Hydra overrides
sbatch scripts/eval/eval_hf.sbatch dataset.split=easy dataset.limit=5
sbatch scripts/eval/eval_hf.sbatch model.eval_finetuned_model=true
```

### Local Testing

```bash
# Run evaluation locally (requires GPU)
uv run -m src.eval.evaluate model.client=hf dataset.limit=1

# With vLLM (start server first)
vllm serve openai/gpt-oss-20b --enable-lora \
    --lora-modules gen_inv_adapter=idopinto/gpt-oss-20b-rlinv-sft-sep \
    --max-lora-rank 8

# Then run evaluation
uv run -m src.eval.evaluate model.client=vllm dataset.limit=1
```

## Configuration

Configuration is managed via Hydra. Override any config value from command line:

### Common Overrides

| Override | Description | Example |
|----------|-------------|---------|
| `model.client` | Inference backend | `hf` or `vllm` |
| `dataset.split` | Data split | `easy` or `hard` |
| `dataset.limit` | Number of samples | `5` or `-1` (all) |
| `dataset.prefix` | Filter by prefix | `loop_` |
| `model.reasoning_effort` | GPT-OSS reasoning | `low`, `medium`, `high` |
| `weave.skip_wandb` | Skip W&B upload | `true` |

### Full Config Reference

See `configs/eval/config.yaml` for all options:

```yaml
model:
  client: vllm              # or hf
  base_model_name: "openai/gpt-oss-20b"
  reasoning_effort: "medium"
  vllm_base_url: "http://localhost:8000/v1"
  vllm_model: "gen_inv_adapter"

dataset:
  name: "idopinto/invbench-evaluation-uautomizer25-k3"
  split: "hard"
  limit: -1

scorer:
  verifier:
    version: "25"
    timeout_seconds: 600.0
```

## Training

### SFT Training

```bash
# Run training
uv run -m src.train.sft.train_sft

# Or with Unsloth (faster)
uv run -m src.train.sft.train_unsloth
```

Training config: `configs/train/config.yaml`

## Models

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| Base | GPT-OSS-20B | `openai/gpt-oss-20b` |
| Fine-tuned | LoRA adapter | `idopinto/gpt-oss-20b-rlinv-sft-sep` |

## Monitoring

- **Weave**: Experiment tracking at [wandb.ai/ip-ai/eval-inv-gen](https://wandb.ai/ip-ai/eval-inv-gen)
- **SLURM logs**: `slurm/eval_*.out` and `slurm/eval_*.err`

```bash
# Check job status
squeue -u $USER

# View logs
tail -f slurm/eval_vllm_<job_id>.out
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  C Program      │────►│  LLM (GPT-OSS)   │────►│  Candidate      │
│  with loops     │     │  + LoRA adapter  │     │  Invariant      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Decision        │◄─────────────┘
                        │  Procedure       │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌─────────────────┐       ┌─────────────────┐
          │  Correctness    │       │  Usefulness     │
          │  Check (V₁)     │       │  Check (V₂)     │
          └────────┬────────┘       └────────┬────────┘
                   │                         │
                   └──────────┬──────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Final Decision │
                    │  TRUE/FALSE/UNK │
                    └─────────────────┘
```

## License

See [LICENSE](LICENSE) for details.
