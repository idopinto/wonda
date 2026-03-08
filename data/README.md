# Data directory

This directory holds inputs and outputs for the WONDA pipeline: raw C programs, property files, and generated evaluation/training datasets. Paths are defined in [`configs/global_config.py`](../configs/global_config.py).

## Layout

```
data/
├── eval/                    # Evaluation datasets and sources
│   ├── orig_programs/       # C programs used for evaluation (e.g. from InvBench)
│   └── <benchmark-name>/    # Built evaluation datasets (JSON, splits). See [eval/README.md](eval/README.md).
├── train/                   # Training datasets and sources
│   ├── orig_programs/       # C programs used for training (e.g. from InvBench)
│   ├── wonda-train-dataset-full-raw/       # Raw verifier-generated invariants
│   ├── wonda-train-dataset-full-v2/        # Curated dataset (WONDA pipeline output)
│   └── sft-ready/           # SFT-ready datasets (e.g. wonda-qwen-nt-sft-v0, v1, v2-g1, v2-g2)
├── properties/              # Verifier property files (e.g. unreach-call.prp)
└── InvBench/                # Optional: clone of [InvBench](https://github.com/Anjiang-Wei/InvBench) (training & evaluation programs)
```

## Using pre-built datasets (recommended)

All released datasets are on HuggingFace. You do **not** need to populate `data/` to run evaluation or training if you use the Hub:

- **Evaluation:** [`idopinto/wonda-eval-benchmark-full`](https://huggingface.co/datasets/idopinto/wonda-eval-benchmark-full) — used by default in the eval config.
- **Training (raw / curated / SFT):** See the [Wonda HuggingFace collection](https://huggingface.co/collections/idopinto/wonda) and [wonda/preprocess/README.md](../wonda/preprocess/README.md).

Configs point to these by default; no local `data/` files are required unless you build datasets from scratch.

## Building datasets from scratch

If you want to rebuild evaluation or training datasets locally (e.g. to match your hardware for baseline timings and easy/hard splits):

1. **Source programs:** Obtain C programs (e.g. clone [InvBench](https://github.com/Anjiang-Wei/InvBench) into `data/InvBench/` or copy programs into `data/eval/orig_programs/` and `data/train/orig_programs/`).
2. **Property files:** Ensure `data/properties/` contains the required `.prp` files (e.g. `unreach-call.prp`).
3. **Run the pipeline:** See [wonda/preprocess/README.md](../wonda/preprocess/README.md) for commands (e.g. `build_eval_dataset`, `build_raw_train_dataset`, `wonda_pipeline`, `build_sft_dataset`).

Scripts and configs are under `configs/preprocess/` and `scripts/preprocess/`.

## Evaluation dataset details

For the structure of InvBench-based evaluation data and notes on easy/hard splits and known issues, see [eval/README.md](eval/README.md).
