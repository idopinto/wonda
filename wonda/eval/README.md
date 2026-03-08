# Evaluation

In the paper, we evaluate loop invariant generation on 123 instances. Each run: the model proposes invariant per loop; the decision procedure checks correctness and sufficiency; we aggregate over multiple trials for confidence intervals.


## Evaluation process

1. **Dataset:** Evaluation benchmark from [`idopinto/wonda-eval-benchmark-full`](https://huggingface.co/datasets/idopinto/wonda-eval-benchmark-full) (see [wonda/preprocess/README.md](../preprocess/README.md)).
2. **Model:** For each benchmark item, the model is given the C program with loop markers and outputs a single candidate invariant per designated loop marker.
3. **Verification:** UAutomizer verifies each candidate (correctness); we also measure whether the invariant is *useful* (speeds up or enables verification). runlim enforces time and memory limits.
4. **Aggregation:** We run multiple independent trials (e.g. `--num_runs=3`) and report means and confidence intervals (e.g. invariant correctness rate, Virtual Best Performance on the verification task).

The metrics (validation, correctness, usefulness, VBS, VBS end-to-end, speedup, speedup end-to-end) are implemented in [inv_gen_scorer.py](inv_gen_scorer.py); the class docstring there defines each metric.

## Models evaluated in the paper

- **Qwen3 (SLMs):** 0.6B, 4B, 8B — each as **base (no-think)** and **fine-tuned** on WONDA data (v2.2). Fine-tuned 8B and 14B use LoRA.
- **Baselines:** GPT-5.2, GPT-OSS-120B (via open-router api)
- **Ablation Study** 0.6B, 8B - on WONDA data (v0, v1, v2.2)
Configs for these are under `configs/eval/models/` (e.g. `qwen3_0.6b_nt_config`, `qwen3_4b_nt_config`, `gpt_5.2_config`).
See the paper for the results tables.

## Running evaluation

From the repository root. Multi-run evaluation with statistical aggregation:

```bash
sbatch --job-name=<name> scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 [HYDRA_OVERRIDES]
```

**Examples:**

```bash
# Base Qwen3-0.6B (no-think)
sbatch --job-name=me_qwen06 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config

# Fine-tuned Qwen3-0.6B v2.2
sbatch --job-name=me_qwen06_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"

# Fine-tuned Qwen3-8B v2.2 (LoRA: add is_lora=true)
sbatch --job-name=me_qwen8b_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true

# Baseline GPT-5.2
sbatch --job-name=me_gpt5 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=gpt_5.2_config
```

**Useful overrides:** `multi_run.num_runs=N`, `dataset.limit=N` (use `-1` for full benchmark), `models=<config_name>`, `models.eval_ft_model=true`, `models.ft_model.sft_version="v2.2"`, `models.ft_model.is_lora=true` (for 8B/14B fine-tuned). An exhaustive list of commands per model is in the comments of [scripts/eval/run_multi_eval.sbatch](../../scripts/eval/run_multi_eval.sbatch).

Multi-run entrypoint: `wonda.eval.run_multi_eval` (Hydra config: `configs/eval/eval_config.yaml`).  
Single-run worker: `wonda.eval.evaluate` (receives a resolved per-run YAML via `--config`).

Results are written under `eval_experiments/<weave_project_name>/<experiment_name>/` and include:

- `config.json` (run metadata and overrides)
- `evaluate_config_run_<k>.yaml` (resolved config per run)
- `summary_run_<k>.json` (raw summary per run)
- `aggregated_results.json` and `aggregated_results.txt` (mean + confidence intervals)
