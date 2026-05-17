# Rebuttal Experiments

Commands for running baseline timing and evaluation on **Lemur** and **Code2Inv** benchmarks.

All commands are run from the project root.

---

## 1. Baseline Timing (Median Verification Time)

Measures UAutomizer verification time on raw programs (no invariants) to establish per-program baselines.

### Lemur (47 programs)

```bash
# UAutomizer v25 (default 3 trials, 600s timeout)
sbatch scripts/rebuttal/run_baseline_timing_lemur.sbatch --uautomizer-version 25

# UAutomizer v23
sbatch scripts/rebuttal/run_baseline_timing_lemur.sbatch --uautomizer-version 23

# Custom trials/timeout
sbatch scripts/rebuttal/run_baseline_timing_lemur.sbatch --uautomizer-version 25 --num-trials 5 --timeout 900
```

Output: `data/eval/lemur_sv_comp/baseline_uauto<version>/baseline_dataset.json`

### Code2Inv (133 programs)

```bash
# UAutomizer v25 (default 3 trials, 600s timeout)
sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch --uautomizer-version 25

# UAutomizer v23
sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch --uautomizer-version 23

# Custom trials/timeout
sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch --uautomizer-version 25 --num-trials 5 --timeout 900
```

Output: `data/eval/code2inv/baseline_uauto<version>/baseline_dataset.json`

---

## 2. Model Evaluation

Uses `scripts/eval/run_multi_eval.sbatch` with Hydra overrides.

### Lemur Evaluation

All Lemur commands use `dataset.baseline_json` to load actual per-program UAutomizer v25 baseline timings.

```bash
# --- Qwen3-4B base model (no-think) ---
sbatch --job-name=me_qwen4b_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-4B fine-tuned v2.2 ---
sbatch --job-name=me_qwen4b_v22_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config \
  models.eval_ft_model=true models.ft_model.sft_version="v2.2" \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-8B full fine-tuned v2.2 ---
sbatch --job-name=me_qwen8b_full_v22_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_8b_full_nt_config \
  models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=false \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-14B full fine-tuned v2.2 ---
sbatch --job-name=me_qwen14b_full_v22_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_full_nt_config \
  models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=false \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-0.6B base model ---
sbatch --job-name=me_qwen06_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-1.7B base model ---
sbatch --job-name=me_qwen17_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-8B base model ---
sbatch --job-name=me_qwen8b_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- Qwen3-14B base model ---
sbatch --job-name=me_qwen14b_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- GPT-5.2 ---
sbatch --job-name=me_gpt5_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=gpt_5.2_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"

# --- GPT-OSS-120B ---
sbatch --job-name=me_gptoss120b_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=gpt_oss_120b_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-lemur"
```

### Code2Inv Evaluation

All Code2Inv commands use `dataset.baseline_json` to load actual per-program UAutomizer v25 baseline timings.

```bash
# --- Qwen3-4B base model (no-think) ---
sbatch --job-name=me_qwen4b_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-4B fine-tuned v2.2 ---
sbatch --job-name=me_qwen4b_v22_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config \
  models.eval_ft_model=true models.ft_model.sft_version="v2.2" \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-8B full fine-tuned v2.2 ---
sbatch --job-name=me_qwen8b_full_v22_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_8b_full_nt_config \
  models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=false \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-14B full fine-tuned v2.2 ---
sbatch --job-name=me_qwen14b_full_v22_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_full_nt_config \
  models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=false \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-0.6B base model ---
sbatch --job-name=me_qwen06_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-1.7B base model ---
sbatch --job-name=me_qwen17_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-8B base model ---
sbatch --job-name=me_qwen8b_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- Qwen3-14B base model ---
sbatch --job-name=me_qwen14b_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- GPT-5.2 ---
sbatch --job-name=me_gpt5_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=gpt_5.2_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"

# --- GPT-OSS-120B ---
sbatch --job-name=me_gptoss120b_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=gpt_oss_120b_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-code2inv"
```

---

## 3. Llama 3.1 8B Baseline Evaluation

Uses `meta-llama/Llama-3.1-8B-Instruct` as an open-source baseline on the **wonda hard split**.
Config: `configs/eval/models/llama3_8b_config.yaml` (Meta-recommended sampling: `temperature=0.6`, `top_p=0.9`).

### Quick test (local, no Weave, 5 samples)

```bash
uv run -m wonda.eval.run_multi_eval \
  multi_run.num_runs=1 \
  dataset.limit=5 \
  models=llama3_8b_config \
  dataset.split=hard \
  weave.skip_weave=true
```

### Full evaluation — wonda hard split (SLURM)

```bash
sbatch --job-name=me_llama38b scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 \
  dataset.limit=-1 \
  models=llama3_8b_config \
  dataset.split=hard \
  weave.project_name="eval-wonda-llama"
```

### Lemur benchmark

```bash
sbatch --job-name=me_llama38b_lemur scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=llama3_8b_config \
  dataset.local_dir="data/eval/lemur_sv_comp/orig_programs" \
  dataset.baseline_json="data/eval/lemur_sv_comp/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-llama"
```

### Code2Inv benchmark

```bash
sbatch --job-name=me_llama38b_c2i scripts/eval/run_multi_eval.sbatch \
  multi_run.num_runs=3 dataset.limit=-1 models=llama3_8b_config \
  dataset.local_dir="data/eval/code2inv/orig_programs" \
  dataset.baseline_json="data/eval/code2inv/baseline_uauto25/baseline_dataset.json" \
  weave.project_name="eval-wonda-llama"
```

---

## 5. Output Locations

| Artifact | Path |
|----------|------|
| Baseline timing (Lemur) | `data/eval/lemur_sv_comp/baseline_uauto<ver>/` |
| Baseline timing (Code2Inv) | `data/eval/code2inv/baseline_uauto<ver>/` |
| Eval results (Lemur) | `eval_experiments/eval-wonda-lemur/<experiment>/` |
| Eval results (Code2Inv) | `eval_experiments/eval-wonda-code2inv/<experiment>/` |
| Eval results (Llama) | `eval_experiments/eval-wonda-llama/<experiment>/` |
| Aggregated results | `<experiment_dir>/aggregated_results.txt` |

## 6. Available Model Configs

| Config | Model |
|--------|-------|
| `qwen3_0.6b_nt_config` | Qwen3-0.6B (no-think) |
| `qwen3_0.6b_t_config` | Qwen3-0.6B (think) |
| `qwen3_1.7b_nt_config` | Qwen3-1.7B (no-think) |
| `qwen3_4b_nt_config` | Qwen3-4B (no-think) |
| `qwen3_8b_nt_config` | Qwen3-8B (no-think) |
| `qwen3_8b_t_config` | Qwen3-8B (think) |
| `qwen3_8b_full_nt_config` | Qwen3-8B full fine-tune (no-think) |
| `qwen3_14b_nt_config` | Qwen3-14B (no-think) |
| `qwen3_14b_t_config` | Qwen3-14B (think) |
| `qwen3_14b_full_nt_config` | Qwen3-14B full fine-tune (no-think) |
| `qwen3_80b_nt_config` | Qwen3-80B (no-think) |
| `llama3_8b_config` | Llama 3.1-8B-Instruct |
| `gpt_5.2_config` | GPT-5.2 |
| `gpt_oss_20b_config` | GPT OSS 20B |
| `gpt_oss_120b_config` | GPT OSS 120B |

## 7. Monitoring

```bash
# Check SLURM job status
squeue -u $USER

# Follow job output
tail -f slurm/<job-name>_<job-id>.out

# Re-aggregate results from an experiment directory
uv run -m wonda.eval.aggregate_results <experiment_dir>
```
