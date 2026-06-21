# Training

We fine-tune Qwen3 (non-think) and Llama 3.1 for loop invariant generation using the WONDA SFT-ready datasets.

## Released trained models

All released checkpoints are on HuggingFace in the [Wonda collection](https://huggingface.co/collections/idopinto/wonda). Published Hub repos may still use the legacy pattern `idopinto/qwen3-{size}-nt-gen-inv-sft-{version}` (e.g. `idopinto/qwen3-0.6b-nt-gen-inv-sft-v2.2-latest`). Available sizes include 0.6B, 1.7B, 4B, 8B, and 14B; versions include v0, v1, and v2.x.

**Local run naming:** training scripts derive canonical names automatically:

```
Wonda-<base>[-nt]-V{n}[-gt<N>][-LoRA]-e<N>
```

Examples: `Wonda-Qwen3-8B-nt-V2-gt2-LoRA-e2`, `Wonda-Llama3.1-8B-Instruct-V2-gt2-e3`. Checkpoints land under `trained_models/<name>/` and Hub pushes use `idopinto/<name>` when `push_to_hub: true`.

**Paper full-FT checkpoints** (Hub + local):

| Model | Hub | Local |
|-------|-----|-------|
| Llama-3.1-8B-Instruct | `idopinto/Wonda-Llama3.1-8B-Instruct-V2-gt2-e3` | `trained_models/Wonda-Llama3.1-8B-Instruct-V2-gt2-e3` |
| Qwen3-14B | `idopinto/Wonda-Qwen3-14B-nt-V2-gt2-e3` | `trained_models/Wonda-Qwen3-14B-nt-V2-gt2-e3` |

## Training your own model

Run from the repository root. Pass a **config name** and optional Hydra overrides.

**Qwen3 (LoRA):**

```bash
sbatch --job-name=<job_name> scripts/train/train_qwen.sbatch <config_name> [hydra overrides...]
```

**Config names:** `qwen3_0.6b` | `qwen3_1.7b` | `qwen3_4b` | `qwen3_8b` | `qwen3_14b`

**Llama 3.1 (full fine-tune by default):**

```bash
sbatch --job-name=<job_name> scripts/train/train_llama.sbatch <config_name> [hydra overrides...]
```

**Config names:** `llama3_8b` | `llama3_8b_full`

**Commands (model size × version; v2 uses min_grade=2):**

```bash
# 0.6B
sbatch --job-name=train_qwen_0.6b_v0 scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v0
sbatch --job-name=train_qwen_0.6b_v1 scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v1
sbatch --job-name=train_qwen_0.6b_v2 scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v2 dataset.min_grade=2
sbatch --job-name=train_qwen_0.6b_v2_test scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v2 dataset.min_grade=2 test_mode=true
# 1.7B
sbatch --job-name=train_qwen_1.7b_v0 scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v0
sbatch --job-name=train_qwen_1.7b_v1 scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v1
sbatch --job-name=train_qwen_1.7b_v2 scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v2 dataset.min_grade=2
sbatch --job-name=train_qwen_1.7b_v2_test scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v2 dataset.min_grade=2 test_mode=true
# 4B
sbatch --job-name=train_qwen_4b_v0 scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v0
sbatch --job-name=train_qwen_4b_v1 scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v1
sbatch --job-name=train_qwen_4b_v2 scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v2 dataset.min_grade=2
sbatch --job-name=train_qwen_4b_v2_test scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v2 dataset.min_grade=2 test_mode=true
# 8B
sbatch --job-name=train_qwen_8b_v0 scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v0
sbatch --job-name=train_qwen_8b_v1 scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v1
sbatch --job-name=train_qwen_8b_v2 scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v2 dataset.min_grade=2
sbatch --job-name=train_qwen_8b_v2_test scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v2 dataset.min_grade=2 test_mode=true
# 14B
sbatch --job-name=train_qwen_14b_v0 scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v0
sbatch --job-name=train_qwen_14b_v1 scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v1
sbatch --job-name=train_qwen_14b_v2 scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v2 dataset.min_grade=2
sbatch --job-name=train_qwen_14b_v2_test scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v2 dataset.min_grade=2 test_mode=true
```

**Useful overrides:**

| Override | Description |
|----------|-------------|
| `dataset.version=v0\|v1\|v2` | SFT dataset version (default in config: v2) |
| `dataset.min_grade=1\|2\|3` | Min quality grade for v2 |
| `dataset.limit=N` | Limit samples (-1 = all) |
| `sft.num_train_epochs=N` | Training epochs |
| `wandb.use_wandb=true\|false` | Enable/disable Weights & Biases |
| `test_mode=true` | Short test run |

Configs live in `configs/train/`. Key training parameters (defaults in config):

| Parameter | Typical | Description |
|-----------|----------|-------------|
| `sft.num_train_epochs` | 2–3 | Number of training epochs |
| `sft.learning_rate` | 1e-4 | Learning rate |
| `lora.r` | 16 | LoRA rank |
| `lora.lora_alpha` | 16 | LoRA alpha |
| `lora.target_modules` | all-linear | Modules to adapt |

Output is written under `trained_models/Wonda-<base>[-nt]-V<n>[-gt<N>][-LoRA]-e<N>/` and can be pushed to the Hub when `push_to_hub: true` in the config.

**SFT cache:** datasets are built on demand into `data/train/sft-ready/Wonda-Training-Dataset-{Family}-V{0|1|2}/text.json` (e.g. `Wonda-Training-Dataset-Qwen3-V2`, `Wonda-Training-Dataset-Llama3.1-V2`). V2 caches include all grade ≥ 1 rows; stricter `min_grade` filters apply at load time.
