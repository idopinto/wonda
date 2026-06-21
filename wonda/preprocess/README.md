# Data Preparation

All datasets are on HuggingFace (see the [Wonda collection](https://huggingface.co/collections/idopinto/wonda) for the full list):

| Role | Dataset | Description |
|------|---------|-------------|
| **Evaluation** | [`idopinto/wonda-eval-benchmark-full`](https://huggingface.co/datasets/idopinto/wonda-eval-benchmark-full) | 219 programs → 362 instances after expansion (see below). |
| **Training (raw)** | [`idopinto/Wonda-Training-Dataset-Full-Raw`](https://huggingface.co/datasets/idopinto/Wonda-Training-Dataset-Full-Raw) | 3,516 programs; raw verifier-generated invariants before WONDA curation. |
| **Training (curated, full)** | [`idopinto/Wonda-Training-Dataset-Full`](https://huggingface.co/datasets/idopinto/Wonda-Training-Dataset-Full) | Full curated dataset (7.76k rows). Not SFT-ready; can be processed for other formats. |
| **Training (SFT-ready)** | Local cache only | Built on demand; see below. |

**SFT-ready caches (local):** training builds chat-formatted datasets under `data/train/sft-ready/Wonda-Training-Dataset-{Family}-V{0|1|2}/` — e.g. `Wonda-Training-Dataset-Qwen3-V0`, `Wonda-Training-Dataset-Qwen3-V2`, `Wonda-Training-Dataset-Llama3.1-V2`. Legacy pre-built Hub datasets (`idopinto/wonda-qwen-nt-sft-v*`) remain available for backward compatibility.

### Evaluation dataset (source and expansion)

For evaluation we use a collection of C programs with loop-invariant verification tasks derived from **SV-COMP** (Beyer & Strejček, 2025), as released by the authors of **InvBench** (Wei et al., 2025). We start from **219 programs**. We then apply **expansion**: each loop in a multi-loop program is turned into a separate evaluation instance (one instance = one loop to annotate). That yields **362 instances** in total.

These instances are partitioned into **Easy** and **Hard** subsets using a 15-second threshold on the UAutomizer baseline verification time (below = Easy, above = Hard). On our infrastructure (NVIDIA L40S, runlim, etc.) we obtained **Easy** *n* = 239 (66%) and **Hard** *n* = 123 (34%). If you rebuild the evaluation dataset on different hardware, baseline timings and thus the Easy/Hard split may differ. In the paper we report results on the **Hard** subset only, as those are the instances where invariant hints can provide meaningful verification speedup; you may, of course, evaluate on Easy, Hard, or both as you prefer.

> **Note:** The evaluation dataset on HuggingFace was built on an NVIDIA L40S GPU to match the hardware used for local model inference (0.6B--8B parameters; up to ~14B fits on an L40S for inference). If you plan to evaluate larger local models requiring a more powerful GPU, we recommend rebuilding the dataset from scratch on your hardware infrastructure, as baseline timings and easy/hard splits may differ across machines.

## Rebuilding the datasets from scratch

Run from the repository root:

```bash
# Build evaluation dataset (requires UAutomizer)
sbatch scripts/preprocess/build_eval_dataset.sbatch

# Build raw training dataset (verifier-generated invariants)
sbatch scripts/preprocess/build_raw_train_dataset.sbatch

# Run WONDA pipeline (normalize, simplify, verify, and grade → curated full dataset)
sbatch scripts/preprocess/wonda_pipeline.sbatch

# Build SFT-ready datasets from raw or curated output (local cache; not pushed to HF)
./scripts/preprocess/build_sft_dataset.sh dataset.version=V0   # → Wonda-Training-Dataset-Qwen3-V0
./scripts/preprocess/build_sft_dataset.sh dataset.version=V1   # → Wonda-Training-Dataset-Qwen3-V1
./scripts/preprocess/build_sft_dataset.sh dataset.version=V2   # → Wonda-Training-Dataset-Qwen3-V2
```
