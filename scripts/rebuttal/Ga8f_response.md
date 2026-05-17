We thank the reviewer for the constructive evaluation.

**Global:** Corrected Qwen3-8B-V0; added Qwen3-4B ablation. Updated [Table 4](https://imgur.com/on91lLV).

**Q1.** We mitigate rather than escape the trap: rather than distilling solver output directly, the LLM uses it as a hint to discover simplified or closed-form expressions that generalize from the program context, while the verifier acts only as a gatekeeper. The resulting invariants are forms the solver would not produce for this specific verification task due to its internal proof strategy. The ablation is consistent with this view: V0 harms 0.6B correctness (28.5% to 18.7%); V2 outperforms V0 at every scale. We acknowledge the mitigation is partial: invariants the verifier cannot confirm remain outside the training distribution. We will add this discussion to the limitations section in the revision.

**Q2.** V0 is exactly this ablation: it trains on raw, un-normalized UAutomizer invariants, isolating the effect of naive SFT without WONDA's curation. The updated [Table 4](https://imgur.com/on91lLV) presents results across three scales (R_correct):
| Scale | Base | V0 (raw) | V2 (ours) |
|---|---|---|---|
| 0.6B | 28.5 | 18.7 | 27.9 |
| 4B (new) | 22.8 | 29.3 | 44.4 |
| 8B (corrected) | 23.9 | 29.8 | 42.8 |

At 0.6B, V0 *hurts* relative to the base model. At 4B and 8B, V0 gives modest gains, but V2 adds +15.1pp and +13.0pp respectively over V0.
R_valid tells a similar story: V0 degrades validation quality (e.g., 4B drops from 99.2% to 81.3%), meaning the model learns to mimic the solver's verbose syntax and produces malformed outputs. V2 restores R_valid to 100% at both 4B and 8B. Overall, naive SFT on solver output provides modest correctness gains at best, actively hurts at small scale, and degrades output quality. The V0-to-V2 gap confirms that WONDA's curation is what drives the improvement.

**Q3.** We have conducted a cross-family experiment by fine-tuning LLaMA 3.1-8B-Instruct on the WONDA V2 dataset:

| Model | Valid. | Correct. | Speedup | VBP (s) |
|---|---|---|---|---|
| LLaMA 3.1-8B (Base) | 96.8% | 19.5% | 8.9% | 182.0 |
| **LLaMA 3.1-8B-V2 (ours)** | **100%** | **44.7%** | **17.9%** | **178.2** |

The +25.2pp gain in R_correct and the +9pp in speedup matches the magnitude observed on Qwen3-8B at the same scale.

**Limitations.** In response to the reviewer's request, we will add a detailed limitations section in the revision.

| Model | Valid. | Correct. | Speedup | VBP (s) |
|---|---|---|---|---|
| Qwen/Qwen3-Next-80B-A3B-Instruct (Base) | 97.6% | 37.4% | 20.3% | 171.3 |
| **LLaMA 3.1-8B-V2 (ours)** | **100%** | **44.7%** | **17.9%** | **178.2** |
| LLaMA 3.1-8B (Base) | 96.8% | 19.5% | 8.9% | 182.0 |
| **LLaMA 3.1-8B-V2 (ours)** | **100%** | **44.7%** | **17.9%** | **178.2** |
| Qwen3-14B (Base) | 96.7% | 39.0% | 15.4% | 182.8 |
| **Qwen3-14B-V2 (ours)** | **100%** | **41.5%** | **26.0%** | **157.4** |