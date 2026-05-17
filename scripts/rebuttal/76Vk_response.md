We thank the reviewer for the feedback.

**Global:** Corrected Qwen3-8B-V0; added Qwen3-4B ablation. Updated  [Table 4](https://imgur.com/on91lLV).

**KQ1.** We performed a post-hoc timeout sweep (T∈{15,30,…,600}s), shown in [Fig. R4](https://imgur.com/XBtDFR8). Correctness rises smoothly from T=15s with no threshold artifacts, and speedup plateaus by 120-180s. Fine-tuning yields 1.5-2x improvement at every threshold. VBP improves by 31s relative to the 193s baseline (162s fine-tuned), confirming the gains are not an artifact of the timeout setting.

**W1.** We agree the discussion should balance these dimensions more explicitly. WONDA improves correctness substantially: Qwen3-4B R_correct nearly doubles (22.8%→44.4%), closing 60% of the gap to GPT-OSS-120B (58.0%), a 30x larger model. However, correctness alone is insufficient; invariants must also be sufficient and compact enough to accelerate verification. WONDA curates for both correctness *and* compactness, steering the model toward invariants that effectively guide the solver.
[Fig. R1](https://imgur.com/wR5WzHs) illustrates this trade-off. Qwen3-4B-V2 matches GPT-OSS-120B on VBP (165.5s vs. 165.8s) despite lower correctness. Base models are dominated on both axes by V2, confirming that WONDA drives the gains. We will expand this in the revision.

**W2 / W3.**
*Benchmark Characterization ([Fig. R2](https://imgur.com/e0GFSIm)).* The 123 Hard instances derive from 72 SV-COMP programs, with one instance per loop entry.
*Invariant complexity.* Our evaluation does not use ground-truth invariants; UAutomizer checks model output directly. We therefore cannot characterize invariant complexity directly, and instead use nonlinear arithmetic as a program-level difficulty proxy (49 nonlinear vs. 74 linear instances).
*Failure analysis ([Fig. R3a](https://imgur.com/lntka3M), [R3b](https://imgur.com/rjsXMTc)).* We classify each instance into: success, sufficiency failure, correctness failure, and validation failure. Correctness failure dominates for both V0 and V2. V2 improves overall success from 18.7% to 33.3%, with validation reaching 100% (vs. 87% for V0). The improvement is broad: nonlinear instances see the largest failure-rate reduction (85.7%→63.3%), while linear instances also improve (78.4%→68.9%). Larger programs (LOC≥200: 80%→30%) and medium variable counts (10-20 vars: 85.7%→42.9%) benefit substantially and nesting depth 2-3 also improves.

**W4 / W5.** We agree that evaluating counterexample-guided refinement is compelling, and that one-shot generation alone does not fully test whether the model has learned verification logic. We believe the concerns are complementary: a better one-shot generator provides a better initialization for any CEGIS loop. We address both through two generalization experiments.
*Easy split.* We evaluate on the easy split (147 programs, 239 instances), held out from both training and main evaluation. Since UAutomizer already solves these within seconds, we focus on correctness and sufficiency rather than speedup. R_sufficient denotes the fraction of generated invariants that are both correct and lead to conclusive verdict.

| Model | R_valid (%) | R_correct (%) | R_sufficient (%) |
|---|---|---|---|
| Qwen3-4B | 100.0±0.0 | 42.3±1.1 | 42.3±1.1 |
| Qwen3-4B-V0 | 85.1±2.5 | 26.9±3.8 | 26.8±3.7 |
| Qwen3-4B-V1 | 99.0±0.6 | 34.5±1.5 | 34.2±1.7 |
| Qwen3-4B-V2 (Ours) | 100.0±0.0 | 50.8±0.9 | 50.4±0.9 |

V2 improves R_correct by +8.5pp and R_sufficient by +8.1pp over the base model while maintaining perfect validation, confirming that the training signal transfers to unseen programs.

*Code2inv.*

| Model | R_correct (%) |
|---|---|
| Qwen3-0.6B | 22.8±1.7 |
| Qwen3-0.6B-V2 | 53.5±2.1 |
| Qwen3-4B | 66.8±0.7 |
| Qwen3-4B-V2 | 64.4±1.9 |
| Qwen3-8B | 70.2±0.9 |
| Qwen3-8B-V2 (LoRA)| 61.4±3.0 |
| Llama3.1-8B-Instruct | 49.1±2.48 |
| Llama3.1-8B-Instruct-V2 | 66.69±3.8 | 

*Lemur.* We evaluated base and WONDA models inside **Lemur** (Wu et al., ICLR 2024), an LLM-based verifier that operates in a generate-verify-refine loop, on its 47 SV-COMP tasks that UAutomizer *cannot* solve within 10 minutes. Since WONDA trains exclusively on programs UAutomizer *can* solve, this is a genuinely out-of-distribution test. The models were used as-is, with no Lemur-specific fine-tuning.

| Model | Solved (%) |
|---|---|
| Qwen3-4B | 12.8 |
| Qwen3-4B-V2| 17.0 |
| Qwen3-14B | 14.9 |
| Qwen3-14B-V2| 17.0 |

The improvement is moderate, but consistent across both model scales, supporting our claim.
Designing a full counterexample-guided LLM loop remains an open challenge. It is still unclear how to distill SMT solver feedback into a form that provides meaningful guidance to the LLM; notably, Lemur itself does not feed explicit solver counterexamples back to the model, relying instead on generic error signals. We believe that designing such a mechanism and studying its interaction with training data quality is a promising direction for future work. Similarly, extending evaluation to other formalisms (e.g., SyGuS-INV or other programming languages) is a natural next step.
**W6.** We will clarify definitions and add a notation table.