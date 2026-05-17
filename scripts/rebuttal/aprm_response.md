We thank the reviewer for the thorough and constructive feedback.

**Global:** Corrected Qwen3-8B-V0; added Qwen3-4B ablation. Updated [Table 4](https://imgur.com/a/wonda-rebuttal-figures-Kswj5Tf#on91lLV).

**W1.** We agree that the gap to larger open-weight and frontier models is real, and that solver-only baselines should be easier to read from the main tables. WONDA targets data curation; we therefore emphasize Base-to-V2 improvement (e.g., 4B VBP 185.6s to 165.5s) as the primary effect of our pipeline. Including inference (VBP_E2E), 4B (165.7s) surpasses GPT-OSS-120B (167.6s) and nearly matches GPT-5.2 (163.4s), so small models can still be competitive on end-to-end verification time. On FLOP-matched comparison to the solver: in our setting, SLM inference is only 1-3s wall-clock while solver runs average 193s (median 112s), so total cost is solver-dominated; VBP and VBP_E2E are nearly identical across our model sizes for that reason, and larger LLMs would only widen the inference gap relative to SLMs. We will add solver-only mean VBP (193s, as in Appendix A) explicitly to Table 3 in the revision.

**W2.** We addressed a closely related concern from Reviewer Ga8f (Q1); we summarize the key argument here. WONDA mitigates rather than escapes the trap: the LLM uses solver output as a *hint* to discover simplified or closed-form expressions that generalize from program context, while the verifier acts only as a gatekeeper. The resulting invariants are forms the solver would not produce on its own due to its internal CEGAR-based proof strategy. The ablation is consistent with this view: V0 harms 0.6B correctness (28.5% to 18.7%); V2 outperforms V0 at every scale. We acknowledge the mitigation is partial and will add this discussion to the limitations section.

**W3.** Agreed, and we discuss RL explicitly in our conclusion. We view SFT as a necessary foundation: RL from a weak policy (23% correctness) risks reward hacking and sample inefficiency. WONDA's curated SFT (44% correctness) provides a strong initialization, and our graded quality signal (correctness, sufficiency, speedup) defines a natural reward structure for RL. We position verifier-based RL as a next step that WONDA enables rather than replaces.

**W4 / KQ2.** First, the 8B using LoRA was not mentioned in the main text and will be clarified. Second, the updated [Table 4](https://imgur.com/a/wonda-rebuttal-figures-Kswj5Tf#on91lLV) now spans three model scales (0.6B, 4B, 8B [LoRA]), with V2 improving over V0 and Base in every configuration. The 4B full-FT ablation further confirms the gains are robust. We expect full FT at 8B or larger to follow the same trend; see also our response to Reviewer Ga8f (**Q3**), where full fine-tuning of LLaMA 3.1-8B on the WONDA V2 dataset shows large Base-to-V2 gains in another model family (table there).

**KQ3:** V2 total token count is 3,767,698 (7,284 samples, mean: 517.3, min: 376, max: 1,023); will be stated explicitly in the revision.

**W5 / KQ1.** V0 models are our closest proxy to Wei et al.: both train on raw solver-generated invariants from the same InvBench program set. Exact replication was not possible as their hyperparameters and prompt template are not public. There is also a task-level difference: they jointly select loop location and generate the invariant, whereas we specify the location explicitly. Despite this, V0 should be interpreted as a reasonable proxy rather than an exact reproduction. The V0-to-V2 improvement at every scale in [Table 4](https://imgur.com/a/wonda-rebuttal-figures-Kswj5Tf#on91lLV) demonstrates WONDA's benefit. We will make this connection explicit in the revision.

**KQ4.** Not yet evaluated. Our intuition is that the full grade >= 2 set (7,284 samples) outperforms grade 3 alone (2,767 samples) due to data volume. Curriculum training (starting with grade 2, progressively filtering to grade 3) is a promising direction we will discuss in the revision.

**Nits.** `\citep` to `\citet` will be fixed throughout.

**Limitations.** We agree LLM inference cost should be discussed more explicitly relative to solver time in limitations.
