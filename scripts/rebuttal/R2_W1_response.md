## R2-W1: Correctness-Speed Trade-off and Evaluation Framing

**Figures:** Figure R1 (`pareto_combined.png`)

---

We thank the reviewer for this insightful point and the opportunity to clarify.

First, regarding the emphasis on speedup: we want to note that our evaluation reports correctness (R_correct, R_valid), speedup (R_speedup, S̄>1), and end-to-end time (VBP, VBP_E2E) equally across all tables. We do not treat speedup as the dominant signal. However, we agree the discussion section could better balance these dimensions, and we will revise it accordingly.

Second, we want to highlight that **WONDA improves correctness substantially**. Comparing V2 to Base at the same scale (the appropriate measure of curation quality): Qwen3-4B R_correct nearly doubles (22.8% → 44.4%), R_valid reaches 100%, and R_speedup more than doubles (11.1% → 24.7%). The remaining gap to GPT-5.2 (72.4%) reflects the difference in model capacity: Qwen3-4B-V2 nonetheless closes roughly half the distance to GPT-OSS-120B (58.0%), a model 30x its size. We believe this demonstrates that the curation pipeline is effective on both axes, while acknowledging that no dataset can fully compensate for a capacity gap of this magnitude.

An important distinction is that GPT-5.2 and GPT-OSS-120B are **reasoning models** that employ extended chain-of-thought, whereas our Qwen3-V2 models are standard (non-reasoning) models. This has direct practical implications: reasoning models incur significantly higher inference latency and cost per query. In verification frameworks that involve repeated LLM invocations (Wu et al., 2023), this overhead compounds substantially. To illustrate this, we include Figure R1, which plots R_correct against VBP (a) and VBP_E2E (b). The arrows show WONDA moving both 4B and 8B models toward higher correctness and faster verification simultaneously. Panel (b) is particularly informative: once inference latency is included, Qwen3-4B-V2 (165.7s) surpasses GPT-OSS-120B (167.6s) and nearly matches GPT-5.2 (163.4s), despite being a non-reasoning model with a fraction of the parameters and inference cost. This suggests that compact, solver-friendly invariants from a fast non-reasoning SLM can be more cost-effective in practice than longer reasoning traces from frontier models. We will add this figure and discussion to the revised manuscript.

**[Figure R1]**
