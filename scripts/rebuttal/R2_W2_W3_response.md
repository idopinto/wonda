## R2-W2/W3: Benchmark Characterization and Failure Analysis

**Figures:** Figure R2 (`benchmark_characterization_baseline.png`), Figure R3a (`failure_correlation_qwen3-4b_v0.png`), Figure R3b (`failure_correlation_qwen3-4b_sft_v2.2.png`)

---

We thank the reviewer for these suggestions. We address W2 and W3 jointly since the failure analysis naturally extends the benchmark characterization.

**Benchmark structure.** The 123 Hard instances are derived from 72 unique C programs sourced from SV-COMP. Each loop entry point in a multi-loop program constitutes a separate evaluation instance, where the model must generate an invariant specifically for that program point. Figure R2 characterizes the 72 programs:

- **Program size**: median 44 LoC, mean 63.9 (range 20--428, std 88.1). The distribution is right-skewed, with 85 instances from programs under 100 LoC and a long tail of larger programs.
- **Loop nesting depth**: 60/72 programs (83%) have nesting depth 1; 10 have depth 2; 2 have depth 3 (mean 1.2).
- **Variable count**: median 5, mean 8.1 (range 2--40, std 8.2). Most programs have fewer than 10 variables, with a few outliers reaching 30--40.

**Invariant complexity.** The reviewer asks whether ground-truth invariants are predominantly linear. We note that our evaluation setting does not use ground-truth invariants: the model generates candidate invariants, and UAutomizer verifies them as correct and sufficient. Instead of characterizing invariant complexity directly, we use a source-code-level proxy: we statically check the full verification task (program body and target assertion) for nonlinear operations between variables (i.e., expressions involving `var*var`, `var/var`, or `var%var`). Programs or assertions containing such operations typically necessitate nonlinear reasoning in any correct invariant. Of the 123 instances, 49 (40%) involve nonlinear arithmetic (in the program, the assertion, or both), while the remaining 74 (60%) involve only linear operations. As shown in the rightmost panel of Figures R3a/b, the nonlinear instances are disproportionately difficult for all models.

**Failure analysis.** To address W3, we conducted a systematic failure analysis by cross-referencing per-instance outcomes with the program properties above. We distinguish three failure modes: (1) *correctness failure* (the invariant is incorrect), (2) *sufficiency failure* (the invariant is correct but insufficient for the verifier to conclude), and (3) *validation failure* (the output is malformed). Figures R3a and R3b show the breakdown for Qwen3-4B (Base) and Qwen3-4B-V2 respectively. Key findings:

- **Nonlinear arithmetic is the strongest failure predictor.** Among the 49 nonlinear instances, both Base and V2 models exhibit high correctness failure rates. However, V2 shows notable improvement on the 74 linear instances (the "No" column), where the green (success) proportion increases substantially.
- **Loop nesting depth compounds difficulty.** Depth-1 programs see the largest absolute improvement from WONDA, while depth-2 and depth-3 programs remain challenging. At depth 1, V2 roughly doubles the success count compared to Base.
- **Variable count has a threshold effect.** Programs with more than 20 variables are almost uniformly hard for both Base and V2, while the improvement from WONDA concentrates on programs with 5--15 variables.

In summary, the hardest residual instances are characterized by the combination of nonlinear arithmetic, higher nesting depth, and large variable counts. WONDA substantially improves performance on the linear, shallow, moderate-variable subset, which constitutes the majority of the benchmark. We will add this characterization and analysis to the revised manuscript.

**[Figure R2] [Figure R3a] [Figure R3b]**
