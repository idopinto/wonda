# How Many Trials for Your Research Paper?

## Quick Answer

**Minimum for publication**: **3-5 trials**  
**Recommended for strong paper**: **5-10 trials**  
**Ideal for top-tier venues**: **10+ trials**

## Detailed Recommendations by Scenario

### 1. **Minimum Acceptable (3 trials)**
**When to use**:
- Preliminary results
- Compute-limited scenarios
- When variance is known to be low
- Supplementary material

**What you can report**:
- Mean ± std
- Basic confidence intervals (less reliable)
- Detect only large differences (>5-10%)

**Limitations**:
- Wide confidence intervals
- Limited statistical power
- Reviewers may question reliability

### 2. **Standard Practice (5 trials)** ⭐ **RECOMMENDED**
**When to use**:
- Main results in paper
- Most conference papers
- Good balance of rigor vs. compute

**What you can report**:
- Mean ± std (standard in ML papers)
- 95% confidence intervals
- Detect moderate differences (3-5%)
- Statistical significance testing

**Why 5 is good**:
- Standard in ML/NLP research
- Reviewers expect this
- Good statistical power
- Reasonable compute cost

### 3. **Strong Results (10 trials)**
**When to use**:
- Top-tier venues (NeurIPS, ICML, ICLR)
- When comparing multiple models
- When variance is high
- When small differences matter

**What you can report**:
- Robust mean ± std
- Narrow confidence intervals
- Detect small differences (1-2%)
- Strong statistical significance
- Effect size calculations

### 4. **Very Rigorous (20+ trials)**
**When to use**:
- High-stakes comparisons
- When publishing variance analysis
- When reviewers specifically request it
- When establishing new benchmarks

## What Reviewers Expect

### Standard Practice in ML/NLP:
- **3-5 runs**: Acceptable, but may get questioned
- **5 runs**: Standard, reviewers are satisfied
- **10+ runs**: Impressive, shows rigor
- **1 run**: **Unacceptable** (will get rejected or major revision)

### What to Report:
```
✅ Good: "We report mean ± std over 5 runs: 67.3% ± 1.2%"
✅ Better: "We report mean ± std over 5 runs with 95% CI: 67.3% ± 1.2% [65.8%, 68.8%]"
❌ Bad: "We achieved 67.3% accuracy" (single run)
```

## Statistical Power Analysis

### Minimum Detectable Effect (MDE)
With different numbers of trials:

| Trials | MDE (95% power) | Use Case |
|--------|----------------|----------|
| 3 | ~8-10% | Large differences only |
| 5 | ~5-7% | Moderate differences |
| 10 | ~3-4% | Small differences |
| 20 | ~2-3% | Very small differences |

**MDE = Minimum difference you can reliably detect**

### Example:
- **5 trials**: Can detect if Model A is 5% better than Model B
- **10 trials**: Can detect if Model A is 3% better than Model B

## Practical Decision Framework

### Step 1: Check Your Variance
Run 3 trials first, check standard deviation:
- **σ < 1%**: Very consistent → 3-5 trials sufficient
- **σ = 1-3%**: Moderate variance → 5-10 trials recommended
- **σ > 3%**: High variance → 10+ trials needed

### Step 2: Consider Your Claims
- **"Model A outperforms Model B"**: Need 5-10 trials
- **"Model A is significantly better"**: Need 10+ trials with statistical tests
- **"Model A achieves X% accuracy"**: Need 5 trials minimum

### Step 3: Consider Your Venue
- **Workshop/ArXiv**: 3-5 trials acceptable
- **Conference (AAAI, EMNLP, etc.)**: 5 trials standard
- **Top-tier (NeurIPS, ICML, ICLR)**: 5-10 trials expected
- **Journal**: 10+ trials often expected

## Cost-Benefit Analysis

### Compute Time Estimate:
Assuming each trial takes ~2-4 hours for full dataset:

| Trials | Total Time | Cost |
|--------|------------|------|
| 3 | 6-12 hours | Low |
| 5 | 10-20 hours | Medium |
| 10 | 20-40 hours | High |
| 20 | 40-80 hours | Very High |

### Recommendation:
- **Start with 5 trials** (standard practice)
- **If variance is high or results are close**: Run 5 more (total 10)
- **If compute is limited**: Minimum 3, but acknowledge limitations

## What to Report in Your Paper

### Essential Information:
1. **Number of trials**: "We report results over 5 independent runs"
2. **Mean ± std**: "67.3% ± 1.2%"
3. **Confidence intervals**: "[65.8%, 68.8%]" (optional but recommended)

### Example Table Format:
```
Model          | Correctness Rate | Speedup Rate
---------------|------------------|-------------
Baseline       | 65.0% ± 0.5%    | 1.00 ± 0.00
Our Model      | 67.3% ± 1.2%    | 1.15 ± 0.08
               | [65.8%, 68.8%]  | [1.09, 1.21]
```

### In Text:
> "We evaluate each model over 5 independent runs with different random seeds. 
> We report mean ± standard deviation, with 95% confidence intervals in brackets. 
> Our model achieves 67.3% ± 1.2% [65.8%, 68.8%] correctness rate, compared to 
> 65.0% ± 0.5% [64.3%, 65.7%] for the baseline."

## Special Considerations

### 1. **Comparing Multiple Models**
If comparing 3+ models:
- **5 trials per model**: Standard
- **10 trials per model**: Better for fair comparison
- Use statistical tests (t-test, ANOVA) to show significance

### 2. **High Variance Models**
If your model has high variance (σ > 3%):
- **Minimum 10 trials** to get reliable estimates
- Consider reporting median instead of mean
- Analyze variance as a feature (some models are more reliable)

### 3. **Limited Compute**
If compute is limited:
- **Minimum 3 trials** (acknowledge in paper)
- Focus on most important comparisons
- Use statistical tests to maximize information from limited data

## Red Flags for Reviewers

❌ **"We ran a single evaluation"** → Major revision or rejection  
❌ **"We report results from one run"** → Unacceptable  
❌ **No mention of multiple runs** → Assumed single run  
✅ **"We report mean ± std over 5 runs"** → Standard  
✅ **"We report mean ± std over 10 runs with 95% CI"** → Excellent

## Final Recommendation

### For Your Paper:
1. **Start with 5 trials** (industry standard)
2. **Check variance**: If σ > 2%, consider 10 trials
3. **For main results**: Use 5-10 trials
4. **For supplementary**: 3 trials acceptable
5. **Always report**: Mean ± std, number of trials

### Command:
```bash
# Standard (5 trials)
sbatch scripts/eval/evaluate_g.sbatch \
  models=qwen_no_think_config \
  models.trials.num_trials=5 \
  dataset.limit=-1

# Strong (10 trials)
sbatch scripts/eval/evaluate_g.sbatch \
  models=qwen_no_think_config \
  models.trials.num_trials=10 \
  dataset.limit=-1
```

## References

1. **ML Best Practices**:
   - "On the Opportunities and Risks of Foundation Models" (Bommasani et al.)
   - "Evaluating Large Language Models" (Chang et al.)

2. **Statistical Guidelines**:
   - Cohen (1988): "Statistical Power Analysis"
   - Wasserstein & Lazar (2016): "The ASA Statement on p-Values"

3. **Reviewer Expectations**:
   - NeurIPS/ICML/ICLR guidelines
   - AAAI/EMNLP/ACL submission guidelines
