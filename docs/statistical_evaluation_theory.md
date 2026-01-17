# Statistical Theory Behind Multi-Trial Evaluation

## Problem: Why Multiple Trials?

**LLMs are non-deterministic** - even with the same input, they can produce different outputs due to:
- Random sampling (temperature > 0)
- Stochastic decoding
- Model variance

**Single evaluation = single sample** → You don't know if your result is:
- Typical performance (representative)
- Lucky/unlucky outlier
- Due to random chance

## Solution: Multiple Trials = Statistical Sampling

Running N trials gives you **N independent samples** of the model's performance distribution.

## Core Statistical Concepts

### 1. **Sample Mean (μ̂)**
The average across trials:
```
μ̂ = (x₁ + x₂ + ... + xₙ) / n
```
- **What it tells you**: Expected/typical performance
- **Example**: If correctness_rate across 5 trials is [0.65, 0.68, 0.66, 0.67, 0.69], mean = 0.67

### 2. **Standard Deviation (σ)**
Measure of variability:
```
σ = √[Σ(xᵢ - μ̂)² / (n-1)]
```
- **What it tells you**: How much results vary
- **Low σ**: Consistent performance (reliable)
- **High σ**: Unpredictable performance (unreliable)
- **Example**: σ = 0.01 → very consistent; σ = 0.10 → high variance

### 3. **Standard Error (SE)**
Uncertainty in the mean estimate:
```
SE = σ / √n
```
- **What it tells you**: How confident you can be in the mean
- **Decreases with more trials**: More trials → more confidence
- **Example**: With 5 trials, SE = σ/√5 ≈ 0.45σ

### 4. **Confidence Interval (CI)**
Range where the true mean likely lies:
```
CI = μ̂ ± z × SE
```
Where z = 1.96 for 95% confidence (from normal distribution)

**Interpretation**: 
- "We are 95% confident the true mean lies in [CI_lower, CI_upper]"
- **Narrow CI**: High confidence in the estimate
- **Wide CI**: High uncertainty

## Theoretical Foundation

### Central Limit Theorem (CLT)
**Key insight**: As sample size increases, the distribution of sample means approaches a normal distribution, regardless of the underlying distribution.

**Implications**:
- Even if individual trial results are non-normal, the **mean across trials** is approximately normal
- This justifies using z-scores/t-scores for confidence intervals
- Works well with n ≥ 5-10 trials

### Sampling Theory
**Each trial = independent sample** from the model's performance distribution.

**Assumptions**:
1. **Independence**: Each trial is independent (same dataset, different model outputs)
2. **Identical distribution**: Same model, same evaluation setup
3. **Random sampling**: Model's stochasticity provides randomness

## Practical Interpretation

### Example Output:
```
Correctness Rate: 0.6700 ± 0.0150 [0.6550, 0.6850]
```

**Reading this**:
- **0.6700**: Mean correctness rate across trials
- **± 0.0150**: Standard deviation (results vary by ~1.5%)
- **[0.6550, 0.6850]**: 95% confidence interval
  - We're 95% confident the true correctness rate is between 65.5% and 68.5%

### Statistical Significance

**Comparing two models**:
- If Model A's CI: [0.65, 0.68]
- And Model B's CI: [0.60, 0.63]
- **No overlap** → Statistically significant difference (p < 0.05)

**If CIs overlap**:
- Cannot confidently say which is better
- Need more trials to reduce uncertainty

## Why This Matters for LLM Evaluation

### 1. **Reliability Assessment**
- **Low variance** (small σ): Model is consistent → results are trustworthy
- **High variance** (large σ): Model is unpredictable → single trial unreliable

### 2. **Confidence in Results**
- **Narrow CI**: High confidence in reported metrics
- **Wide CI**: Results are uncertain, need more trials

### 3. **Fair Comparison**
- Comparing models requires accounting for variance
- A model with mean=0.65 ± 0.10 is less reliable than one with mean=0.64 ± 0.01

### 4. **Decision Making**
- **CI doesn't include threshold**: Confident decision
- **CI includes threshold**: Uncertain, need more data

## Implementation Details

### Current Implementation
```python
# Standard Error
std_err = stdev(values) / sqrt(n)

# 95% Confidence Interval (z = 1.96)
CI = mean ± 1.96 × std_err
```

### Notes:
- Using **z-score approximation** (assumes normal distribution)
- For small n (< 30), should use **t-distribution** instead
- Current: z = 1.96 for 95% CI (works well for n ≥ 5)

### Future Improvements:
1. Use **t-distribution** for small n (more accurate)
2. **Bootstrap confidence intervals** (non-parametric, no distribution assumptions)
3. **Effect size** calculations (Cohen's d for comparing models)
4. **Hypothesis testing** (t-tests for comparing models)

## Recommended Number of Trials

- **n = 1**: No statistical confidence (current baseline)
- **n = 3-5**: Basic confidence, detect major differences
- **n = 10**: Good confidence, detect moderate differences
- **n = 20+**: High confidence, detect small differences

**Trade-off**: More trials = more confidence, but also more compute time.

## References

1. **Central Limit Theorem**: Any statistics textbook
2. **Confidence Intervals**: 
   - "Statistical Inference" by Casella & Berger
   - "Introduction to Statistical Learning" by James et al.
3. **LLM Evaluation**:
   - "On the Opportunities and Risks of Foundation Models" (Bommasani et al.)
   - "Evaluating Large Language Models" (Chang et al.)
