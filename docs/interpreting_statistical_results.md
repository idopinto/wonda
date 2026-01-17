# Interpreting Statistical Evaluation Results

## Format Explanation

All metrics follow this format:
```
Metric Name: mean ± std [CI_lower, CI_upper]
```

Where:
- **mean**: Average value across all trials
- **± std**: Standard deviation (measure of variability)
- **[CI_lower, CI_upper]**: 95% Confidence Interval (range where true value likely lies)

---

## Example Output Breakdown

### 1. **Validation Rate: 0.8500 ± 0.0707 [0.7614, 0.9386]**

**What it means**:
- **0.8500 (85%)**: On average, 85% of generated invariants are syntactically valid
- **± 0.0707 (7.07%)**: Low-moderate variability
- **[0.7614, 0.9386]**: We're 95% confident the true validation rate is between 76.1% and 93.9%

**Interpretation**:
- ✅ **Good mean**: 85% syntactic validity is good
- ✅ **Low variance**: ±7% is relatively stable
- ✅ **Narrow CI**: [76%, 94%] is reasonable
- **Insight**: Model generates mostly syntactically valid invariants

**For paper**: "Our model generates 85.0% ± 7.1% syntactically valid invariants [76.1%, 93.9%]."

**Note**: This is the first filter - only syntactically valid invariants proceed to correctness checking.

---

### 2. **Correctness Rate: 0.5000 ± 0.1414 [0.3040, 0.6960]**

**What it means**:
- **0.5000 (50%)**: On average, 50% of generated invariants are correct
- **± 0.1414 (14.14%)**: Results vary by ±14% across trials
- **[0.3040, 0.6960]**: We're 95% confident the true correctness rate is between 30.4% and 69.6%

**Interpretation**:
- ✅ **Mean**: Moderate performance (50% correct)
- ⚠️ **High variance**: ±14% is quite high - model is inconsistent
- ⚠️ **Wide CI**: [30%, 70%] is very wide - high uncertainty
- **Insight**: Model performance is highly variable. In some trials it's good (~70%), in others poor (~30%)

**For paper**: "Our model achieves 50.0% ± 14.1% correctness rate [30.4%, 69.6%], indicating moderate but variable performance."

---

### 3. **Speedup Rate (w/o Model Latency): 0.4000 ± 0.2828 [0.0080, 0.7920]**

**What it means**:
- **0.4000 (40%)**: On average, 40% of correct invariants provide speedup
- **± 0.2828 (28.28%)**: High variability - sometimes 0%, sometimes 80%
- **[0.0080, 0.7920]**: 95% confident true rate is between 0.8% and 79.2%

**Interpretation**:
- ⚠️ **High variance**: ±28% is very high
- ⚠️ **Wide CI**: [0.8%, 79.2%] spans almost the entire range
- **Insight**: Speedup is highly unpredictable - sometimes works well, sometimes not at all

**For paper**: "40.0% ± 28.3% of correct invariants provide speedup [0.8%, 79.2%], showing high variability in speedup achievement."

---

### 4. **Speedup >1 (w/o Model Latency): 4.0549 ± 3.6630 [-1.0218, 9.1316]**

**What it means**:
- **4.0549x**: Average speedup (when speedup > 1) is 4.05x faster
- **± 3.6630**: Very high variability - sometimes 1x, sometimes 8x
- **[-1.0218, 9.1316]**: CI includes negative values (statistical artifact)

**Interpretation**:
- ✅ **Good mean**: 4x speedup is excellent
- ⚠️ **Very high variance**: ±3.66x means results are unpredictable
- ⚠️ **Negative CI lower bound**: This is a statistical artifact - speedup can't be negative. The CI is wide because variance is high.

**For paper**: "When speedup occurs, average speedup is 4.05x ± 3.66x [0.0x, 9.1x] (lower bound truncated at 1.0x)."

**Note**: The negative lower bound is a statistical artifact. In practice, speedup ≥ 1.0x, so you'd report [1.0x, 9.1x] or just note the mean ± std.

---

### 5. **Speedup >1 (w/ Model Latency): 2.8912 ± 2.5485 [-0.6409, 6.4233]**

**What it means**:
- **2.8912x**: Average speedup including model generation time is 2.89x
- **± 2.5485**: High variability
- **[-0.6409, 6.4233]**: Wide CI (negative lower bound is artifact)

**Interpretation**:
- ✅ **Good mean**: ~3x speedup including generation time is good
- ⚠️ **High variance**: Still unpredictable
- **Insight**: Including model latency reduces speedup from 4x to 3x (expected)

**For paper**: "Including model generation time, average speedup is 2.89x ± 2.55x [1.0x, 6.4x]."

---

### 6. **Speedup All (w/o Model Latency): 2.7400 ± 2.3293 [-0.4882, 5.9682]**

**What it means**:
- **2.7400x**: Average speedup across ALL examples (including those without speedup, set to 1.0x)
- **± 2.3293**: High variability
- **[-0.4882, 5.9682]**: Wide CI

**Interpretation**:
- ✅ **Good mean**: 2.74x average speedup overall
- ⚠️ **High variance**: Unpredictable
- **Difference from "Speedup >1"**: This includes all examples (non-speedup = 1.0x), so lower than "Speedup >1"

**For paper**: "Overall average speedup (including all examples) is 2.74x ± 2.33x [1.0x, 5.97x]."

---

### 7. **Speedup All (w/ Model Latency): 2.1169 ± 1.5543 [-0.0373, 4.2711]**

**What it means**:
- **2.1169x**: Average speedup including model latency across all examples
- **± 1.5543**: Moderate-high variability
- **[-0.0373, 4.2711]**: Wide CI (negative lower bound is artifact)

**Interpretation**:
- ✅ **Good mean**: ~2x speedup including generation time
- ⚠️ **Moderate variance**: Still variable but better than others
- **Insight**: This is the most realistic metric (includes all costs)

**For paper**: "End-to-end speedup (including model generation) is 2.12x ± 1.55x [1.0x, 4.27x]."

---

## Key Insights from Your Results

### 1. **High Variance Across All Metrics**
- All metrics show high standard deviations (±14% to ±36%)
- **Implication**: Model performance is **unpredictable** - results vary significantly between runs
- **Action**: Consider running more trials (10+) to get better estimates

### 2. **Wide Confidence Intervals**
- All CIs are very wide (e.g., [30%, 70%] for correctness)
- **Implication**: **High uncertainty** in true performance
- **Action**: Need more trials to narrow the CI

### 3. **Good Mean Performance**
- Correctness: 50% (moderate)
- Speedup: 2-4x (good)
- **Implication**: When it works, it works well, but it's inconsistent

### 4. **Negative CI Lower Bounds**
- Some CIs have negative lower bounds (e.g., [-1.02, 9.13])
- **Why**: High variance + statistical calculation
- **Reality**: Speedup can't be negative, so truncate at 0 or 1.0x
- **Action**: Report as [1.0x, upper] or note the truncation

---

## How to Report in Your Paper

### Option 1: Full Statistical Report
```
Our model achieves:
- Validation Rate: 85.0% ± 7.1% [76.1%, 93.9%]
- Correctness Rate: 50.0% ± 14.1% [30.4%, 69.6%]
- Speedup Rate: 40.0% ± 28.3% [0.8%, 79.2%]
- Average Speedup (when >1x): 4.05x ± 3.66x [1.0x, 9.1x]
- End-to-end Speedup: 2.12x ± 1.55x [1.0x, 4.27x]
```

### Option 2: Simplified (Mean ± Std)
```
Our model achieves:
- Validation Rate: 85.0% ± 7.1%
- Correctness Rate: 50.0% ± 14.1%
- Speedup Rate: 40.0% ± 28.3%
- Average Speedup: 4.05x ± 3.66x (when speedup occurs)
- End-to-end Speedup: 2.12x ± 1.55x
```

### Option 3: With Interpretation
```
Our model generates 85.0% ± 7.1% syntactically valid invariants, of which 
50.0% ± 14.1% are correct. Among correct invariants, 40.0% ± 28.3% provide 
speedup. When speedup occurs, average speedup is 4.05x ± 3.66x, resulting in 
an end-to-end speedup of 2.12x ± 1.55x including model generation time. The 
high variance (±14-28%) in correctness and speedup rates indicates 
performance variability across runs.
```

---

## Recommendations

### 1. **Run More Trials**
Your high variance suggests you need more trials:
- **Current**: Likely 2-3 trials (based on wide CIs)
- **Recommended**: 10+ trials to get reliable estimates

### 2. **Address High Variance**
High variance could indicate:
- Model instability
- Dataset difficulty variation
- Need for better hyperparameters

### 3. **Report Conservatively**
Given wide CIs:
- Report mean ± std
- Note the high variability
- Acknowledge uncertainty in conclusions

### 4. **Consider Median**
For highly variable metrics, median might be more robust:
- Median is less affected by outliers
- Better for skewed distributions

---

## Understanding the Metrics

### Validation Rate
- **Definition**: % of generated invariants that are syntactically valid (parse correctly)
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Interpretation**: First filter - only valid invariants can be correct
- **Typical values**: 80-95% is good, <70% indicates generation issues

### Correctness Rate
- **Definition**: % of generated invariants that are correct
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Your result**: 50% average, but highly variable

### Speedup Rate
- **Definition**: % of correct invariants that provide speedup
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Your result**: 40% average, very variable

### Speedup >1
- **Definition**: Average speedup value (only for examples with speedup > 1x)
- **Range**: ≥ 1.0 (can be any positive number)
- **Your result**: 4.05x average (excellent when it happens)

### Speedup All
- **Definition**: Average speedup across ALL examples (non-speedup = 1.0x)
- **Range**: ≥ 1.0
- **Your result**: 2.74x average (realistic overall performance)

---

## Visual Interpretation

```
Correctness Rate: 50% ± 14%
    30% ──────────●────────── 70%
         (worst)  (mean)  (best)
    
    Interpretation: Sometimes 30%, sometimes 70%, average 50%
```

```
Speedup >1: 4.05x ± 3.66x
    1x ────────●────────────── 8x
    (min)    (mean)         (max)
    
    Interpretation: When speedup happens, it's 1x to 8x, average 4x
```

---

## Questions to Consider

1. **Why is variance so high?**
   - Model instability?
   - Dataset difficulty?
   - Need more trials?

2. **Is 50% correctness acceptable?**
   - Depends on baseline
   - Depends on application
   - Consider error analysis

3. **Is 2-4x speedup good?**
   - Yes, if consistent
   - But high variance is concerning
   - Need to improve reliability
