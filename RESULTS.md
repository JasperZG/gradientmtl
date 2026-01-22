# Gradient-Based Causal Discovery: Experimental Results

## Executive Summary

This document summarizes results from experiments investigating whether gradient conflicts in multi-task learning reveal mechanistic relationships between molecular properties.

### Key Findings

| Experiment | Result | Interpretation |
|------------|--------|----------------|
| **Exp 2: SAR Validation** | r = 0.918*** | Gradient conflicts strongly correlate with empirical property correlations |
| **Exp 7: Representation** | r = 0.853 | Patterns consistent across ECFP and GNN representations |
| **Exp 4: Task Selection** | Greedy +26-39% vs random | Gradient-informed selection outperforms random at mid-range budgets |
| **Exp 5: PCGrad** | No significant effect | PCGrad shows no differential benefit for "conflicting" vs "synergistic" pairs |

### Critical Discovery
**Gradient conflict analysis requires 100% compound overlap across tasks.** Validated on Tox21 (100% overlap, r=0.918) vs ADME (~1% overlap, r=0.394 n.s.).

---

## Experiment 1: Gradient Conflict Matrix Construction

### Overview
Trained a GNN multi-task model on Tox21 dataset (12 toxicity endpoints) and computed pairwise gradient correlations during training.

### Dataset
- **Molecules**: 5,666 (after filtering for ≥10 valid labels)
- **Tasks**: 12 Tox21 endpoints
- **Task types**: All binary classification (toxicity assays)

### Gradient Conflict Matrix

The 12×12 matrix of pairwise gradient cosine similarities:

| Task | NR-AR | NR-AR-LBD | NR-AhR | NR-Arom | NR-ER | NR-ER-LBD | NR-PPAR | SR-ARE | SR-ATAD5 | SR-HSE | SR-MMP | SR-p53 |
|------|-------|-----------|--------|---------|-------|-----------|---------|--------|----------|--------|--------|--------|
| NR-AR | 1.000 | 0.239 | 0.001 | 0.012 | 0.040 | 0.033 | 0.009 | 0.001 | 0.035 | 0.007 | 0.000 | 0.011 |
| NR-AR-LBD | 0.239 | 1.000 | -0.003 | 0.013 | 0.030 | 0.044 | 0.030 | 0.006 | 0.017 | 0.008 | -0.002 | 0.027 |
| NR-AhR | 0.001 | -0.003 | 1.000 | 0.041 | 0.040 | 0.020 | 0.007 | 0.066 | 0.036 | 0.012 | 0.075 | 0.018 |
| NR-Aromatase | 0.012 | 0.013 | 0.041 | 1.000 | 0.013 | 0.028 | 0.021 | 0.040 | 0.031 | 0.011 | 0.036 | 0.044 |
| NR-ER | 0.040 | 0.030 | 0.040 | 0.013 | 1.000 | 0.237 | 0.006 | 0.033 | 0.062 | 0.008 | 0.050 | 0.017 |
| NR-ER-LBD | 0.033 | 0.044 | 0.020 | 0.028 | 0.237 | 1.000 | 0.020 | 0.048 | 0.042 | 0.024 | 0.074 | 0.056 |
| NR-PPAR-gamma | 0.009 | 0.030 | 0.007 | 0.021 | 0.006 | 0.020 | 1.000 | 0.026 | 0.055 | 0.028 | 0.012 | 0.069 |
| SR-ARE | 0.001 | 0.006 | 0.066 | 0.040 | 0.033 | 0.048 | 0.026 | 1.000 | 0.055 | 0.038 | 0.092 | 0.050 |
| SR-ATAD5 | 0.035 | 0.017 | 0.036 | 0.031 | 0.062 | 0.042 | 0.055 | 0.055 | 1.000 | 0.026 | 0.033 | 0.101 |
| SR-HSE | 0.007 | 0.008 | 0.012 | 0.011 | 0.008 | 0.024 | 0.028 | 0.038 | 0.026 | 1.000 | 0.016 | 0.068 |
| SR-MMP | 0.000 | -0.002 | 0.075 | 0.036 | 0.050 | 0.074 | 0.012 | 0.092 | 0.033 | 0.016 | 1.000 | 0.046 |
| SR-p53 | 0.011 | 0.027 | 0.018 | 0.044 | 0.017 | 0.056 | 0.069 | 0.050 | 0.101 | 0.068 | 0.046 | 1.000 |

### Matrix Statistics

| Metric | Value |
|--------|-------|
| Mean correlation | 0.038 |
| Std deviation | 0.042 |
| Minimum | -0.003 |
| Maximum | 0.239 |
| % positive | 97.0% |
| % negative | 3.0% |

### Top Correlations (Biologically Meaningful)

| Rank | Task Pair | Correlation | Biological Interpretation |
|------|-----------|-------------|---------------------------|
| 1 | NR-AR ↔ NR-AR-LBD | **0.239** | Same receptor (Androgen), different binding domains |
| 2 | NR-ER ↔ NR-ER-LBD | **0.237** | Same receptor (Estrogen), different binding domains |
| 3 | SR-ATAD5 ↔ SR-p53 | **0.101** | Both DNA damage response pathways |
| 4 | SR-ARE ↔ SR-MMP | **0.092** | Both cellular stress response |
| 5 | NR-AhR ↔ SR-MMP | **0.075** | Xenobiotic metabolism connection |

### Key Observation
**Very few negative correlations exist in this dataset.** The Tox21 tasks are mostly weakly positively correlated or independent, with no strong mechanistic conflicts detected. This has implications for downstream experiments.

---

## Experiment 2: SAR Validation (Empirical Correlations)

### Hypothesis
Gradient conflict patterns correlate with empirical correlations computed directly from measured property values.

### Result
**Pearson r = 0.918*** (p < 0.001)

### Interpretation
This is the strongest validation of the core hypothesis. When comparing the gradient conflict matrix G to the empirical correlation matrix E (computed from actual measured labels), there is near-perfect agreement. This confirms that:

1. Gradient conflicts during training capture real statistical relationships in the data
2. The neural network implicitly learns which properties co-vary
3. The method is valid for datasets with 100% compound overlap

### Critical Caveat
This correlation **drops to r = 0.394 (not significant)** when applied to ADME datasets with only ~1% compound overlap. The method requires the same molecules to be measured across all tasks.

---

## Experiment 7: Representation Generalization

### Hypothesis
Gradient conflict patterns should be consistent across different molecular representations if they reflect true mechanistic relationships.

### Comparison
- **ECFP4**: 2048-bit Morgan fingerprints (fixed, interpretable)
- **GCN**: Graph convolutional network (learned representations)

### Result
**Pearson r = 0.853** between ECFP and GNN gradient matrices

### Interpretation
The high correlation indicates that gradient conflicts are largely representation-invariant. This supports the claim that the patterns reflect genuine property relationships rather than artifacts of a specific representation choice.

---

## Experiment 4: Task Selection

### Hypothesis
A gradient-informed greedy algorithm can identify a minimal subset of tasks that maximizes predictive coverage of the remaining tasks.

### Methods Compared
1. **Greedy**: Iteratively select task maximizing positive gradient correlation to unselected tasks
2. **Clustering**: Hierarchical clustering, select one representative per cluster
3. **Diversity**: Maximize pairwise dissimilarity
4. **Random**: Baseline (100 draws)

### Results

| Budget | Greedy | Clustering | Diversity | Random (mean±std) | Greedy vs Random |
|--------|--------|------------|-----------|-------------------|------------------|
| 3 | 0.079 | **0.100** | 0.071 | 0.070 ± 0.015 | +12.4% |
| 4 | 0.109 | **0.110** | 0.079 | 0.078 ± 0.018 | **+39.0%** |
| 5 | **0.116** | 0.116 | 0.113 | 0.087 ± 0.020 | **+33.8%** |
| 6 | 0.122 | **0.136** | 0.125 | 0.097 ± 0.023 | **+26.2%** |
| 7 | 0.099 | **0.149** | 0.144 | 0.104 ± 0.027 | -4.8% |
| 8 | 0.113 | **0.167** | 0.167 | 0.112 ± 0.033 | +0.5% |

### Greedy Selection Order

| Budget | Selected Tasks |
|--------|----------------|
| 3 | NR-ER-LBD, SR-ATAD5, SR-ARE |
| 4 | + NR-AR-LBD |
| 5 | + SR-p53 |
| 6 | + SR-MMP |
| 7 | + NR-ER |
| 8 | + NR-Aromatase |

### Success Criteria Evaluation

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Greedy > Random by 20%+ | All budgets | Budgets 4-6 only | **Partial** |
| Coverage at budget 5-6 | ≥ 0.75 | 0.12 | **Failed** |

### Interpretation

**Positive findings:**
- Greedy selection significantly outperforms random at budgets 4-6 (26-39% improvement)
- The algorithm correctly identifies high-connectivity tasks (NR-ER-LBD, SR-ATAD5) first

**Negative findings:**
- Coverage values are very low (~0.12) compared to the 0.75 target
- Greedy algorithm performance degrades at higher budgets (7-8), where clustering dominates
- The low coverage reflects the weak correlation structure in the gradient matrix (most correlations are 0.01-0.07)

**Root cause:**
The Tox21 tasks have mostly weak positive correlations with no strong synergies to exploit. Selecting a subset provides limited predictive power over the rest because the tasks are relatively independent.

**Recommendation:**
Clustering-based selection outperforms greedy at higher budgets and should be preferred when budget > 6.

---

## Experiment 5: PCGrad Validation

### Hypothesis
If gradient conflicts represent real mechanistic incompatibilities:
- PCGrad should **help** high-conflict pairs (reduce interference)
- PCGrad should **not help** synergistic pairs (no interference to fix)

### Task Pair Categories

**High-conflict pairs** (hypothesized based on biology):
- NR-AR vs NR-ER (Androgen vs Estrogen receptors)
- NR-AR vs NR-Aromatase (AR signaling vs aromatase inhibition)
- NR-PPAR-gamma vs SR-MMP (Metabolic vs stress response)
- NR-AhR vs SR-HSE (Xenobiotic vs heat shock)
- NR-Aromatase vs SR-ARE (Enzyme vs oxidative stress)

**Synergistic pairs** (same receptor/pathway families):
- NR-AR vs NR-AR-LBD (Same receptor, different binding sites)
- NR-ER vs NR-ER-LBD (Same receptor, different binding sites)
- SR-ARE vs SR-HSE (Both stress response)
- SR-MMP vs SR-p53 (Both stress/apoptosis related)
- SR-ATAD5 vs SR-p53 (Both DNA damage related)

**Random pairs** (control):
- NR-AR-LBD vs SR-MMP
- NR-AhR vs NR-ER-LBD
- NR-PPAR-gamma vs SR-ARE
- NR-Aromatase vs SR-ATAD5
- NR-ER vs SR-HSE

### Results

| Category | N | Avg PCGrad Improvement | Std | Pairs Helped (>1%) |
|----------|---|------------------------|-----|-------------------|
| High-conflict | 5 | **-0.0002** | 0.0031 | 0/5 |
| Synergistic | 5 | **-0.0033** | 0.0092 | 0/5 |
| Random | 5 | **+0.0000** | 0.0019 | 0/5 |

### Detailed Results

| Task Pair | Category | Baseline AUC | PCGrad AUC | Delta |
|-----------|----------|--------------|------------|-------|
| NR-AR vs NR-AR-LBD | synergistic | 0.894 | 0.890 | -0.004 |
| NR-ER vs NR-ER-LBD | synergistic | 0.751 | 0.751 | +0.001 |
| SR-ATAD5 vs SR-p53 | synergistic | 0.863 | 0.859 | -0.004 |
| SR-MMP vs SR-p53 | synergistic | 0.846 | 0.855 | +0.009 |
| SR-ARE vs SR-HSE | synergistic | 0.799 | 0.780 | -0.019 |
| NR-AR vs NR-ER | high_conflict | 0.765 | 0.766 | +0.001 |
| NR-AR vs NR-Aromatase | high_conflict | 0.842 | 0.841 | -0.001 |
| NR-PPAR-gamma vs SR-MMP | high_conflict | 0.866 | 0.868 | +0.001 |
| NR-Aromatase vs SR-ARE | high_conflict | 0.816 | 0.819 | +0.003 |
| NR-AhR vs SR-HSE | high_conflict | 0.828 | 0.822 | -0.006 |
| NR-AR-LBD vs SR-MMP | random | 0.906 | 0.909 | +0.003 |
| NR-AhR vs NR-ER-LBD | random | 0.871 | 0.872 | +0.001 |
| NR-PPAR-gamma vs SR-ARE | random | 0.832 | 0.831 | -0.000 |
| NR-Aromatase vs SR-ATAD5 | random | 0.832 | 0.831 | -0.001 |
| NR-ER vs SR-HSE | random | 0.718 | 0.715 | -0.002 |

### Statistical Test

| Comparison | t-statistic | p-value |
|------------|-------------|---------|
| High-conflict vs Synergistic | 0.645 | 0.537 |

**Not statistically significant** (p > 0.05)

### Interpretation

**The hypothesis was NOT supported.** PCGrad shows no differential benefit for any category:

1. **No conflicts to resolve**: The gradient matrix shows almost no negative correlations (97% positive). The "high-conflict" pairs were hypothesized based on biology, but the actual gradient correlations were near-zero or slightly positive, not negative.

2. **Baseline performance already high**: Most task pairs achieve 0.75-0.90 AUC without PCGrad, leaving little room for improvement.

3. **Variance dominates signal**: The improvements/degradations (±0.01) are within noise range and show no systematic pattern.

**Conclusion**: PCGrad is unnecessary for Tox21 because there are no meaningful gradient conflicts to resolve. This is consistent with the gradient matrix showing mostly weak positive correlations.

---

## Overall Conclusions

### What Worked

1. **Core validation (Exp 2)**: Gradient conflicts strongly correlate with empirical property correlations (r = 0.918) when compound overlap is 100%

2. **Representation invariance (Exp 7)**: Patterns are consistent across ECFP and GNN (r = 0.853)

3. **Task selection (Exp 4)**: Greedy selection outperforms random by 26-39% for mid-range budgets

### What Didn't Work as Expected

1. **Low coverage (Exp 4)**: Coverage values (~0.12) far below 0.75 target due to weak task correlations

2. **PCGrad validation (Exp 5)**: No differential effect because Tox21 lacks true gradient conflicts

3. **Novel trade-off discovery**: Not possible because the gradient matrix has no strong negative correlations

### Root Cause Analysis

**The Tox21 dataset may not be ideal for this research:**

| Expected | Actual |
|----------|--------|
| Strong positive AND negative correlations | 97% weakly positive, 3% near-zero |
| Clear mechanistic clusters | Weak clustering structure |
| Tasks with conflicting requirements | Tasks are mostly independent |

The 12 Tox21 endpoints, while having 100% compound overlap, represent relatively independent toxicity mechanisms that don't strongly conflict or synergize.

### Recommendations

1. **Try different datasets**: ChEMBL or TDC datasets with known mechanistic relationships may show stronger conflict patterns

2. **Expand to more diverse properties**: Combining binding, ADME, and toxicity tasks (with matched compounds) may reveal more interesting trade-offs

3. **Focus on clustering-based selection**: Outperforms greedy at higher budgets

4. **De-prioritize PCGrad experiments**: Unlikely to show effects without true conflicts

---

## Appendix: File Locations

| Output | Path |
|--------|------|
| Gradient matrix | `outputs/gradients/gnn_conflict_matrices.npz` |
| Task selection results | `outputs/experiment4/task_selection_summary.csv` |
| Task selection plots | `outputs/experiment4/coverage_curves.png` |
| Task dendrogram | `outputs/experiment4/task_dendrogram.png` |
| Greedy selections | `outputs/experiment4/greedy_selections.json` |
| PCGrad results | `outputs/pcgrad/*.json` |

---

## Appendix: Experiment Status

| Experiment | Status | Notes |
|------------|--------|-------|
| Exp 1: Gradient Matrix | ✅ Complete | 12×12 matrix saved |
| Exp 2: SAR Validation | ✅ Complete | r = 0.918 |
| Exp 3: Transfer Learning | ❌ Not run | 792 jobs, 5-8 days on single GPU |
| Exp 4: Task Selection | ✅ Complete | Greedy works for budgets 4-6 |
| Exp 5: PCGrad | ✅ Complete | No significant effect |
| Exp 6: Novel Discovery | ❌ Not run | Low priority given weak conflicts |
| Exp 7: Representation | ✅ Complete | r = 0.853 |
