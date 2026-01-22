"""
Analyze why Tox21 gradient correlations are weaker than expected.

Hypotheses:
1. Batch composition: Different batches have different task coverage
2. Gradient magnitude: Some tasks dominate due to imbalanced data
3. Early training noise: Correlations might be stronger later in training
4. Architecture: Separate heads might naturally orthogonalize gradients
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load gradient data
data = np.load('outputs/gradients/conflict_matrices.npz', allow_pickle=True)
history = data['history']
task_names = data['task_names'].tolist()
n_steps = len(history)

print("=" * 70)
print("ANALYSIS: Why Are Gradient Correlations Weak?")
print("=" * 70)

# 1. Check if correlations strengthen over training
print("\n1. TEMPORAL EVOLUTION OF KEY CORRELATIONS")
print("-" * 70)

key_pairs = [
    ('NR-AR', 'NR-AR-LBD', 'Same receptor'),
    ('NR-ER', 'NR-ER-LBD', 'Same receptor'),
    ('SR-ARE', 'SR-MMP', 'Same pathway'),
    ('NR-AR', 'SR-p53', 'Different pathways'),
]

for t1, t2, desc in key_pairs:
    if t1 in task_names and t2 in task_names:
        i, j = task_names.index(t1), task_names.index(t2)
        values = history[:, i, j]

        # Split into thirds
        n = len(values)
        early = values[:n//3].mean()
        mid = values[n//3:2*n//3].mean()
        late = values[2*n//3:].mean()

        print(f"{t1} vs {t2} ({desc}):")
        print(f"  Early: {early:+.4f}  Mid: {mid:+.4f}  Late: {late:+.4f}")
        print(f"  Max: {values.max():+.4f}  Min: {values.min():+.4f}")

# 2. Check variance of correlations over time
print("\n2. STABILITY OF CORRELATIONS")
print("-" * 70)

# Compute variance for each pair
pair_vars = []
for i in range(len(task_names)):
    for j in range(i+1, len(task_names)):
        vals = history[:, i, j]
        pair_vars.append((task_names[i], task_names[j], vals.std(), vals.mean()))

# Sort by variance
pair_vars.sort(key=lambda x: -x[2])

print("Most variable pairs (high noise):")
for t1, t2, std, mean in pair_vars[:5]:
    print(f"  {t1} vs {t2}: mean={mean:+.4f}, std={std:.4f}")

print("\nMost stable pairs (low noise):")
for t1, t2, std, mean in pair_vars[-5:]:
    print(f"  {t1} vs {t2}: mean={mean:+.4f}, std={std:.4f}")

# 3. Check if there's a pattern in the correlation matrix structure
print("\n3. MECHANISTIC GROUPING ANALYSIS")
print("-" * 70)

avg_matrix = data['averaged']

# Define groups
nr_tasks = [t for t in task_names if t.startswith('NR-')]
sr_tasks = [t for t in task_names if t.startswith('SR-')]

# Compute within-group and between-group correlations
def get_group_correlations(group1, group2, matrix, names):
    vals = []
    for t1 in group1:
        for t2 in group2:
            if t1 != t2:
                i, j = names.index(t1), names.index(t2)
                vals.append(matrix[i, j])
    return vals

within_nr = get_group_correlations(nr_tasks, nr_tasks, avg_matrix, task_names)
within_sr = get_group_correlations(sr_tasks, sr_tasks, avg_matrix, task_names)
between = get_group_correlations(nr_tasks, sr_tasks, avg_matrix, task_names)

print(f"Within NR group ({len(nr_tasks)} tasks): mean={np.mean(within_nr):+.4f}, std={np.std(within_nr):.4f}")
print(f"Within SR group ({len(sr_tasks)} tasks): mean={np.mean(within_sr):+.4f}, std={np.std(within_sr):.4f}")
print(f"Between NR-SR groups:                    mean={np.mean(between):+.4f}, std={np.std(between):.4f}")

# Statistical test: is within-group > between-group?
from scipy import stats
t_nr, p_nr = stats.ttest_ind(within_nr, between)
t_sr, p_sr = stats.ttest_ind(within_sr, between)

print(f"\nStatistical tests (within vs between):")
print(f"  NR within vs between: t={t_nr:.2f}, p={p_nr:.4f}")
print(f"  SR within vs between: t={t_sr:.2f}, p={p_sr:.4f}")

# 4. Check diagonal values (should be ~1)
print("\n4. SANITY CHECK: DIAGONAL VALUES")
print("-" * 70)

diag_vals = [avg_matrix[i, i] for i in range(len(task_names))]
print(f"Diagonal values (self-similarity): mean={np.mean(diag_vals):.4f}, min={np.min(diag_vals):.4f}")

if np.min(diag_vals) < 0.9:
    print("WARNING: Some diagonal values are low - possible issue with gradient computation")

# 5. Check absolute values - maybe correlations exist but are small
print("\n5. ABSOLUTE VALUE ANALYSIS")
print("-" * 70)

# Off-diagonal absolute values
off_diag = []
for i in range(len(task_names)):
    for j in range(len(task_names)):
        if i != j:
            off_diag.append(abs(avg_matrix[i, j]))

print(f"Off-diagonal |correlation|: mean={np.mean(off_diag):.4f}, max={np.max(off_diag):.4f}")
print(f"Expected if random: ~0")
print(f"Expected if meaningful: 0.1-0.5")

# 6. Suggestions
print("\n" + "=" * 70)
print("CONCLUSIONS AND SUGGESTIONS")
print("=" * 70)

if np.mean(within_nr) > np.mean(between):
    print("[+] Within-NR correlations ARE higher than between-group (correct direction)")
else:
    print("[!] Within-NR correlations NOT higher than between-group")

if np.mean(within_sr) > np.mean(between):
    print("[+] Within-SR correlations ARE higher than between-group (correct direction)")
else:
    print("[!] Within-SR correlations NOT higher than between-group")

print("""
Possible reasons for weak correlations:

1. ECFP fingerprints are FIXED - the model only learns linear/nonlinear
   transformations, not representations. Graph NNs might show stronger patterns.

2. Separate task heads may orthogonalize gradients - the encoder learns
   general features while heads specialize.

3. Class imbalance (3-16% positive rate) affects gradient magnitudes.

4. The shared encoder architecture may be too simple for capturing
   mechanistic relationships.

Suggested improvements:
- Try graph neural networks (learn representations)
- Use deeper/wider encoder
- Try PCGrad to see if removing conflicts helps
- Focus on molecules with labels for ALL tasks
""")
