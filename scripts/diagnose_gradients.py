"""Diagnostic script to investigate gradient computation."""

import torch
import numpy as np
from pathlib import Path
import yaml

# Load the gradient data
data = np.load('outputs/gradients/conflict_matrices.npz', allow_pickle=True)

print("Gradient Conflict Data Analysis")
print("=" * 60)

averaged = data['averaged']
history = data['history']
task_names = data['task_names'].tolist()
log_count = data['log_count'].item()

print(f"Task names: {task_names}")
print(f"Logged steps: {log_count}")
print(f"History shape: {history.shape}")
print()

# Check the evolution of conflicts over training
print("Gradient Conflict Evolution (first 10, middle 10, last 10 logged steps):")
print("-" * 60)

# Get indices for first, middle, last
n = len(history)
indices = list(range(min(10, n))) + list(range(n//2-5, n//2+5)) + list(range(max(0, n-10), n))
indices = sorted(set(indices))

for idx in indices[:5]:
    print(f"\nStep {idx}:")
    matrix = history[idx]
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i < j:
                print(f"  {t1} vs {t2}: {matrix[i,j]:+.4f}")

print("\n" + "=" * 60)
print("Analysis of gradient conflict distribution:")
print()

# Analyze the off-diagonal values
off_diag_values = []
for i in range(len(task_names)):
    for j in range(len(task_names)):
        if i != j:
            off_diag_values.extend(history[:, i, j].tolist())

off_diag = np.array(off_diag_values)
print(f"Off-diagonal gradient conflicts:")
print(f"  Mean: {off_diag.mean():.6f}")
print(f"  Std:  {off_diag.std():.6f}")
print(f"  Min:  {off_diag.min():.6f}")
print(f"  Max:  {off_diag.max():.6f}")
print()

# Check if diagonal values are ~1 (sanity check)
diag_values = []
for i in range(len(task_names)):
    diag_values.extend(history[:, i, i].tolist())
diag = np.array(diag_values)
print(f"Diagonal values (should be ~1):")
print(f"  Mean: {diag.mean():.6f}")
print(f"  Std:  {diag.std():.6f}")
print()

# Check specific pairs
print("Evolution of key hypothesized pairs:")
pairs = [
    ('esol', 'lipophilicity'),
    ('bbbp', 'herg'),
    ('bace', 'bbbp'),
]

for t1, t2 in pairs:
    if t1 in task_names and t2 in task_names:
        i, j = task_names.index(t1), task_names.index(t2)
        values = history[:, i, j]
        print(f"\n{t1} vs {t2}:")
        print(f"  Initial (first 10): {values[:10].mean():+.4f}")
        print(f"  Middle: {values[len(values)//2-5:len(values)//2+5].mean():+.4f}")
        print(f"  Final (last 10): {values[-10:].mean():+.4f}")
        print(f"  Range: [{values.min():+.4f}, {values.max():+.4f}]")
