#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('outputs/kinase_data/kinase_all_activity_matrix.csv')
task_cols = [c for c in df.columns if c.endswith('_pIC50')]
coverage = df[task_cols].notna().mean()
high_cov = coverage[coverage > 0.25].index.tolist()
print(f'High coverage tasks (>25%): {high_cov}')
keep_cols = ['chembl_id', 'smiles'] + high_cov
df[keep_cols].to_csv('outputs/kinase_data/kinase_highcov_activity_matrix.csv', index=False)
print(f'Saved {len(df)} compounds x {len(high_cov)} kinases')
