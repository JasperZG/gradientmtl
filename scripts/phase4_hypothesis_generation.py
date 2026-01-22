#!/usr/bin/env python3
"""
Phase 4C: Hypothesis Generation

Identify unexpected gradient patterns and generate testable hypotheses.

Key outputs:
1. Unexpected high-correlation pairs (different families)
2. Unexpected low-correlation pairs (same family)
3. Testable hypotheses for each
4. Suggested validation experiments
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# Kinase family definitions
KINASE_FAMILIES = {
    'CDK': ['CDK1', 'CDK2', 'CDK4', 'CDK5', 'CDK6', 'CDK7', 'CDK9'],
    'JAK': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
    'SRC': ['SRC', 'FYN', 'LCK', 'LYN', 'YES1', 'HCK'],
    'Aurora': ['AURKA', 'AURKB', 'AURKC'],
    'EGFR': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4'],
}


def load_gradient_matrix(results_dir: str) -> tuple:
    """Load gradient matrix and task names."""
    data = np.load(f"{results_dir}/gradient_matrices.npz", allow_pickle=True)
    G = data['average_matrix']
    tasks = list(data['tasks'])
    return G, tasks


def get_kinase_family(kinase: str) -> str:
    """Get family for a kinase."""
    for family, members in KINASE_FAMILIES.items():
        if kinase in members:
            return family
    return 'Other'


def find_unexpected_patterns(G: np.ndarray, tasks: list) -> dict:
    """Identify unexpected gradient patterns."""

    n = len(tasks)
    unexpected = {
        'high_correlation_different_family': [],
        'low_correlation_same_family': [],
        'negative_correlations': []
    }

    for i in range(n):
        for j in range(i + 1, n):
            k1 = tasks[i].replace('_pIC50', '')
            k2 = tasks[j].replace('_pIC50', '')
            g_val = G[i, j]

            fam1 = get_kinase_family(k1)
            fam2 = get_kinase_family(k2)
            same_family = (fam1 == fam2) and fam1 != 'Other'

            # Unexpected high correlation between different families
            if not same_family and g_val > 0.25:
                unexpected['high_correlation_different_family'].append({
                    'kinase1': k1,
                    'kinase2': k2,
                    'family1': fam1,
                    'family2': fam2,
                    'gradient_G': round(g_val, 4)
                })

            # Unexpected low correlation within same family
            if same_family and g_val < 0.15:
                unexpected['low_correlation_same_family'].append({
                    'kinase1': k1,
                    'kinase2': k2,
                    'family': fam1,
                    'gradient_G': round(g_val, 4)
                })

            # Any negative correlations
            if g_val < -0.02:
                unexpected['negative_correlations'].append({
                    'kinase1': k1,
                    'kinase2': k2,
                    'family1': fam1,
                    'family2': fam2,
                    'gradient_G': round(g_val, 4)
                })

    # Sort by gradient value
    unexpected['high_correlation_different_family'].sort(key=lambda x: -x['gradient_G'])
    unexpected['low_correlation_same_family'].sort(key=lambda x: x['gradient_G'])
    unexpected['negative_correlations'].sort(key=lambda x: x['gradient_G'])

    return unexpected


def generate_hypothesis(pattern_type: str, pair: dict) -> dict:
    """Generate testable hypothesis for an unexpected pattern."""

    k1, k2 = pair['kinase1'], pair['kinase2']
    g_val = pair['gradient_G']

    if pattern_type == 'high_correlation_different_family':
        f1, f2 = pair['family1'], pair['family2']
        hypothesis = {
            'observation': f"{k1} ({f1}) and {k2} ({f2}) show high gradient correlation (G={g_val:.3f}) despite being in different kinase families",
            'hypothesis': f"{k1} and {k2} share similar ATP-binding site features or substrate recognition motifs that lead to similar compound sensitivity profiles",
            'mechanism_candidates': [
                "Convergent evolution of binding site geometry",
                "Shared allosteric pocket features",
                "Similar gatekeeper residue properties",
                "Common DFG motif conformation preferences"
            ],
            'validation_experiments': [
                f"Compare X-ray structures of {k1} and {k2} bound to same inhibitor",
                f"Test known {k1}-selective compounds on {k2} and vice versa",
                f"Sequence alignment of ATP-binding site residues",
                f"Molecular docking of shared high-affinity compounds"
            ],
            'prediction': f"Compounds optimized for {k1} selectivity will likely also hit {k2}"
        }

    elif pattern_type == 'low_correlation_same_family':
        fam = pair['family']
        hypothesis = {
            'observation': f"{k1} and {k2} (both {fam} family) show unexpectedly low gradient correlation (G={g_val:.3f})",
            'hypothesis': f"Despite sequence homology, {k1} and {k2} have divergent binding site features that allow selective inhibitor design",
            'mechanism_candidates': [
                "Differences in gatekeeper residue",
                "Unique back pocket accessibility",
                "Distinct DFG-out binding preferences",
                "Different induced-fit conformational changes"
            ],
            'validation_experiments': [
                f"Identify compounds selective for {k1} over {k2} in ChEMBL",
                f"Compare binding site volumes and shapes",
                f"Analyze co-crystal structures for selectivity determinants",
                f"Test Type II vs Type I inhibitor preferences"
            ],
            'prediction': f"Selective {k1}/{k2} inhibitors are achievable despite family membership"
        }

    elif pattern_type == 'negative_correlations':
        f1 = pair.get('family1', 'Unknown')
        f2 = pair.get('family2', 'Unknown')
        hypothesis = {
            'observation': f"{k1} and {k2} show negative gradient correlation (G={g_val:.3f}), indicating selectivity trade-off",
            'hypothesis': f"Structural features favoring {k1} binding are incompatible with {k2} binding, creating a selectivity axis",
            'mechanism_candidates': [
                "Opposite steric requirements in binding pocket",
                "Conflicting electrostatic preferences",
                "Mutually exclusive binding conformations",
                "Inverse correlation of key residue properties"
            ],
            'validation_experiments': [
                f"Identify compounds with {k1}-selectivity profile in ChEMBL",
                f"Structural superposition to identify selectivity determinants",
                f"SAR analysis of selectivity-driving substitutions",
                f"Design selectivity assay panel for {k1}/{k2}"
            ],
            'prediction': f"Optimizing for {k1} potency will decrease {k2} potency and vice versa"
        }

    else:
        hypothesis = {'observation': 'Unknown pattern type', 'hypothesis': '', 'validation_experiments': []}

    return hypothesis


def generate_hypotheses_report(unexpected: dict, output_dir: str):
    """Generate comprehensive hypothesis report."""

    lines = [
        "# Hypothesis Generation Report",
        "",
        "## Overview",
        "",
        "This report identifies unexpected gradient patterns and generates",
        "testable hypotheses to explain them. These represent opportunities",
        "for novel discoveries about kinase relationships.",
        "",
        "---",
        ""
    ]

    # High correlation different family
    if unexpected['high_correlation_different_family']:
        lines.extend([
            "## Unexpected Cross-Family Correlations",
            "",
            "These kinase pairs from different families show unexpectedly high",
            "gradient correlation, suggesting shared binding site features.",
            ""
        ])

        for i, pair in enumerate(unexpected['high_correlation_different_family'][:5], 1):
            hyp = generate_hypothesis('high_correlation_different_family', pair)

            lines.extend([
                f"### Finding {i}: {pair['kinase1']} ↔ {pair['kinase2']}",
                "",
                f"**Observation:** {hyp['observation']}",
                "",
                f"**Hypothesis:** {hyp['hypothesis']}",
                "",
                "**Possible mechanisms:**",
            ])
            for m in hyp['mechanism_candidates']:
                lines.append(f"- {m}")

            lines.extend([
                "",
                "**Validation experiments:**",
            ])
            for v in hyp['validation_experiments']:
                lines.append(f"1. {v}")

            lines.extend([
                "",
                f"**Prediction:** {hyp['prediction']}",
                "",
                "---",
                ""
            ])

    # Low correlation same family
    if unexpected['low_correlation_same_family']:
        lines.extend([
            "## Unexpected Within-Family Divergence",
            "",
            "These kinase pairs from the same family show unexpectedly low",
            "gradient correlation, suggesting selectivity opportunities.",
            ""
        ])

        for i, pair in enumerate(unexpected['low_correlation_same_family'][:5], 1):
            hyp = generate_hypothesis('low_correlation_same_family', pair)

            lines.extend([
                f"### Finding {i}: {pair['kinase1']} ↔ {pair['kinase2']} ({pair['family']} family)",
                "",
                f"**Observation:** {hyp['observation']}",
                "",
                f"**Hypothesis:** {hyp['hypothesis']}",
                "",
                "**Possible mechanisms:**",
            ])
            for m in hyp['mechanism_candidates']:
                lines.append(f"- {m}")

            lines.extend([
                "",
                "**Validation experiments:**",
            ])
            for v in hyp['validation_experiments']:
                lines.append(f"1. {v}")

            lines.extend([
                "",
                f"**Prediction:** {hyp['prediction']}",
                "",
                "---",
                ""
            ])

    # Negative correlations
    if unexpected['negative_correlations']:
        lines.extend([
            "## Selectivity Trade-offs (Negative Correlations)",
            "",
            "These pairs show negative gradient correlation, indicating",
            "that optimizing for one kinase impairs binding to the other.",
            ""
        ])

        for i, pair in enumerate(unexpected['negative_correlations'][:5], 1):
            hyp = generate_hypothesis('negative_correlations', pair)

            lines.extend([
                f"### Finding {i}: {pair['kinase1']} ↔ {pair['kinase2']}",
                "",
                f"**Observation:** {hyp['observation']}",
                "",
                f"**Hypothesis:** {hyp['hypothesis']}",
                "",
                "**Possible mechanisms:**",
            ])
            for m in hyp['mechanism_candidates']:
                lines.append(f"- {m}")

            lines.extend([
                "",
                "**Validation experiments:**",
            ])
            for v in hyp['validation_experiments']:
                lines.append(f"1. {v}")

            lines.extend([
                "",
                f"**Prediction:** {hyp['prediction']}",
                "",
                "---",
                ""
            ])

    # Summary table
    lines.extend([
        "## Summary of Testable Hypotheses",
        "",
        "| Pattern | Kinase Pair | G | Key Prediction |",
        "|---------|-------------|---|----------------|"
    ])

    for pair in unexpected['high_correlation_different_family'][:3]:
        lines.append(f"| Cross-family high | {pair['kinase1']}-{pair['kinase2']} | {pair['gradient_G']:.3f} | Shared compound sensitivity |")

    for pair in unexpected['low_correlation_same_family'][:3]:
        lines.append(f"| Same-family low | {pair['kinase1']}-{pair['kinase2']} | {pair['gradient_G']:.3f} | Selectivity achievable |")

    for pair in unexpected['negative_correlations'][:3]:
        lines.append(f"| Negative | {pair['kinase1']}-{pair['kinase2']} | {pair['gradient_G']:.3f} | Selectivity trade-off |")

    with open(f"{output_dir}/hypothesis_report.md", 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved hypothesis report to {output_dir}/hypothesis_report.md")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 4C: Hypothesis Generation')
    parser.add_argument('--results-dir', default='outputs/kinase_all_results',
                        help='Directory with gradient matrices')
    parser.add_argument('--output-dir', default='outputs/phase4_hypotheses',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 4C: Hypothesis Generation")
    print("=" * 50)

    # Load gradient matrix
    print("\nLoading gradient matrix...")
    G, tasks = load_gradient_matrix(args.results_dir)
    print(f"Loaded {len(tasks)} kinases")

    # Find unexpected patterns
    print("\nIdentifying unexpected patterns...")
    unexpected = find_unexpected_patterns(G, tasks)

    print(f"  High-G cross-family pairs: {len(unexpected['high_correlation_different_family'])}")
    print(f"  Low-G same-family pairs: {len(unexpected['low_correlation_same_family'])}")
    print(f"  Negative correlations: {len(unexpected['negative_correlations'])}")

    # Save raw findings
    with open(f"{output_dir}/unexpected_patterns.json", 'w') as f:
        json.dump(unexpected, f, indent=2)

    # Generate hypothesis report
    print("\nGenerating hypothesis report...")
    generate_hypotheses_report(unexpected, str(output_dir))

    # Print key findings
    print("\n" + "=" * 50)
    print("KEY FINDINGS")
    print("=" * 50)

    if unexpected['high_correlation_different_family']:
        print("\nUnexpected cross-family correlations:")
        for p in unexpected['high_correlation_different_family'][:3]:
            print(f"  {p['kinase1']} ({p['family1']}) ↔ {p['kinase2']} ({p['family2']}): G={p['gradient_G']:.3f}")

    if unexpected['negative_correlations']:
        print("\nSelectivity trade-offs (negative G):")
        for p in unexpected['negative_correlations'][:3]:
            print(f"  {p['kinase1']} ↔ {p['kinase2']}: G={p['gradient_G']:.3f}")


if __name__ == '__main__':
    main()
