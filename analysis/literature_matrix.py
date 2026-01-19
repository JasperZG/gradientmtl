"""
Literature Relationship Matrix for SAR Validation.

Contains documented mechanistic relationships from medicinal chemistry literature:
- Wermuth's Practice of Medicinal Chemistry
- Silverman's Organic Chemistry of Drug Design
- Recent ADMET review articles

Each entry represents expected gradient correlation based on known mechanisms:
- Positive values: synergistic properties (shared mechanisms)
- Negative values: antagonistic properties (trade-offs)
- Zero: independent properties (no documented relationship)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# Comprehensive literature-derived property relationships
# Format: (task1, task2) -> expected_correlation
LITERATURE_RELATIONSHIPS = {
    # =========================================================================
    # PHYSICOCHEMICAL TRADE-OFFS (Well-documented anti-correlations)
    # =========================================================================

    # Solubility vs Lipophilicity - fundamental physicochemical trade-off
    # Reference: Lipinski's Rule of Five, Leeson & Springthorpe 2007
    ('ESOL', 'Lipophilicity'): -0.8,
    ('Solubility_AqSolDB', 'Lipophilicity'): -0.8,

    # Permeability requires lipophilicity but solubility opposes it
    ('ESOL', 'Caco2_Wang'): -0.4,
    ('ESOL', 'BBB_Martins'): -0.5,
    ('ESOL', 'BBBP'): -0.5,

    # =========================================================================
    # ADME CLUSTER (Positive correlations - shared transport mechanisms)
    # =========================================================================

    # BBB and Caco-2 - both measure membrane permeability
    ('BBBP', 'Caco2_Wang'): 0.6,
    ('BBB_Martins', 'Caco2_Wang'): 0.6,
    ('BBBP', 'BBB_Martins'): 0.9,  # Same endpoint, different datasets

    # Permeability and lipophilicity - passive diffusion
    ('BBBP', 'Lipophilicity'): 0.5,
    ('BBB_Martins', 'Lipophilicity'): 0.5,
    ('Caco2_Wang', 'Lipophilicity'): 0.4,

    # Bioavailability cluster
    ('HIA_Hou', 'Bioavailability_Ma'): 0.6,
    ('Caco2_Wang', 'HIA_Hou'): 0.5,
    ('Caco2_Wang', 'Bioavailability_Ma'): 0.4,

    # P-gp efflux reduces permeability
    ('Pgp_Broccatelli', 'BBBP'): -0.4,
    ('Pgp_Broccatelli', 'BBB_Martins'): -0.4,
    ('Pgp_Broccatelli', 'Caco2_Wang'): -0.3,

    # =========================================================================
    # CYP ENZYME CLUSTER (Positive correlations - similar binding sites)
    # =========================================================================

    # CYP enzymes share similar substrate preferences
    # Reference: Kirchmair et al. 2015, Nature Reviews Drug Discovery
    ('CYP2D6_Veith', 'CYP3A4_Veith'): 0.4,
    ('CYP2D6_Veith', 'CYP2C9_Veith'): 0.5,
    ('CYP2D6_Veith', 'CYP2C19_Veith'): 0.6,
    ('CYP2D6_Veith', 'CYP1A2_Veith'): 0.3,
    ('CYP3A4_Veith', 'CYP2C9_Veith'): 0.4,
    ('CYP3A4_Veith', 'CYP2C19_Veith'): 0.4,
    ('CYP3A4_Veith', 'CYP1A2_Veith'): 0.3,
    ('CYP2C9_Veith', 'CYP2C19_Veith'): 0.7,  # Same subfamily
    ('CYP2C9_Veith', 'CYP1A2_Veith'): 0.3,
    ('CYP2C19_Veith', 'CYP1A2_Veith'): 0.3,

    # =========================================================================
    # CLEARANCE CLUSTER (Positive correlations)
    # =========================================================================

    # Different clearance measurements are related
    ('Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ'): 0.7,
    ('Clearance_Hepatocyte_AZ', 'Half_Life_Obach'): -0.5,  # High clearance = short half-life
    ('Clearance_Microsome_AZ', 'Half_Life_Obach'): -0.5,

    # CYP inhibitors tend to have lower clearance
    ('CYP3A4_Veith', 'Clearance_Hepatocyte_AZ'): -0.3,
    ('CYP3A4_Veith', 'Clearance_Microsome_AZ'): -0.3,

    # =========================================================================
    # TOXICITY ENDPOINTS
    # =========================================================================

    # Tox21 nuclear receptor cluster - same receptor families
    ('Tox21_NR-AR', 'Tox21_NR-AR-LBD'): 0.8,
    ('Tox21_NR-ER', 'Tox21_NR-ER-LBD'): 0.8,
    ('Tox21_NR-AR', 'Tox21_NR-ER'): 0.4,  # Both steroid receptors
    ('Tox21_NR-AR-LBD', 'Tox21_NR-ER-LBD'): 0.4,

    # Tox21 stress response cluster
    ('Tox21_SR-ARE', 'Tox21_SR-HSE'): 0.5,
    ('Tox21_SR-ARE', 'Tox21_SR-MMP'): 0.4,
    ('Tox21_SR-HSE', 'Tox21_SR-MMP'): 0.4,
    ('Tox21_SR-ATAD5', 'Tox21_SR-p53'): 0.5,  # DNA damage response

    # hERG and lipophilicity - lipophilic compounds more likely to block hERG
    # Reference: Aronov 2005, Drug Discovery Today
    ('hERG', 'Lipophilicity'): 0.4,

    # AMES mutagenicity - reactive groups
    ('AMES', 'Carcinogens_Lagunin'): 0.5,

    # DILI - hepatotoxicity correlates with metabolism
    ('DILI', 'CYP3A4_Veith'): 0.3,

    # =========================================================================
    # BINDING AFFINITY RELATIONSHIPS
    # =========================================================================

    # BACE and CNS penetration - Alzheimer's drugs need BBB penetration
    ('BACE', 'BBBP'): 0.4,
    ('BACE', 'BBB_Martins'): 0.4,

    # HIV inhibitors - often lipophilic
    ('HIV', 'Lipophilicity'): 0.3,

    # =========================================================================
    # KNOWN TRADE-OFFS (Negative correlations)
    # =========================================================================

    # Selectivity vs promiscuity trade-off
    # Highly potent compounds often hit multiple targets
    ('BACE', 'hERG'): -0.2,  # Potent compounds may have off-target effects

    # Metabolic stability vs clearance
    # CYP inhibitors accumulate (lower clearance) but may have DDI issues
    ('CYP3A4_Veith', 'Half_Life_Obach'): 0.3,

    # Solubility vs metabolic stability
    # Lipophilic compounds are often more metabolically labile
    ('ESOL', 'Clearance_Hepatocyte_AZ'): -0.3,

    # =========================================================================
    # INDEPENDENT PROPERTIES (Near-zero expected correlation)
    # =========================================================================

    # Genotoxicity vs transport - different mechanisms
    ('AMES', 'BBBP'): 0.0,
    ('AMES', 'Caco2_Wang'): 0.0,

    # Electronic properties vs ADME (if using QM9)
    # These should be largely independent
}

# Confidence levels for relationships
RELATIONSHIP_CONFIDENCE = {
    'high': 0.9,    # Multiple independent studies
    'medium': 0.7,  # Well-established but fewer studies
    'low': 0.5,     # Mechanistically plausible, limited data
}

# Sources for key relationships
RELATIONSHIP_SOURCES = {
    ('ESOL', 'Lipophilicity'): [
        'Lipinski et al. 2001, Adv Drug Deliv Rev',
        'Leeson & Springthorpe 2007, Nat Rev Drug Discov',
    ],
    ('BBBP', 'Lipophilicity'): [
        'Pardridge 2012, NeuroRx',
        'Di et al. 2013, Drug Metab Dispos',
    ],
    ('hERG', 'Lipophilicity'): [
        'Aronov 2005, Drug Discov Today',
        'Waring et al. 2015, Nat Rev Drug Discov',
    ],
}


def get_literature_matrix(task_names: List[str]) -> np.ndarray:
    """
    Construct literature relationship matrix for given tasks.

    Args:
        task_names: List of task names (order determines matrix indices)

    Returns:
        K×K matrix where L[i,j] is expected correlation between tasks i and j
    """
    K = len(task_names)
    L = np.zeros((K, K))

    # Fill diagonal with 1 (perfect self-correlation)
    np.fill_diagonal(L, 1.0)

    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i >= j:
                continue

            # Check for direct relationship
            key = (task_i, task_j)
            key_rev = (task_j, task_i)

            if key in LITERATURE_RELATIONSHIPS:
                L[i, j] = LITERATURE_RELATIONSHIPS[key]
                L[j, i] = LITERATURE_RELATIONSHIPS[key]
            elif key_rev in LITERATURE_RELATIONSHIPS:
                L[i, j] = LITERATURE_RELATIONSHIPS[key_rev]
                L[j, i] = LITERATURE_RELATIONSHIPS[key_rev]
            else:
                # Check for partial matches (handle dataset naming variations)
                for lit_key, value in LITERATURE_RELATIONSHIPS.items():
                    t1, t2 = lit_key
                    # Check if task names contain the literature key names
                    if (t1 in task_i or task_i in t1) and (t2 in task_j or task_j in t2):
                        L[i, j] = value
                        L[j, i] = value
                        break
                    if (t2 in task_i or task_i in t2) and (t1 in task_j or task_j in t1):
                        L[i, j] = value
                        L[j, i] = value
                        break

    return L


def get_literature_tradeoffs_list() -> List[Tuple[str, str, float, str]]:
    """
    Get list of documented trade-offs with descriptions.

    Returns:
        List of (task1, task2, expected_correlation, description) tuples
    """
    tradeoffs = []

    descriptions = {
        ('ESOL', 'Lipophilicity'): 'Solubility-lipophilicity trade-off: hydrophilic groups increase solubility but decrease membrane permeability',
        ('BBBP', 'ESOL'): 'CNS penetration requires lipophilicity which opposes aqueous solubility',
        ('Pgp_Broccatelli', 'BBBP'): 'P-gp efflux actively removes compounds from brain, reducing BBB penetration',
        ('hERG', 'Lipophilicity'): 'Lipophilic compounds more likely to block hERG channel (cardiotoxicity)',
        ('Clearance_Hepatocyte_AZ', 'Half_Life_Obach'): 'High clearance leads to short half-life (inverse relationship)',
    }

    for (t1, t2), corr in LITERATURE_RELATIONSHIPS.items():
        if corr < -0.2:  # Trade-offs
            desc = descriptions.get((t1, t2), f'Mechanistic antagonism between {t1} and {t2}')
            tradeoffs.append((t1, t2, corr, desc))

    return tradeoffs


def get_literature_synergies_list() -> List[Tuple[str, str, float, str]]:
    """
    Get list of documented synergies with descriptions.

    Returns:
        List of (task1, task2, expected_correlation, description) tuples
    """
    synergies = []

    descriptions = {
        ('BBBP', 'Caco2_Wang'): 'Both measure membrane permeability via passive diffusion',
        ('CYP2D6_Veith', 'CYP2C19_Veith'): 'Same CYP subfamily with similar substrate preferences',
        ('Tox21_NR-AR', 'Tox21_NR-AR-LBD'): 'Same androgen receptor, different binding assays',
    }

    for (t1, t2), corr in LITERATURE_RELATIONSHIPS.items():
        if corr > 0.3:  # Synergies
            desc = descriptions.get((t1, t2), f'Mechanistic synergy between {t1} and {t2}')
            synergies.append((t1, t2, corr, desc))

    return synergies


def validate_gradient_matrix(
    G: np.ndarray,
    task_names: List[str],
    verbose: bool = True
) -> Dict:
    """
    Validate gradient matrix against literature expectations.

    Args:
        G: Gradient conflict matrix (K×K)
        task_names: List of task names
        verbose: Print detailed results

    Returns:
        Dict with validation metrics
    """
    from scipy import stats

    L = get_literature_matrix(task_names)

    # Extract off-diagonal elements
    K = len(task_names)
    mask = ~np.eye(K, dtype=bool)
    g_flat = G[mask]
    l_flat = L[mask]

    # Filter to pairs with documented relationships (non-zero in L)
    nonzero_mask = l_flat != 0
    g_documented = g_flat[nonzero_mask]
    l_documented = l_flat[nonzero_mask]

    # Compute correlation
    if len(g_documented) > 2:
        pearson_r, pearson_p = stats.pearsonr(g_documented, l_documented)
        spearman_r, spearman_p = stats.spearmanr(g_documented, l_documented)
    else:
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1

    # Compute sign agreement
    sign_agreement = np.mean(np.sign(g_documented) == np.sign(l_documented))

    # Identify matches and mismatches
    matches = []
    mismatches = []

    for i, task_i in enumerate(task_names):
        for j, task_j in enumerate(task_names):
            if i >= j:
                continue
            if L[i, j] == 0:
                continue

            g_val = G[i, j]
            l_val = L[i, j]

            if np.sign(g_val) == np.sign(l_val) and abs(g_val - l_val) < 0.3:
                matches.append((task_i, task_j, g_val, l_val))
            elif np.sign(g_val) != np.sign(l_val):
                mismatches.append((task_i, task_j, g_val, l_val))

    results = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'sign_agreement': sign_agreement,
        'n_documented_pairs': len(g_documented),
        'n_matches': len(matches),
        'n_mismatches': len(mismatches),
        'matches': matches,
        'mismatches': mismatches,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("LITERATURE VALIDATION RESULTS")
        print("=" * 60)
        print(f"Documented pairs analyzed: {len(g_documented)}")
        print(f"Pearson correlation: r = {pearson_r:.3f} (p = {pearson_p:.4f})")
        print(f"Spearman correlation: ρ = {spearman_r:.3f} (p = {spearman_p:.4f})")
        print(f"Sign agreement: {sign_agreement:.1%}")
        print(f"\nMatches (correct direction): {len(matches)}")
        print(f"Mismatches (wrong direction): {len(mismatches)}")

        if mismatches:
            print("\nKey mismatches:")
            for t1, t2, g, l in mismatches[:5]:
                print(f"  {t1} vs {t2}: gradient={g:.3f}, expected={l:.3f}")

    return results


if __name__ == '__main__':
    print("Literature Relationship Matrix")
    print("=" * 60)

    # Example task list
    tasks = [
        'ESOL', 'Lipophilicity', 'BBBP', 'Caco2_Wang',
        'CYP2D6_Veith', 'CYP3A4_Veith', 'hERG', 'AMES'
    ]

    L = get_literature_matrix(tasks)

    print("\nLiterature Matrix:")
    print("Tasks:", tasks)
    print(L)

    print("\n\nDocumented Trade-offs:")
    for t1, t2, corr, desc in get_literature_tradeoffs_list()[:5]:
        print(f"  {t1} vs {t2}: {corr:.2f}")
        print(f"    {desc}\n")

    print("\nDocumented Synergies:")
    for t1, t2, corr, desc in get_literature_synergies_list()[:5]:
        print(f"  {t1} vs {t2}: {corr:.2f}")
        print(f"    {desc}\n")
