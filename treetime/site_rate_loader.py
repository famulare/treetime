"""site_rate_loader.py — load IQ-TREE per-site FreeRate estimates into GTR_site_specific.

Public API
----------
load_site_rates(rate_file, iqtree_file=None, invariant_policy='include', normalize=True)
    Parse IQ-TREE .rate file → (rates_array shape (L,), metadata_dict).

load_site_rate_posteriors(siteprob_file, iqtree_file)
    Parse .siteprob + .iqtree → (p_ik (L,K), r_k (K,), metadata_dict).

build_site_specific_gtr(rates, base_gtr, seq_len=None)
    Build GTR_site_specific for Tier A: _mu = rates (mean-1), Pi/W from base_gtr.

build_mixture_gtr(base_gtr, seq_len)
    Build GTR_site_specific for Tier B: _mu = ones(L). Rates supplied via r_k at eval time.
"""

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .rate file parser
# ---------------------------------------------------------------------------

def _parse_rate_file(rate_file):
    """Return (sites, rates) arrays from an IQ-TREE .rate file.

    IQ-TREE .rate format (columns vary by version; first two always site, rate):
        Site    Rate    Cat (optional)
        1       0.123   1
        ...
    Lines starting with 'Site' or '#' are skipped.
    """
    sites, rates = [], []
    with open(rate_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('site'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                sites.append(int(parts[0]))
                rates.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(sites, dtype=int), np.array(rates, dtype=float)


def _extract_partition_weights_from_iqtree(iqtree_file, n_sites_total):
    """Extract per-partition site counts from an IQ-TREE .iqtree report.

    Returns dict {partition_name: fraction_of_total} or empty dict if not parseable.
    Used to detect and correct per-partition normalization in .rate files.
    """
    weights = {}
    if not Path(iqtree_file).exists():
        return weights

    with open(iqtree_file) as fh:
        text = fh.read()

    # Look for partition table lines like:  pos1   302   ...
    # Various IQ-TREE versions format this differently; try several patterns.
    patterns = [
        r'(\w+)\s+(\d+)\s+\d+\s+-\d+',          # name  n_sites  n_seqs  -logL
        r'Partition\s+(\w+):\s+(\d+)\s+sites',   # Partition pos1: 302 sites
        r'(\w+)\s+.*?(\d+)\s+sites',             # loose fallback
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            name, count = m.group(1), int(m.group(2))
            if 0 < count < n_sites_total:
                weights[name] = count / n_sites_total
        if weights:
            break

    # Sanity: weights should sum to ~1 (tight tolerance to catch regex false positives)
    if weights and abs(sum(weights.values()) - 1.0) > 0.03:
        weights = {}

    return weights


# ---------------------------------------------------------------------------
# Public: load_site_rates
# ---------------------------------------------------------------------------

def load_site_rates(rate_file, iqtree_file=None, invariant_policy='include', normalize=True):
    """Parse IQ-TREE .rate file → normalized per-nt-site rate vector.

    Parameters
    ----------
    rate_file : str or Path
        IQ-TREE .rate output file (one row per alignment site).
    iqtree_file : str or Path or None
        IQ-TREE .iqtree report. When provided: detect per-partition normalization
        and correct to a single global mean-1 vector.
    invariant_policy : {'include', 'exclude', 'epsilon'}
        How to handle near-zero (r < 1e-6) sites.
        'include'  — keep as-is (default; model can treat them as invariant).
        'exclude'  — set to NaN; caller must handle (used for ablation tests).
        'epsilon'  — replace with 1e-6 to avoid numerical degeneracy.
    normalize : bool
        If True (default), divide all rates by their mean so mean(rates) = 1.

    Returns
    -------
    rates : np.ndarray, shape (L,)
        Per-site substitution rate multipliers, mean ≈ 1.0.
    meta : dict
        Diagnostic metadata: 'raw_mean', 'norm_mean', 'n_sites',
        'n_invariant', 'per_partition_correction_applied'.
    """
    rate_file = Path(rate_file)
    sites, rates = _parse_rate_file(rate_file)
    L = len(rates)
    if L == 0:
        raise ValueError(f"No rates parsed from {rate_file}")

    raw_mean = float(rates.mean())
    logger.info(f"Loaded {L} site rates from {rate_file.name}; raw mean={raw_mean:.4f}")

    # Detect per-partition normalization situation (diagnostic only).
    # NOTE: in both the per-partition and the global cases the fix is identical —
    # divide by raw_mean to produce a mean-1 vector. The partition weights are
    # parsed for logging/provenance; they do NOT change the normalization arithmetic.
    partition_structure_detected = False
    if iqtree_file is not None and abs(raw_mean - 1.0) > 0.05:
        pw = _extract_partition_weights_from_iqtree(iqtree_file, L)
        if pw:
            logger.info(
                f"Partition structure detected {pw}; raw mean {raw_mean:.4f} — "
                "will normalize globally (divide by raw_mean)."
            )
            partition_structure_detected = True
        else:
            logger.warning(
                f"Raw mean {raw_mean:.4f} != 1.0; applying global normalization "
                "(could not parse partition weights from .iqtree)."
            )

    if normalize and raw_mean > 0:
        rates = rates / raw_mean

    # Apply invariant policy
    n_inv = int((rates < 1e-6).sum())
    if invariant_policy == 'exclude':
        rates = rates.astype(float)
        rates[rates < 1e-6] = np.nan
    elif invariant_policy == 'epsilon':
        rates = np.where(rates < 1e-6, 1e-6, rates)

    norm_mean = float(np.nanmean(rates))
    logger.info(
        f"After normalization: mean={norm_mean:.6f}, n_invariant(r<1e-6)={n_inv}"
    )

    meta = {
        'raw_mean': raw_mean,
        'norm_mean': norm_mean,
        'n_sites': L,
        'n_invariant': n_inv,
        'partition_structure_detected': partition_structure_detected,
    }
    return rates, meta


# ---------------------------------------------------------------------------
# Public: load_site_rate_posteriors
# ---------------------------------------------------------------------------

def load_site_rate_posteriors(siteprob_file, iqtree_file):
    """Parse IQ-TREE .siteprob + .iqtree → per-site category posteriors.

    IQ-TREE .siteprob format (space-separated):
        Site  p1  p2  ... pK
        1     0.1 0.7 ... 0.2
        ...

    Category rates r_k are extracted from the .iqtree report.

    Parameters
    ----------
    siteprob_file : str or Path
    iqtree_file : str or Path

    Returns
    -------
    p_ik : np.ndarray, shape (L, K)
        Posterior probability that site i belongs to category k.
    r_k : np.ndarray, shape (K,)
        Category rate multipliers (r_0 = 0 for invariant class if +I model).
    meta : dict
        'K', 'has_invariant', 'r_k', 'w_k' (prior weights from .iqtree).
    """
    siteprob_file = Path(siteprob_file)
    iqtree_file = Path(iqtree_file)

    # Parse .siteprob
    rows = []
    with open(siteprob_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.lower().startswith('site') or line.startswith('#'):
                continue
            parts = line.split()
            try:
                # First column is site index, rest are probabilities
                probs = [float(x) for x in parts[1:]]
                rows.append(probs)
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No category posteriors parsed from {siteprob_file}")

    p_ik = np.array(rows, dtype=float)  # (L, K)
    L, K = p_ik.shape

    # Normalize rows (should sum to 1, but floating point)
    row_sums = p_ik.sum(axis=1, keepdims=True)
    p_ik = p_ik / np.where(row_sums > 0, row_sums, 1.0)

    # Parse category rates from .iqtree
    r_k, w_k = _extract_freerate_categories(iqtree_file, K)
    has_invariant = (r_k[0] == 0.0) if len(r_k) > 0 else False

    logger.info(
        f"Loaded {L}×{K} site posteriors; "
        f"r_k={np.round(r_k, 4).tolist()}; has_invariant={has_invariant}"
    )
    meta = {'K': K, 'has_invariant': has_invariant, 'r_k': r_k.tolist(), 'w_k': w_k.tolist()}
    return p_ik, r_k, meta


def _extract_freerate_categories(iqtree_file, K):
    """Extract FreeRate category rates and weights from .iqtree report.

    Returns (r_k, w_k) arrays of length K.
    Falls back to uniform spacing if parsing fails.
    """
    r_k = np.zeros(K, dtype=float)
    w_k = np.ones(K, dtype=float) / K

    with open(iqtree_file) as fh:
        text = fh.read()

    # Pattern: "Category  Rel_rate  Proportion" table or FreeRate parameters
    # IQ-TREE writes lines like:  1    0.0530    0.0641
    #                              2    0.2814    0.1234
    cat_pattern = re.findall(
        r'^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$', text, re.MULTILINE
    )
    parsed = [(int(c), float(r), float(w)) for c, r, w in cat_pattern if 0 < float(w) < 1]

    if len(parsed) >= K:
        # Take first K entries (may include invariant as category 0 in +I+R models)
        parsed = parsed[:K]
        r_k = np.array([x[1] for x in parsed])
        w_k = np.array([x[2] for x in parsed])
        # Normalize weights
        w_k = w_k / w_k.sum()
    else:
        # Do not fabricate rates — raise so the caller knows to supply a .iqtree file.
        raise ValueError(
            f"Could not parse {K} FreeRate categories from {iqtree_file}. "
            "Provide the .iqtree report file alongside the .siteprob file, or "
            "supply category rates and weights directly via categories_file."
        )

    return r_k, w_k


# ---------------------------------------------------------------------------
# Public: build_site_specific_gtr (Tier A)
# ---------------------------------------------------------------------------

def build_site_specific_gtr(rates, base_gtr, seq_len=None):
    """Construct GTR_site_specific for Tier A from normalized rates + fitted scalar GTR.

    Parameters
    ----------
    rates : np.ndarray, shape (L,)
        Mean-1 per-site rate multipliers from load_site_rates().
    base_gtr : GTR
        A fitted scalar GTR providing Pi (shape (n,)) and W (shape (n,n)).
        NEVER pass a default unfitted GTR — Pi/W must come from actual data.
    seq_len : int or None
        Alignment length. Inferred from len(rates) if None.

    Returns
    -------
    GTR_site_specific with:
        _mu  = rates  (shape (L,), mean ≈ 1)
        _Pi  = base_gtr.Pi broadcast to (n, L)
        _W   = base_gtr.W
    assign_rates() automatically calls _eig() and _make_expQt_interpolator().
    """
    from .gtr_site_specific import GTR_site_specific

    L = int(len(rates)) if seq_len is None else int(seq_len)
    if len(rates) != L:
        raise ValueError(f"len(rates)={len(rates)} != seq_len={L}")

    mean_r = float(np.nanmean(rates))
    if abs(mean_r - 1.0) > 0.1:
        logger.warning(
            f"build_site_specific_gtr: rate mean={mean_r:.4f} (expected ~1.0); "
            "pass normalize=True to load_site_rates() first."
        )

    gtr_ss = GTR_site_specific(seq_len=L, alphabet=base_gtr.alphabet)
    # assign_rates broadcasts Pi if it's 1-D (scalar GTR Pi shape=(n,))
    gtr_ss.assign_rates(mu=rates, pi=base_gtr.Pi, W=base_gtr.W)
    # assign_rates calls _eig() + _make_expQt_interpolator() automatically
    logger.info(
        f"Built GTR_site_specific (Tier A): L={L}, "
        f"mu mean={float(gtr_ss.mu.mean()):.4f}, Pi shape={gtr_ss.Pi.shape}"
    )
    return gtr_ss


# ---------------------------------------------------------------------------
# Public: build_mixture_gtr (Tier B)
# ---------------------------------------------------------------------------

def build_mixture_gtr(base_gtr, seq_len):
    """Construct GTR_site_specific for Tier B with flat _mu = ones(L).

    For Tier B, site-rate variation is encoded via p_ik / r_k at likelihood evaluation
    time (in prob_t_profiles_mixture). The GTR must have _mu = ones(L) so that
    _expQt(t * r_k) = exp(t * r_k * Q)  — NOT exp(t * r_k * r_hat_i * Q).

    Parameters
    ----------
    base_gtr : GTR
        Fitted scalar GTR providing Pi and W.
    seq_len : int
        Alignment length L.

    Returns
    -------
    GTR_site_specific with _mu = ones(L), Pi/W from base_gtr.
    """
    from .gtr_site_specific import GTR_site_specific

    L = int(seq_len)
    gtr_b = GTR_site_specific(seq_len=L, alphabet=base_gtr.alphabet)
    gtr_b.assign_rates(mu=np.ones(L), pi=base_gtr.Pi, W=base_gtr.W)
    logger.info(
        f"Built GTR_site_specific (Tier B / mixture): L={L}, "
        f"mu all ones (rate variation via category mixture)"
    )
    return gtr_b
