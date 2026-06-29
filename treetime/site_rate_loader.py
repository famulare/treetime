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

    IQ-TREE .rate formats (tab-separated; header detected from column names):

    Format A — partitioned model (from -p):
        Part  Site  Rate  Cat  C_Rate
        1     1     0.83  2    0.84
        ...
    Format B — single-model (from -m):
        Site  Rate  Cat
        1     0.123 1
        ...

    Lines starting with '#' are comments and are skipped. The header line is
    detected by looking for non-numeric first token (e.g. 'Part' or 'Site').
    Rate column index is auto-detected from the header.
    """
    sites, rates = [], []
    rate_col = None  # 0-based column index for the Rate value

    with open(rate_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split('\t') if '\t' in line else line.split()
            if not cols:
                continue
            # Detect header row (first token is non-numeric)
            try:
                float(cols[0])
            except ValueError:
                # This is the header — find the 'Rate' column
                lower = [c.lower() for c in cols]
                if 'rate' in lower:
                    rate_col = lower.index('rate')
                    site_col = lower.index('site') if 'site' in lower else 1
                else:
                    rate_col = 1  # fallback
                    site_col = 0
                continue
            # Data row
            if rate_col is None:
                # No header seen yet — assume format B (Site Rate ...)
                rate_col = 1
                site_col = 0
            try:
                sites.append(int(cols[site_col]))
                rates.append(float(cols[rate_col]))
            except (ValueError, IndexError):
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

def _parse_partition_rates_from_nex(nex_file, K):
    """Parse per-partition FreeRate category rates from an IQ-TREE best_model.nex.

    Returns dict {partition_id (int): np.array of shape (K,) with raw category rates}
    or empty dict if parsing fails.
    """
    try:
        nex = Path(nex_file).read_text()
    except (OSError, TypeError):
        return {}
    pattern = re.compile(r'R\d+\{([^}]+)\}[^:]*:\s*pos(\d+)\{([^}]+)\}')
    result = {}
    for m in pattern.finditer(nex):
        params_str, part_id = m.group(1), int(m.group(2))
        vals = [float(x) for x in params_str.split(',')]
        rates = np.array(vals[1::2], dtype=float)
        if len(rates) == K:
            result[part_id] = rates
    return result


def load_site_rate_posteriors(siteprob_file, iqtree_file, best_model_nex=None):
    """Parse IQ-TREE .siteprob + model info → per-site category posteriors.

    Handles both unpartitioned and partitioned IQ-TREE outputs.

    For **partitioned** models (detected when .siteprob has a 'Part' column), each
    partition has its own category-rate parameters. This function builds a per-site
    rate matrix `r_ik` of shape (L, K) using each site's partition-specific rates,
    which is required for `prob_t_profiles_mixture`. Pass `best_model_nex` (the
    IQ-TREE `*.best_model.nex` file) to enable this. Without it, falls back to a
    global average r_k.

    Parameters
    ----------
    siteprob_file : str or Path
    iqtree_file : str or Path
        IQ-TREE .iqtree report (for .iqtree category table; single-model runs).
    best_model_nex : str or Path or None
        IQ-TREE best_model.nex (for per-partition rates; partitioned runs).
        If None, tries to auto-detect alongside iqtree_file.

    Returns
    -------
    p_ik : np.ndarray, shape (L, K)
        Posterior probability that site i belongs to category k.
    r_ik : np.ndarray, shape (L, K) or (K,)
        Per-site category rate multipliers. Shape (L, K) for partitioned models
        (each site uses its partition's rates); shape (K,) for unpartitioned.
        prob_t_profiles_mixture accepts both shapes.
    meta : dict
        'K', 'partitioned', 'r_k_mean' (global mean category rates).
    """
    siteprob_file = Path(siteprob_file)
    iqtree_file = Path(iqtree_file)

    # Auto-detect best_model.nex alongside iqtree_file
    if best_model_nex is None:
        candidate = Path(str(iqtree_file).replace('.iqtree', '.best_model.nex'))
        if candidate.exists():
            best_model_nex = candidate

    # Parse .siteprob — handles both partitioned (Part Site p1..pK) and
    # unpartitioned (Site p1..pK) formats. Detect from header.
    rows = []
    partition_ids = []
    prob_start_col = 1
    has_part_col = False

    with open(siteprob_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            try:
                float(parts[0])
            except ValueError:
                lower = [c.lower() for c in parts]
                if lower[0] == 'part' or (len(lower) > 1 and lower[1] == 'site'):
                    prob_start_col = 2
                    has_part_col = True
                else:
                    prob_start_col = 1
                continue
            try:
                if has_part_col:
                    partition_ids.append(int(parts[0]))
                probs = [float(x) for x in parts[prob_start_col:]]
                rows.append(probs)
            except (ValueError, IndexError):
                continue

    if not rows:
        raise ValueError(f"No category posteriors parsed from {siteprob_file}")

    p_ik = np.array(rows, dtype=float)
    L, K = p_ik.shape
    row_sums = p_ik.sum(axis=1, keepdims=True)
    p_ik = p_ik / np.where(row_sums > 0, row_sums, 1.0)

    # Build per-site rate matrix
    is_partitioned = has_part_col and len(partition_ids) == L

    if is_partitioned and best_model_nex is not None:
        part_rates = _parse_partition_rates_from_nex(best_model_nex, K)
        if len(part_rates) >= 1:
            # Build (L, K) matrix: each row = rates for that site's partition
            r_ik = np.zeros((L, K), dtype=float)
            for i, pid in enumerate(partition_ids):
                if pid in part_rates:
                    r_ik[i] = part_rates[pid]
                else:
                    # Fallback: mean across known partitions
                    r_ik[i] = np.mean(list(part_rates.values()), axis=0)
            # Normalize: each partition's rates are already ~mean-1 within partition;
            # global mean should be near 1 after normalization
            global_mean = r_ik.mean()
            if global_mean > 0:
                r_ik /= global_mean
            logger.info(
                f"Built per-site r_ik ({L}×{K}) from {len(part_rates)} partitions; "
                f"global mean={float(r_ik.mean()):.4f}"
            )
            meta = {'K': K, 'partitioned': True, 'r_k_mean': r_ik.mean(axis=0).tolist()}
            return p_ik, r_ik, meta
        else:
            logger.warning("Could not parse partition rates from best_model.nex; using global fallback")

    # Fallback: single global r_k from .iqtree report
    r_k, w_k = _extract_freerate_categories(iqtree_file, K)
    logger.info(
        f"Loaded {L}×{K} site posteriors (unpartitioned); r_k={np.round(r_k,4).tolist()}"
    )
    meta = {'K': K, 'partitioned': False, 'r_k_mean': r_k.tolist()}
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
