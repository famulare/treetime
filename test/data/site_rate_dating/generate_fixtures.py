#!/usr/bin/env python3
"""Generate synthetic test fixtures for site-rate-dating tests.

Creates in the same directory:
  tiny.nwk          — 10-tip tree with known topology and dates 1977-2020
  tiny.fasta         — 50-site alignment simulated under mixed site rates
  tiny_dates.tsv     — strain → date table (tab-separated)
  tiny.rate          — synthetic .rate file (50 sites, 3-pattern: slow/medium/fast)
  tiny.siteprob      — synthetic .siteprob file (50 sites × 3 categories)
  tiny_categories.tsv — FreeRate category table for Tier B tests

Site-rate pattern:
  sites 1-15  (codon pos 1): category 1, rate 0.20  (conserved)
  sites 16-35 (codon pos 2): category 2, rate 0.80  (medium)
  sites 36-50 (codon pos 3): category 3, rate 3.50  (fast/saturated)
  → global mean ≈ 1.0 (weighted)

Run:  python test/data/site_rate_dating/generate_fixtures.py
from the treetime repo root, or just:  python generate_fixtures.py
from this directory.
"""
import os, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Tiny tree (Newick, 10 tips, known dates)
# ---------------------------------------------------------------------------
NWK = (
    "((((A_1977:0.015,B_1981:0.020):0.008,(C_1989:0.025,D_1993:0.018):0.006):0.010,"
    "((E_2001:0.030,F_2005:0.022):0.005,(G_2010:0.035,H_2013:0.028):0.007):0.004):0.012,"
    "(I_2017:0.040,J_2020:0.038):0.003);"
)

DATES = {
    "A_1977": "1977-06-01",
    "B_1981": "1981-03-15",
    "C_1989": "1989-09-01",
    "D_1993": "1993-04-20",
    "E_2001": "2001-07-01",
    "F_2005": "2005-11-15",
    "G_2010": "2010-02-28",
    "H_2013": "2013-08-01",
    "I_2017": "2017-05-01",
    "J_2020": "2020-01-15",
}

# ---------------------------------------------------------------------------
# 2. Site-rate pattern (50 sites, 3 categories)
# ---------------------------------------------------------------------------
L = 50
TIPS = list(DATES.keys())
N = len(TIPS)

# Per-site rates: [0.20]*15 + [0.80]*20 + [3.50]*15, global mean ≈ 1.0
rates = np.array([0.20]*15 + [0.80]*20 + [3.50]*15)
assert len(rates) == L
global_mean = rates.mean()
rates_norm = rates / global_mean  # normalize to mean=1

# Category labels (1-based for IQ-TREE convention)
# category 1 = rate 0.20/mean, category 2 = 0.80/mean, category 3 = 3.50/mean
r_k = np.array([0.20, 0.80, 3.50]) / global_mean  # category rates
w_k = np.array([15, 20, 15]) / L                    # prior weights

# Per-site posteriors: concentrated on the true category (0.9/0.05/0.05 split)
HIGH, LOW = 0.90, 0.05
p_ik = np.zeros((L, 3))
p_ik[:15, 0]  = HIGH; p_ik[:15, 1]  = LOW; p_ik[:15, 2]  = LOW
p_ik[15:35, 0] = LOW; p_ik[15:35, 1] = HIGH; p_ik[15:35, 2] = LOW
p_ik[35:, 0]   = LOW; p_ik[35:, 1]   = LOW; p_ik[35:, 2]   = HIGH

# ---------------------------------------------------------------------------
# 3. Simulate alignment with SeqGen
# ---------------------------------------------------------------------------
try:
    import sys; sys.path.insert(0, str(Path(__file__).parents[3]))
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    from treetime.seqgen import SeqGen
    from treetime import GTR
    from Bio import Phylo
    from io import StringIO

    # Build a site-specific GTR matching the rate pattern
    from treetime.gtr_site_specific import GTR_site_specific
    from treetime.site_rate_loader import build_site_specific_gtr

    base_gtr = GTR.standard('Jukes-Cantor', alphabet='nuc')
    gtr_ss = build_site_specific_gtr(rates_norm, base_gtr, seq_len=L)

    tree = Phylo.read(StringIO(NWK), 'newick')
    sg = SeqGen(L, tree=tree, gtr=gtr_ss)
    root_seq = ''.join(np.random.RandomState(42).choice(list('ACGT'), size=L))
    sg.evolve(root_seq=root_seq)
    aln = sg.get_aln()
    records = [SeqRecord(Seq(str(r.seq)), id=r.id, description='') for r in aln]
    print(f"Simulated {len(records)} sequences × {len(records[0].seq)} sites")
except Exception as e:
    # Fallback: write dummy alignment (ACGT repeated) for loader/parser tests
    print(f"SeqGen simulation failed ({e}); writing placeholder alignment")
    np.random.seed(42)
    records = []
    for tip in TIPS:
        seq = ''.join(np.random.choice(list('ACGT'), size=L))
        records.append(SeqRecord(Seq(seq), id=tip, description=''))

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
(HERE / "tiny.nwk").write_text(NWK + "\n")
print(f"wrote {HERE}/tiny.nwk")

with open(HERE / "tiny_dates.tsv", "w") as f:
    f.write("name\tdate\n")
    for name, date in DATES.items():
        f.write(f"{name}\t{date}\n")
print(f"wrote {HERE}/tiny_dates.tsv")

SeqIO.write(records, str(HERE / "tiny.fasta"), "fasta")
print(f"wrote {HERE}/tiny.fasta ({len(records)} seqs × {L} sites)")

# .rate file (IQ-TREE format: Site Rate [Cat])
with open(HERE / "tiny.rate", "w") as f:
    f.write("Site\tRate\tCat\n")
    for i, r in enumerate(rates):  # write raw (un-normalized) rates
        cat = 1 if i < 15 else (2 if i < 35 else 3)
        f.write(f"{i+1}\t{r:.6f}\t{cat}\n")
print(f"wrote {HERE}/tiny.rate (raw mean={rates.mean():.4f})")

# .siteprob file (IQ-TREE format: Site p1 p2 p3)
with open(HERE / "tiny.siteprob", "w") as f:
    f.write("Site\tp1\tp2\tp3\n")
    for i in range(L):
        f.write(f"{i+1}\t{p_ik[i,0]:.4f}\t{p_ik[i,1]:.4f}\t{p_ik[i,2]:.4f}\n")
print(f"wrote {HERE}/tiny.siteprob")

# categories TSV (for Tier B)
with open(HERE / "tiny_categories.tsv", "w") as f:
    f.write("category\trate\tprior_weight\n")
    for k, (rk, wk) in enumerate(zip(r_k, w_k)):
        f.write(f"{k+1}\t{rk:.6f}\t{wk:.6f}\n")
print(f"wrote {HERE}/tiny_categories.tsv (r_k={r_k.round(4).tolist()})")
