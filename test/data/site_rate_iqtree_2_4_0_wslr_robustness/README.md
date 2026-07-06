# `-wslr` `.sitelh` robustness fixtures

These fixtures exercise numerical realities of real IQ-TREE `-wslr` `.sitelh`
output that the earlier synthetic fixtures never hit and that broke the loader
on real `+R6` data:

1. **`-inf` LnLW entries** — IQ-TREE writes `-inf` (= log 0) for categories
   whose weighted site likelihood underflowed to zero. These reconstruct to a
   category responsibility of exactly zero.
2. **Rows that do not sum to one** — on numerically-extreme sites IQ-TREE
   violates its own documented `.sitelh` invariant (`sum_k exp(LnLW_k) = exp(LnL)`),
   so `sum_k q_k` can drift below/above 1. IQ-TREE's own `.siteprob` hides this by
   renormalizing among categories; the loader mirrors that for non-`+I` models.

## Hand-built synthetic fixtures (deterministic)

`synthetic_r.{sitelh,iqtree}` — a non-`+I` `GTR+F+R2` `.sitelh` with `LnL = 0`
so `exp(LnLW_k)` reconstructs each `q_k` directly. It contains a clean row, an
`-inf` row, an under-summing row, and an `-inf`-plus-under-sum row. See the
header comment in the file for the per-site design.

`synthetic_ir.{sitelh,iqtree}` — a `GTR+F+I+R2` `.sitelh` with `-inf` in a
variable category, confirming the `+I` invariant deficit `p_i0 = 1 - sum_k q_k`
is preserved (kept, not renormalized away).

## `--alisim`-generated real fixture (genuine `-inf`)

`alisim_r.{sitelh,rate,iqtree}` is a 40-site excerpt (sites 85-124, renumbered
1-40) of a genuine IQ-TREE `-wslr` run on an `--alisim`-simulated alignment with
strong rate variation, so IQ-TREE emits real `-inf` LnLW on extreme sites. No
private data is involved. Reproduce the full run with the installed IQ-TREE 2.4.0
ARM64 release:

```sh
# random 80-taxon tree
iqtree2 -r 80 tree80.nwk
# simulate 3000 sites under GTR+F+R6 with widely separated category rates
iqtree2 --alisim sim80 -t tree80.nwk \
  -m "GTR{2,4,1,1.5,3}+F{0.3,0.2,0.2,0.3}+R6{0.5,0.000001,0.2,0.05,0.15,0.5,0.1,3,0.04,15,0.01,500}" \
  --branch-scale 6 --length 3000 --seqtype DNA --seed 31 -redo
# fit the same model (optimizing branch lengths on the fixed topology) with -wslr
iqtree2 -s sim80.phy -te tree80.nwk \
  -m "GTR{2,4,1,1.5,3}+F{0.3,0.2,0.2,0.3}+R6{0.5,0.000001,0.2,0.05,0.15,0.5,0.1,3,0.04,15,0.01,500}" \
  -wslr -wsr -pre fit80 --seed 1 -redo
```

The full `fit80.sitelh` has 105 `-inf` LnLW entries across 3000 sites; the
checked-in 40-site excerpt keeps four of them. The report writes the model as
`GTR+FU+R6` (fixed user frequencies); its `Category`/`Relative_rate`/`Proportion`
table drives the loader. The `.rate` cross-check passes on the excerpt.
