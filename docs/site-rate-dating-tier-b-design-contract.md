# Design contract: production Tier B ELBO site-rate-aware dating

Status: proposed implementation contract for an upstream TreeTime pull request.

This contract is written against TreeTime 0.12.1 and the IQ-TREE 2.4 output
writers. Normative terms such as MUST, SHOULD, and MAY are used in their usual
requirements sense.

## 1. Outcome

The pull request will add a general, validated handoff from externally estimated
site-rate models into TreeTime's fixed-topology date inference.

It will support:

1. **Tier A, posterior mean**: one fixed relative rate per alignment site.
2. **Tier B, frozen-responsibility ELBO**: per-site posterior weights over rate
   categories, evaluated as one empirical-Bayes variational M-step.

The IQ-TREE loader will be one adapter into a generic in-memory site-rate model.
TreeTime's date message-passing algorithm will remain unchanged.

The PR is successful when the feature is correct for unpartitioned and
partitioned inputs, has explicit statistical semantics, fails safely on inputs
that cannot be represented, runs in normal TreeTime workflows without hidden
state on tree nodes, and is documented and tested independently of any
downstream dataset.

## 2. Scope and non-goals

### 2.1 In scope

- Dense nucleotide and amino-acid alignments.
- A fixed topology shared by the external rate analysis and TreeTime.
- Tier A input from any IQ-TREE `.rate` model that emits one scalar rate per
  site.
- Tier B input from IQ-TREE FreeRate (`+R`) models.
- Edge-equal and edge-linked-proportional IQ-TREE partition models.
- Interleaved or contiguous partition charsets.
- Different numbers of rate categories in different partitions.
- An explicitly represented category with rate zero in the canonical model.
- A shared TreeTime substitution generator `Q` with site-specific scalar rate
  multipliers.
- Marginal ancestral-state reconstruction with alignment compression disabled.

### 2.2 Explicit non-goals

- Full Bayesian dating.
- Re-estimating category posteriors during TreeTime optimization.
- The exact FreeRate tree likelihood in which one latent category `z_i` is
  integrated jointly across every branch.
- Edge-unlinked partition models, which do not have one partition speed scalar
  representable by TreeTime's shared branch-time process.
- Partition-specific substitution matrices in the TreeTime likelihood.
- IQ-TREE mixture-class outputs from `-wspm` or `-wspmr`; Tier B consumes
  rate-category posteriors from `-wspr`.
- IQ-TREE `+I+R` input until the relationship between its invariant component
  and `-wspr` columns is fixture-verified; the adapter fails rather than
  guessing this mapping.
- Codon-state substitution models.
- Sparse VCF input in the first upstream PR.
- Topology inference or topology changes attributable to site-rate data.
- Pattern compression for site-specific rates.

Exact whole-tree FreeRate inference is outside both the architecture and the
performance expectations of TreeTime. TreeTime reduces each branch's sequence
evidence to a one-dimensional branch-length distribution before date message
passing; it does not carry site-by-category state through the date engine. This
contract does not treat exact joint category integration as a future TreeTime
deliverable. If it is required, it belongs in a different inference engine.

Tier B instead performs a frozen-responsibility empirical-Bayes handoff:
IQ-TREE supplies one posterior responsibility vector per site, and TreeTime
optimizes dates conditional on those fixed responsibilities.

## 3. Scientific and statistical contract

### 3.1 Units and identifiability

Let:

- `mu` be TreeTime's global molecular clock in substitutions/site/year;
- `delta_t_b` be the duration of branch `b` in years;
- `Q` be a shared substitution generator normalized to unit mean rate;
- `g(i)` be the partition containing alignment site `i`;
- `s_g` be the relative speed of partition `g`;
- `rho_gk` be the within-partition rate of category `k`;
- `p_ik` be the fixed posterior probability that site `i` belongs to category
  `k`.

IQ-TREE category rates are normalized within their partition. The global
category multiplier used by TreeTime is:

```text
a_ik = s_g(i) * rho_g(i),k
```

The model MUST then apply one global normalization constant:

```text
c = (1 / L_active) * sum_i sum_k p_ik * a_ik
r_ik = a_ik / c
```

so that:

```text
(1 / L_active) * sum_i sum_k p_ik * r_ik = 1.
```

For a mean-only input, replace `sum_k p_ik * a_ik` with the mapped and
partition-scaled posterior mean rate for site `i`.

This normalization is mandatory. A CLI option to disable it would make the
reported global clock depend on an arbitrary external scale and MUST NOT be
part of the upstream interface.

The transition length for site `i`, category `k`, and branch `b` is:

```text
ell_bik = mu * delta_t_b * r_ik.
```

The site-rate model contains relative multipliers only. It MUST NOT contain or
absorb `mu`.

### 3.2 Tier A objective

Define the posterior mean global rate:

```text
r_hat_i = sum_k p_ik * r_ik.
```

Tier A constructs a `GTR_site_specific` with `_mu = r_hat`. For a parent/child
profile pair at branch `b`, it uses TreeTime's existing site-specific marginal
likelihood:

```text
log L_A,b(ell_b) =
    sum_i m_i * gap_weight_i
          * log P_Q(profile_parent_i, profile_child_i; ell_b * r_hat_i).
```

`m_i` is the uncompressed site multiplicity and is normally one.

### 3.3 Tier B objective

Tier B uses IQ-TREE's whole-tree posterior category probabilities as fixed
empirical-Bayes responsibilities. Define the category-specific profile
transition probability for one branch:

```text
P_bik(ell_b) =
    profile_child_i^T exp(Q * ell_b * r_ik) profile_parent_i
```

The date-dependent part of the frozen-responsibility evidence lower bound
(ELBO) is:

```text
ELBO_B =
    sum_b sum_i m_i * gap_weight_i
        * sum_k p_ik * log(P_bik(ell_b)).
```

For category prior weights `w_ik`, the complete per-site variational bound also
contains:

```text
sum_k p_ik * (log(w_ik) - log(p_ik)).
```

That term is constant while TreeTime optimizes dates, so it does not enter a
branch-length distribution. The same `p_ik` weights category-specific log
transitions on every branch; categories are never remixed independently within
branches.

This is the normative and only Tier B objective in this contract. The current
local objective
`sum_b,i log(sum_k p_ik * P_bik)` is incoherent with a site-level latent
FreeRate category and MUST be removed rather than retained as an experimental
or compatibility mode.

When the imported responsibilities and the evaluated likelihood belong to the
same latent-category model, this is the standard frozen-E-step variational
objective. It touches the marginal likelihood and has the same first
derivative at the guide parameters when `p_ik` is the guide posterior. It omits
posterior adaptation and the corresponding score-variance contribution to
curvature. Consequently, Tier B point estimates are locally principled, but
ordinary likelihood-based uncertainty MUST NOT be presented as exact
FreeRate-calibrated uncertainty.

The CLI value will be `posterior-elbo`. The unreleased local value
`posterior-mixture` will not receive a compatibility alias.

### 3.4 Ancestral profiles and preconditioning

Tier B MUST NOT infer ancestral profiles with a flat-rate GTR.

Ancestral reconstruction and preliminary branch optimization will use the Tier
A posterior-mean `GTR_site_specific`. Tier B is applied when constructing the
branch-length distributions used by date inference. This makes the scaffold
state explicit:

- Tier A rates define the approximate ancestral profile pair at each branch.
- Tier B weights and category rates define the branch-length likelihood
  conditional on those profiles.

This is still an approximation. The limitation MUST be documented, and a test
MUST verify that one-hot category posteriors give the same result as a
site-specific fixed-rate model.

### 3.5 ELBO calibration boundary

Tier B freezes responsibilities estimated on an IQ-TREE guide tree. Its main
risks are therefore guide-tree anchoring and underestimated uncertainty, not
branchwise category incoherence.

Before merge, simulations MUST estimate responsibilities on guide trees with
branch scales below, at, and above the generating scale. Tier B must then be
tested for:

- clock and node-date bias;
- sensitivity to guide-tree scale;
- interval coverage under concentrated and diffuse responsibilities; and
- convergence toward Tier A as posterior rate variance approaches zero.

No exact FreeRate date engine is required or proposed. Small direct algebra
tests of the ELBO identity are sufficient to validate the variational
calculation.

## 4. Canonical in-memory model

Introduce one analysis-owned object, provisionally:

```python
SiteRateModel(
    mean_rates,            # (L,), finite, nonnegative, global mean 1
    posterior_weights,     # optional (L, Kmax), finite, nonnegative
    category_rates,        # optional (L, Kmax), finite, nonnegative
    category_prior_weights, # optional (L, Kmax), finite, nonnegative
    category_mask,         # optional (L, Kmax), valid categories per site
    partition_index,       # (L,), diagnostic/provenance
    metadata,              # source, normalization, parser diagnostics
)
```

The exact class name is not normative. Its contracts are:

- Alignment site is the first axis and always means the original, global,
  zero-based alignment coordinate.
- `posterior_weights` and `category_rates` are padded for partitions with
  fewer categories; padded posterior weights are zero.
- `category_prior_weights` records the IQ-TREE category prior used to validate
  and audit the variational handoff; it is not needed in the date-dependent
  branch objective.
- Valid posterior and category-prior rows each sum to one within a strict
  tolerance.
- `mean_rates` equals
  `sum_k posterior_weights * category_rates` when posterior data are present.
- The mean of `mean_rates` over active sites equals one within tolerance.
- Arrays are validated once and treated as immutable thereafter.
- Source metadata records partition names, original category counts,
  partition speeds, normalization constant, and source paths.

The model belongs to `TreeTime`/`ClockTree`, not to individual nodes. The PR
MUST remove the current `node.site_rate_posteriors` handshake. New nodes created
during polytomy resolution will receive the model through the owning
`TreeTime`, just as any other branch interpolator does.

## 5. IQ-TREE input contract

### 5.1 Required files

For unpartitioned Tier A:

- `.rate`

For unpartitioned Tier B:

- `.siteprob`
- `.iqtree` report containing category rates and weights
- optional `.rate` for a posterior-mean cross-check

For partitioned Tier A or Tier B:

- `.rate` and/or `.siteprob`, according to mode
- `.best_model.nex` or the original partition definition, to map partition-local
  sites back to global alignment coordinates
- `.iqtree` report, to identify the partition model and read `Speed`
- for Tier B, category rates and weights from `.best_model.nex` or another
  unambiguous structured model source

The CLI MUST accept these paths explicitly. It MAY discover sibling files by
exact suffix replacement, but it MUST print every discovered path and MUST
allow explicit override. Arbitrary string replacement such as
`path.replace(".rate", ".iqtree")` is prohibited.

### 5.2 Partition coordinate mapping

IQ-TREE partitioned `.rate` and `.siteprob` rows are emitted
partition-by-partition. `Site` is the one-based site within that partition, not
the global alignment coordinate.

The loader MUST:

1. parse ordered charset coordinates from the partition/model file;
2. map IQ-TREE `Part` or `Set` ID to the corresponding ordered charset;
3. map `(partition ID, local site)` to one global alignment coordinate;
4. reorder all mean rates, posterior rows, and category-rate rows onto the
   original alignment axis;
5. verify a bijection over the expected `L` sites.

It MUST reject:

- duplicate global coordinates;
- overlapping partition assignments;
- missing alignment coordinates;
- out-of-range local or global indices;
- a row count different from the alignment length;
- a partition ID absent from the partition definition.

Partition names are arbitrary. No parser may assume names such as `pos1`,
`pos2`, or `pos3`.

A structured parser or explicit token grammar MUST be used for nested IQ-TREE
model expressions. A loose regular expression over arbitrary report text is
not acceptable.

### 5.3 Partition speed

For an edge-linked-proportional model, the loader MUST read the `Speed` column
from the IQ-TREE `SUBSTITUTION PROCESS` table and multiply every within-
partition category or posterior-mean rate by that speed before global
normalization.

For an edge-equal model, the partition speed is one unless IQ-TREE explicitly
reports another shared scaling convention.

For an edge-unlinked model, loading MUST fail with an explanation that a single
partition scalar cannot represent branch-specific partition lengths.

There is no fallback from missing partition speed to one for a proportional
model. Silent fallback would change the estimand.

### 5.4 Cross-file checks

When `.rate` and `.siteprob` are both provided, the loader MUST verify within
each partition, before partition-speed scaling, that:

```text
rate_file_mean_i ~= sum_k p_ik * rho_gk
```

within a documented tolerance reflecting IQ-TREE output precision.

This catches:

- category order mismatches;
- wrong partition/model files;
- malformed posterior columns;
- unsupported invariant-category conventions.

The loader MUST also verify the sequence length and partition sizes reported in
`.iqtree` against the alignment and mapped rows.

It cannot prove that the alignment characters are identical because IQ-TREE
does not emit an alignment hash. Documentation MUST require the exact same
alignment and column order for IQ-TREE and TreeTime.

### 5.5 Zero-rate categories

Rate zero is valid: `exp(Q * 0)` is the identity. Tier B MUST retain an
explicitly represented zero-rate category rather than replacing it with
`NaN`.

The canonical model and ELBO evaluator support zero-rate categories directly.
The IQ-TREE adapter rejects `+I+R` until its `-wspr` category convention is
verified; it does not infer an invariant responsibility from `.rate` category
assignments. The initial upstream PR will not expose
`include|exclude|epsilon` policy flags. Site exclusion is a distinct analysis
choice and should use an explicit site mask in a later feature.

### 5.6 Validation policy

Parsers MUST fail on malformed data; they MUST NOT skip bad rows and continue.

Reject:

- NaN or infinite rates/probabilities;
- negative rates or probabilities;
- posterior rows with zero total mass;
- materially non-unit posterior row sums;
- materially non-unit category-prior row sums;
- inconsistent category counts not representable by padding;
- absent headers when column meaning is ambiguous;
- unsupported mixture output such as mixture-class posteriors passed as
  rate-category posteriors.

Only small floating-point row-sum deviations may be renormalized, and the
renormalization count and maximum deviation MUST be logged.

## 6. TreeTime architecture

### 6.1 Separate data from the substitution model

The site-rate model carries site/category data. A scalar `GTR` carries the
shared substitution generator `Q`. A `GTR_site_specific` derived from the
scalar GTR and `mean_rates` carries the Tier A ancestral model.

Tier B transition matrices SHOULD be evaluated with the scalar GTR:

```text
base_gtr.expQt(t * category_rate)
```

This avoids double-scaling through `GTR_site_specific._mu` and avoids building
an `(alphabet, alphabet, L)` matrix for every category and partition.

The Tier B evaluator SHOULD group sites by unique category-rate vectors, then
reuse the `K` scalar transition matrices for every site in that group. It MUST
compute the posterior-weighted sum of category-specific log probabilities,
not the log of a posterior-weighted probability. The current
`prob_t_profiles_mixture` implementation should be deleted or replaced by an
unambiguously named ELBO evaluator such as `prob_t_profiles_elbo`.

### 6.2 Ownership and call path

The intended path is:

1. The wrapper validates and loads `SiteRateModel`.
2. The wrapper obtains the scalar base GTR:
   - infer it when `--gtr infer`; or
   - preserve the user-selected/custom GTR without re-inference.
3. The wrapper constructs a Tier A `GTR_site_specific` using
   `SiteRateModel.mean_rates`, base `Pi`, and base `W`.
4. `TreeTime` is constructed with:
   - the Tier A GTR;
   - `site_rate_model`;
   - `branch_length_mode="marginal"`;
   - `compress=False`.
5. Marginal ancestral inference and branch preconditioning use the Tier A GTR.
6. `ClockTree` passes the analysis-owned site-rate model and scalar base GTR
   directly to each `BranchLenInterpolator`.
7. In marginal mode, the interpolator uses Tier A or Tier B according to the
   site-rate model's evaluation mode.

No model data are copied onto `Clade` objects.

### 6.3 Do not create a branch-length mode

`marginal_mixture` MUST NOT be a new `branch_length_mode`.

Branch-length mode describes joint, marginal, or input treatment of ancestral
states. Site-rate evaluation is an orthogonal model choice. The implementation
will keep `branch_length_mode="marginal"` and select Tier A or Tier B through
`site_rate_mode`.

This removes special registrations from `TreeTime._set_branch_length_mode`,
`TreeAnc.optimize_tree`, polytomy resolution, and stochastic subtree creation.

### 6.4 GTR behavior

- `GTR_site_specific.prob_t_compressed` MUST raise `NotImplementedError`.
- TreeTime MUST reject compression with site-specific GTRs before inference.
- `run_timetree` MUST pass `infer_gtr` explicitly.
- The wrapper MUST NOT mutate `params.gtr` to a sentinel value.
- The base GTR alphabet MUST follow the alignment/user configuration; it MUST
  not be hardcoded to nucleotide JC.
- User-specified GTR and custom-GTR behavior MUST have regression tests.

### 6.5 CLI

Proposed options:

```text
--site-rates RATE_FILE
--site-rate-mode {mean,posterior-elbo}
--site-rate-posteriors SITEPROB_FILE
--site-rate-report IQTREE_FILE
--site-rate-model BEST_MODEL_NEX
```

Rules:

- `mean` requires `.rate`, unless a complete posterior model is provided from
  which the mean can be computed.
- `posterior-elbo` requires `.siteprob` plus unambiguous category rates.
- Partitioned input additionally requires coordinate mapping and partition
  speed sources.
- `--no-compress` is set automatically for site-rate modes.
- Explicit `--branch-length-mode joint` or `input` is an error.
- `auto` becomes `marginal`; explicit `marginal` is accepted.
- VCF input is rejected with a focused message in the first PR.
- The deprecated local flags `--no-normalize-site-rates` and
  `--site-rate-invariant-policy` are not included upstream.

## 7. Numerical contract

For every evaluated `t`:

- Negative `t` returns the standard impossible-length penalty.
- Scalar transition matrices are finite and nonnegative.
- Category-specific transition probabilities are converted to log
  probabilities before posterior weighting.
- Terms with zero posterior weight are excluded before taking logarithms and
  do not contribute.
- A zero category rate returns the identity transition.
- A zero category-specific transition probability with positive posterior
  weight is floored only at TreeTime's documented likelihood floor; it is not
  converted to a plausible probability.
- No NaN or infinity may enter a branch-length `Distribution`.

The regression test for negative eigendecomposition roundoff MUST reproduce a
real high-`t * rate` numerical case and compare against
`scipy.linalg.expm` within tolerance. Injecting an artificial transition
probability such as `-0.5` and checking only finiteness is insufficient.

## 8. Performance contract

Tier B is opt-in and may cost approximately `K` times Tier A, but accidental
extra factors of partition count or sequence length are not acceptable.

Required benchmarks:

1. likelihood microbenchmark over one branch grid for:
   - `L=1,000`, `K=4`, one partition;
   - `L=1,000`, `K=6`, three partitions;
   - concentrated and diffuse posteriors;
2. end-to-end benchmark on a checked-in or reproducibly generated medium tree;
3. memory measurement for uncompressed marginal profiles plus `L x K` arrays.

Acceptance targets:

- Runtime scales approximately linearly in `L * K * branches * grid_points`.
- Partition grouping does not recompute an `L`-site transition stack once per
  partition/category.
- Tier B on the medium benchmark is no more than 8x Tier A.
- Peak memory is documented and does not grow as
  `branches * L * K` from duplicated model arrays.

If a large input is expected to take hours, the CLI SHOULD print an estimate or
at minimum a warning based on `branches * L * K * grid_points`.

## 9. Test contract

### 9.1 Loader tests

Checked-in fixtures MUST cover:

- unpartitioned `.rate`;
- unpartitioned `.siteprob`;
- a partitioned file with interleaved arbitrary charset names;
- a partitioned file with contiguous charsets;
- unequal partition sizes;
- unequal category counts, padded correctly;
- an explicit zero-rate category in the canonical-model tests;
- edge-equal and edge-linked-proportional speeds;
- rejection of edge-unlinked partitions;
- reordered input rows;
- duplicate, missing, malformed, negative, NaN, and out-of-range rows;
- `.rate` versus posterior-mean cross-check.

At least one fixture MUST be copied from or generated exactly like the local
IQ-TREE output writers, including `Part` for `.rate`, `Set` for `.siteprob`,
and partition-local site numbering.

The critical invariant test is:

```text
global_rate[global_coordinate(part, local_site)] == source_rate_row
```

for every row, not merely equal row counts or equal means.

### 9.2 Likelihood unit tests

- `K=1`, rate one equals vanilla marginal likelihood tightly.
- One-hot posteriors equal a fixed per-site category-rate GTR.
- Tier A equals Tier B in the zero-variance rate case.
- A hand-calculated two-site/two-category frozen-responsibility ELBO matches
  exactly.
- Category order permutations leave the likelihood unchanged.
- Partition row permutations leave the likelihood unchanged after mapping.
- Gap weighting matches vanilla `prob_t_profiles`.
- Rate zero is finite and correct.
- Large `t * rate` is stable and agrees with a matrix-exponential reference.
- Shape/value validation fails before numerical evaluation.

### 9.3 Integration tests

Use a deterministic, clock-calibrated simulation whose topology, tip dates,
internal dates, global clock, partition speeds, site categories, and
substitution model are known.

Required comparisons:

- vanilla TreeTime;
- uniform-rate site model;
- Tier A with true/posterior-mean rates;
- Tier B with one-hot posteriors;
- Tier B with calibrated diffuse posteriors.

Assertions MUST use predeclared tolerances for clock and internal-node date
recovery. Tests that only assert finiteness or skip assertions when dates are
missing are not acceptance tests.

Include:

- CLI smoke tests for Tier A and Tier B;
- greedy and stochastic polytomy resolution;
- explicit/custom GTR preservation;
- inferred GTR behavior;
- amino-acid input if included in the documented support surface;
- failure for VCF, joint mode, and compressed input.

### 9.4 ELBO identity and calibration tests

For small arrays of positive category likelihoods, test the variational
identity directly:

```text
ELBO(p) =
    sum_k p_k * log(L_k)
    + sum_k p_k * (log(w_k) - log(p_k)).
```

Required checks:

- `ELBO(p)` does not exceed the corresponding log marginal likelihood;
- the bound is tight when `p` is the posterior at the guide parameters;
- finite-difference gradients agree at that guide point;
- the frozen ELBO omits the posterior score-variance curvature term; and
- guide-scale perturbation simulations quantify the resulting point bias and
  interval coverage.

These are algebra and calibration tests, not an implementation of an exact
FreeRate date engine. Scenarios MUST be selected before examining results.
Desired agreement with a downstream clock estimate is not an acceptance
criterion.

### 9.5 CI

The new tests MUST be invoked by repository CI. Adding a test file without
adding it to `test.sh` or replacing the selective calls with an appropriate
pytest invocation is insufficient.

The PR MUST pass:

- the supported Python-version matrix;
- existing command-line, TreeTime, and VCF tests;
- new site-rate tests;
- Ruff lint and format;
- documentation build.

## 10. User documentation and provenance

Minimal upstream documentation includes:

1. a timetree tutorial subsection with complete unpartitioned and partitioned
   IQ-TREE commands;
2. a table of required files by mode;
3. the Tier A and frozen-responsibility ELBO objectives in user-level
   language;
4. the same-alignment/order requirement;
5. partition mapping and speed behavior;
6. forced marginal mode and disabled compression;
7. unsupported VCF, codon, and edge-unlinked cases, plus the explicit boundary
   that exact FreeRate dating is outside TreeTime's scope;
8. expected Tier B runtime;
9. an IQ-TREE citation and a statement that the handoff is empirical Bayes;
10. an `Unreleased` changelog entry.

Each run SHOULD write a small audit table in the output directory containing:

```text
site  partition  partition_site  partition_speed  posterior_mean_rate
```

and a metadata summary containing source paths, model mode, category counts,
normalization constant, validation tolerances, and TreeTime version.

This output is the check that would have exposed the original partition-order
bug and is part of the scientific provenance, not optional debug logging.

## 11. Implementation and review sequence

The current iterative branch should be retained as a reference, not submitted
with its existing history. Build the upstream PR from current `master` in
reviewable commits:

1. **General correctness prerequisites**
   - guard `GTR_site_specific.prob_t_compressed`;
   - pass `infer_gtr` explicitly;
   - add focused regression tests.
2. **Canonical model and IQ-TREE adapter**
   - implement validated global coordinate mapping;
   - parse partition speed and category models;
   - add loader fixtures and failure tests.
3. **Tier A integration**
   - preserve or infer the scalar GTR correctly;
   - construct the posterior-mean site-specific GTR;
   - force marginal/uncompressed execution.
4. **Tier B evaluator**
   - use scalar transition matrices and the normative frozen-responsibility
     ELBO;
   - keep model ownership at `TreeTime`, not nodes;
   - add numerical, ELBO-identity, and guide-scale calibration tests.
5. **CLI, provenance, performance, and documentation**
   - add both CLI paths;
   - add audit output and warnings;
   - add benchmarks, tutorial, API notes, and changelog.

Do not include:

- the local three-line `uv.lock`;
- private data artifacts or expected results from a downstream analysis;
- phase-numbered commit messages;
- compatibility aliases for an unreleased `marginal_mixture` mode;
- silent parser fallbacks.

## 12. Pull-request acceptance gates

The PR is ready for upstream review only when all are true:

### Correctness

- Partitioned rows map bijectively to global alignment coordinates.
- Partition speeds are applied before one global normalization.
- `.rate` agrees with the posterior/category model within tolerance.
- Tier A and Tier B units preserve a mean-one relative rate and a separate
  global clock.
- Tier B implements the frozen-responsibility ELBO and does not remix
  categories independently by branch.
- Invariant categories work without NaN.
- Unsupported models fail before inference.

### Regression safety

- Uniform/K=1 behavior matches vanilla TreeTime.
- Custom/fixed GTRs are not silently re-inferred.
- Existing TreeTime tests pass.
- No site-rate state is duplicated onto tree nodes.

### Validation

- Clock-calibrated simulations recover the known scale.
- Guide-scale perturbation does not produce unacceptable systematic Tier B
  bias under the predeclared scenarios.
- Interval coverage and the omitted score-variance curvature are reported
  explicitly.
- Interleaved-partition fixtures reproduce the IQ-TREE writer's real ordering.

### Integration quality

- Site-rate tests run in CI.
- Lint, format, docs, and Python matrix pass.
- Public docs state inputs, objective, limitations, and performance.
- Audit output makes the final global site mapping inspectable.

### Performance

- The medium Tier B benchmark is within the stated target.
- Memory does not duplicate `L x K` data per branch.
- Large-run cost is warned about and documented.
