#!/usr/bin/env python3
"""Reproducible microbenchmark for TreeTime site-rate branch likelihoods.

Run from the repository root with:

    uv run python test/benchmark_site_rate_model.py

The benchmark times one 75-point branch grid and reports peak Python-tracked
memory. It is intentionally not a CI test because wall-clock thresholds are
hardware dependent.
"""

from dataclasses import dataclass, replace
from time import perf_counter
import tracemalloc

import numpy as np
from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from treetime import GTR, SiteRateModel, TreeTime, build_site_rate_gtrs


@dataclass
class Result:
    scenario: str
    tier_a_seconds: float
    tier_b_seconds: float
    ratio: float
    peak_megabytes: float
    model_megabytes: float


def _model(length, categories, partitions, concentrated, rng):
    partition_index = np.arange(length) % partitions
    partition_site = np.zeros(length, dtype=int)
    for partition in range(partitions):
        sites = np.flatnonzero(partition_index == partition)
        partition_site[sites] = np.arange(1, len(sites) + 1)

    rates_by_partition = np.vstack(
        [
            np.geomspace(0.05 * (1 + 0.1 * partition), 3.0 * (1 + 0.1 * partition), categories)
            for partition in range(partitions)
        ]
    )
    selected = rng.integers(0, categories, size=length)
    if concentrated:
        posterior = np.full((length, categories), 0.03 / (categories - 1))
        posterior[np.arange(length), selected] = 0.97
    else:
        posterior = rng.dirichlet(np.ones(categories) * 2, size=length)
    category_rates = rates_by_partition[partition_index]
    normalization = np.mean(np.sum(posterior * category_rates, axis=1))
    category_rates = category_rates / normalization
    mean_rates = np.sum(posterior * category_rates, axis=1)
    return SiteRateModel(
        mean_rates=mean_rates,
        posterior_weights=posterior,
        category_rates=category_rates,
        category_prior_weights=np.full((length, categories), 1 / categories),
        category_mask=np.ones((length, categories), dtype=bool),
        partition_index=partition_index,
        partition_site=partition_site,
        partition_names=tuple(f'partition-{index + 1}' for index in range(partitions)),
        partition_speeds=np.ones(partitions),
        evaluation_mode='posterior-elbo',
    )


def _time_grid(function, grid):
    start = perf_counter()
    for branch_length in grid:
        function(branch_length)
    return perf_counter() - start


def benchmark(length, categories, partitions, concentrated, seed=13):
    rng = np.random.default_rng(seed)
    model = _model(length, categories, partitions, concentrated, rng)
    parent = rng.dirichlet(np.ones(5), size=length)
    child = rng.dirichlet(np.ones(5), size=length)
    profiles = (parent, child)
    multiplicity = np.ones(length)
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, GTR.standard('JC69', alphabet='nuc'))
    grid = np.linspace(0, 0.1, 75)

    site_gtr.prob_t_profiles(profiles, multiplicity, grid[1], return_log=True)
    model.prob_t_profiles_elbo(scalar_gtr, profiles, multiplicity, grid[1], return_log=True)
    tier_a_seconds = _time_grid(
        lambda branch_length: site_gtr.prob_t_profiles(
            profiles,
            multiplicity,
            branch_length,
            return_log=True,
        ),
        grid,
    )

    tracemalloc.start()
    tier_b_seconds = _time_grid(
        lambda branch_length: model.prob_t_profiles_elbo(
            scalar_gtr,
            profiles,
            multiplicity,
            branch_length,
            return_log=True,
        ),
        grid,
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    arrays = (
        model.mean_rates,
        model.partition_index,
        model.partition_site,
        model.partition_speeds,
        model.posterior_weights,
        model.category_rates,
        model.category_prior_weights,
        model.category_mask,
    )
    model_bytes = sum(array.nbytes for array in arrays)
    concentration = 'concentrated' if concentrated else 'diffuse'
    return Result(
        scenario=f'L={length},K={categories},P={partitions},{concentration}',
        tier_a_seconds=tier_a_seconds,
        tier_b_seconds=tier_b_seconds,
        ratio=tier_b_seconds / tier_a_seconds,
        peak_megabytes=peak_bytes / 1024**2,
        model_megabytes=model_bytes / 1024**2,
    )


def _medium_inputs(length=400, tips=32, seed=19):
    rng = np.random.default_rng(seed)
    names = [f'tip-{index:02d}' for index in range(tips)]
    clades = [
        Phylo.BaseTree.Clade(branch_length=0.006 + 0.0002 * (index % 5), name=name) for index, name in enumerate(names)
    ]
    while len(clades) > 1:
        parents = []
        for index in range(0, len(clades), 2):
            children = clades[index : index + 2]
            parents.append(Phylo.BaseTree.Clade(branch_length=0.006, clades=children))
        clades = parents
    tree = Phylo.BaseTree.Tree(root=clades[0])

    root_sequence = rng.choice(np.asarray(list('ACGT')), size=length)
    records = []
    for index, name in enumerate(names):
        sequence = root_sequence.copy()
        mutation_count = 4 + index // 3
        positions = rng.choice(length, size=mutation_count, replace=False)
        sequence[positions] = rng.choice(np.asarray(list('ACGT')), size=mutation_count)
        records.append(SeqRecord(Seq(''.join(sequence)), id=name, description=''))
    alignment = MultipleSeqAlignment(records)
    dates = {name: 2000.0 + 0.8 * index for index, name in enumerate(names)}
    return tree, alignment, dates


def benchmark_end_to_end(seed=23):
    rng = np.random.default_rng(seed)
    model = _model(400, 4, 2, True, rng)
    tier_a_model = replace(model, evaluation_mode='mean')

    def run(site_rate_model):
        tree, alignment, dates = _medium_inputs()
        site_gtr, scalar_gtr = build_site_rate_gtrs(site_rate_model, GTR.standard('JC69', alphabet='nuc'))
        analysis = TreeTime(
            tree=tree,
            aln=alignment,
            dates=dates,
            gtr=site_gtr,
            compress=False,
            branch_length_mode='marginal',
            site_rate_model=site_rate_model,
            site_rate_base_gtr=scalar_gtr,
            verbose=0,
            rng_seed=seed,
        )
        start = perf_counter()
        analysis.run(
            root=None,
            infer_gtr=False,
            resolve_polytomies=False,
            max_iter=0,
            fixed_clock_rate=0.001,
            branch_length_mode='marginal',
            time_marginal='never',
            raise_uncaught_exceptions=True,
        )
        return perf_counter() - start

    tier_a_seconds = run(tier_a_model)
    tracemalloc.start()
    tier_b_seconds = run(model)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return tier_a_seconds, tier_b_seconds, peak_bytes / 1024**2


def main():
    print('scenario\ttier_a_s\ttier_b_s\tratio\tpeak_mb\tmodel_mb')
    ratios = []
    for length, categories, partitions in ((1000, 4, 1), (1000, 6, 3)):
        for concentrated in (True, False):
            result = benchmark(length, categories, partitions, concentrated)
            ratios.append(result.ratio)
            print(
                f'{result.scenario}\t{result.tier_a_seconds:.4f}\t'
                f'{result.tier_b_seconds:.4f}\t{result.ratio:.2f}\t'
                f'{result.peak_megabytes:.2f}\t{result.model_megabytes:.2f}'
            )
    tier_a_seconds, tier_b_seconds, peak_megabytes = benchmark_end_to_end()
    ratios.append(tier_b_seconds / tier_a_seconds)
    print(
        f'end-to-end:L=400,tips=32,K=4,P=2\t{tier_a_seconds:.4f}\t'
        f'{tier_b_seconds:.4f}\t{tier_b_seconds / tier_a_seconds:.2f}\t'
        f'{peak_megabytes:.2f}\t-'
    )
    if max(ratios) >= 8:
        raise SystemExit(f'Tier B benchmark exceeded the 8x acceptance boundary: {max(ratios):.2f}x')


if __name__ == '__main__':
    main()
