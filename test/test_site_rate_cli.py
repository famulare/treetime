from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from Bio import Phylo

from treetime import GTR, SiteRateModel, TreeAnc, TreeTime, UnknownMethodError, make_parser
from treetime import wrappers
from treetime.iqtree_site_rates import load_iqtree_site_rate_posteriors, load_iqtree_site_rates
from treetime.seq_utils import seq2prof
from treetime.seqgen import SeqGen
from treetime.site_rate_model import build_site_rate_gtrs, write_site_rate_audit
from treetime import utils
from treetime.wrappers import (
    _load_site_rate_model,
    _prepare_site_rate_analysis,
    _report_site_rate_setup,
    _site_rate_requested,
)


DATA = Path(__file__).parent / 'data' / 'site_rate_cli'
REPOSITORY = Path(__file__).parent.parent


def _site_rate_params(**changes):
    values = {
        'aln': str(DATA / 'tiny.fasta'),
        'branch_length_mode': 'marginal',
        'custom_gtr': None,
        'gtr': 'JC69',
        'keep_overhangs': False,
        'rng_seed': 7,
        'sequence_length': None,
        'site_rate_mode': 'mean',
        'site_rate_model': None,
        'site_rate_loglh': None,
        'site_rate_report': None,
        'site_rates': str(DATA / 'tiny.rate'),
        'tree': str(DATA / 'tiny.nwk'),
        'verbose': 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _run_cli(output_directory, mode):
    command = [
        sys.executable,
        '-m',
        'treetime',
        '--tree',
        str(DATA / 'tiny.nwk'),
        '--dates',
        str(DATA / 'tiny_dates.tsv'),
        '--aln',
        str(DATA / 'tiny.fasta'),
        '--branch-length-mode',
        'marginal',
        '--gtr',
        'JC69',
        '--clock-rate',
        '0.001',
        '--max-iter',
        '0',
        '--keep-root',
        '--keep-polytomies',
        '--outdir',
        str(output_directory),
        '--verbose',
        '0',
        '--no-tip-labels',
    ]
    if mode == 'mean':
        command.extend(['--site-rates', str(DATA / 'tiny.rate')])
    else:
        command.extend(
            [
                '--site-rate-mode',
                'posterior-elbo',
                '--site-rate-loglh',
                str(DATA / 'tiny.sitelh'),
                '--site-rate-report',
                str(DATA / 'tiny.iqtree'),
            ]
        )
    environment = os.environ.copy()
    environment['MPLCONFIGDIR'] = str(output_directory.parent / 'matplotlib')
    return subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_site_rate_flags_are_public_timetree_options():
    params = make_parser().parse_args(
        [
            '--tree',
            'tree.nwk',
            '--dates',
            'dates.tsv',
            '--aln',
            'alignment.fasta',
            '--site-rates',
            'analysis.rate',
            '--site-rate-mode',
            'posterior-elbo',
            '--site-rate-loglh',
            'analysis.sitelh',
            '--site-rate-report',
            'analysis.iqtree',
            '--site-rate-model',
            'analysis.best_model.nex',
        ]
    )

    assert params.site_rates == 'analysis.rate'
    assert params.site_rate_mode == 'posterior-elbo'
    assert params.site_rate_loglh == 'analysis.sitelh'
    assert params.site_rate_report == 'analysis.iqtree'
    assert params.site_rate_model == 'analysis.best_model.nex'


def test_explicit_mean_mode_without_input_does_not_silently_run_vanilla():
    params = make_parser().parse_args(
        [
            '--tree',
            'tree.nwk',
            '--dates',
            'dates.tsv',
            '--aln',
            'alignment.fasta',
            '--site-rate-mode',
            'mean',
        ]
    )

    assert _site_rate_requested(params)
    with pytest.raises(ValueError, match='requires --site-rates'):
        _load_site_rate_model(params, sequence_length=10)


def test_cli_requires_shared_tree_before_topology_inference(monkeypatch):
    params = make_parser().parse_args(
        [
            '--dates',
            str(DATA / 'tiny_dates.tsv'),
            '--aln',
            str(DATA / 'tiny.fasta'),
            '--site-rates',
            str(DATA / 'tiny.rate'),
        ]
    )
    topology_inference_called = False

    def record_topology_inference(*args, **kwargs):
        nonlocal topology_inference_called
        topology_inference_called = True
        return 0

    monkeypatch.setattr(wrappers, 'assure_tree', record_topology_inference)
    with pytest.raises(ValueError, match='explicit --tree'):
        wrappers.timetree(params)
    assert topology_inference_called is False


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        ({'site_rate_mode': 'posterior-elbo'}, 'requires --site-rate-loglh'),
        ({'site_rates': None}, 'requires --site-rates'),
        ({'site_rates': None, 'site_rate_loglh': 'x.sitelh'}, 'requires --site-rate-report'),
    ],
)
def test_cli_site_rate_inputs_fail_closed(changes, message):
    with pytest.raises(ValueError, match=message):
        _load_site_rate_model(_site_rate_params(**changes), sequence_length=50)


def test_complete_posterior_input_can_supply_tier_a_means():
    model = _load_site_rate_model(
        _site_rate_params(
            site_rates=None,
            site_rate_loglh=str(DATA / 'tiny.sitelh'),
            site_rate_report=str(DATA / 'tiny.iqtree'),
        ),
        sequence_length=50,
    )

    assert model.evaluation_mode == 'mean'
    assert model.posterior_weights is not None
    assert model.mean_rates.mean() == pytest.approx(1.0)


def test_posterior_confidence_warning_states_conditional_semantics(capsys):
    model = load_iqtree_site_rate_posteriors(
        DATA / 'tiny.sitelh',
        report_file=DATA / 'tiny.iqtree',
        sequence_length=50,
    )
    fake_phylogeny = SimpleNamespace(
        get_nonterminals=lambda: [object()],
        get_terminals=lambda: [object(), object()],
    )
    fake_tree = SimpleNamespace(tree=fake_phylogeny, branch_grid_points=125)

    _report_site_rate_setup(fake_tree, model, confidence=True)

    assert 'not calibrated FreeRate uncertainty intervals' in capsys.readouterr().err


def test_site_rate_setup_preserves_fixed_gtr_and_infers_only_from_rate_aware_profiles(monkeypatch):
    pi = np.array([0.3, 0.2, 0.25, 0.2, 0.05])
    exchangeability = np.array(
        [
            [0, 1, 2, 3, 0.5],
            [1, 0, 4, 1.5, 0.7],
            [2, 4, 0, 2.5, 0.9],
            [3, 1.5, 2.5, 0, 1.1],
            [0.5, 0.7, 0.9, 1.1, 0],
        ],
        dtype=float,
    )
    fixed_gtr = GTR.custom(mu=2.5, pi=pi, W=exchangeability, alphabet='nuc')
    _, _, fixed_scalar, inferred = _prepare_site_rate_analysis(
        _site_rate_params(gtr='custom', custom_gtr='model.txt'),
        str(DATA / 'tiny.fasta'),
        None,
        None,
        fixed_gtr,
    )

    assert inferred is False
    np.testing.assert_allclose(fixed_scalar.Pi, fixed_gtr.Pi)
    np.testing.assert_allclose(fixed_scalar.W, fixed_gtr.W)
    assert fixed_scalar.average_rate() == pytest.approx(1.0)

    reconstruction_models = []
    original_reconstruction = TreeAnc.infer_ancestral_sequences

    def record_reconstruction_model(tree, *args, **kwargs):
        reconstruction_models.append(tree.gtr.is_site_specific)
        return original_reconstruction(tree, *args, **kwargs)

    monkeypatch.setattr(TreeAnc, 'infer_ancestral_sequences', record_reconstruction_model)
    _, _, inferred_scalar, inferred = _prepare_site_rate_analysis(
        _site_rate_params(gtr='infer'),
        str(DATA / 'tiny.fasta'),
        None,
        None,
        GTR.standard('JC69', alphabet='nuc'),
    )

    assert inferred is True
    assert reconstruction_models == [True]
    assert inferred_scalar.average_rate() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ('changes', 'ref', 'message'),
    [
        ({}, 'ACGT', 'does not support VCF'),
        ({'tree': None}, None, 'explicit --tree'),
        ({'branch_length_mode': 'joint'}, None, 'requires --branch-length-mode marginal'),
        ({'sequence_length': 50}, None, 'does not support --sequence-length'),
    ],
)
def test_site_rate_setup_rejects_unsupported_execution_modes(changes, ref, message):
    with pytest.raises(ValueError, match=message):
        _prepare_site_rate_analysis(
            _site_rate_params(**changes),
            str(DATA / 'tiny.fasta'),
            ref,
            None,
            GTR.standard('JC69', alphabet='nuc'),
        )


def test_site_rate_model_rejects_implicit_sequence_compression():
    model = load_iqtree_site_rates(DATA / 'tiny.rate', sequence_length=50)
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, GTR.standard('JC69', alphabet='nuc'))
    dates = utils.parse_dates(str(DATA / 'tiny_dates.tsv'))

    with pytest.raises(TypeError, match='sequence compression'):
        TreeTime(
            dates=dates,
            tree=str(DATA / 'tiny.nwk'),
            aln=str(DATA / 'tiny.fasta'),
            gtr=site_gtr,
            branch_length_mode='marginal',
            site_rate_model=model,
            site_rate_base_gtr=scalar_gtr,
        )


def test_public_treetime_run_rejects_gtr_inference_after_site_model_attachment():
    model = load_iqtree_site_rates(DATA / 'tiny.rate', sequence_length=50)
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, GTR.standard('JC69', alphabet='nuc'))
    analysis = TreeTime(
        dates=utils.parse_dates(str(DATA / 'tiny_dates.tsv')),
        tree=str(DATA / 'tiny.nwk'),
        aln=str(DATA / 'tiny.fasta'),
        gtr=site_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
        verbose=0,
    )

    with pytest.raises(UnknownMethodError, match='infer it before construction'):
        analysis.run(
            branch_length_mode='marginal',
            raise_uncaught_exceptions=True,
        )
    assert analysis.gtr.is_site_specific


def test_site_rate_audit_records_global_partition_mapping(tmp_path):
    fixture = Path(__file__).parent / 'data' / 'site_rate_model'
    model = load_iqtree_site_rates(
        fixture / 'partitioned.rate',
        sequence_length=6,
        partition_file=fixture / 'partitioned.best_model.nex',
        report_file=fixture / 'partitioned.iqtree',
    )

    table_path, metadata_path = write_site_rate_audit(model, tmp_path)
    rows = table_path.read_text(encoding='utf-8').splitlines()
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    assert rows[0].split('\t') == [
        'site',
        'partition',
        'partition_site',
        'partition_speed',
        'posterior_mean_rate',
    ]
    assert [row.split('\t')[1:3] for row in rows[1:]] == [
        ['alpha_sites', '1'],
        ['beta-block', '1'],
        ['alpha_sites', '2'],
        ['beta-block', '2'],
        ['alpha_sites', '3'],
        ['beta-block', '3'],
    ]
    assert metadata['evaluation_mode'] == 'mean'
    assert metadata['source_metadata']['partition_model'] == 'edge-linked-proportional'
    assert metadata['global_mean_rate'] == pytest.approx(1.0)


def _run_tree(model=None):
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    arguments = {}
    if model is None:
        gtr = base_gtr
    else:
        gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
        arguments = {
            'site_rate_model': model,
            'site_rate_base_gtr': scalar_gtr,
        }
    tree = TreeTime(
        dates=utils.parse_dates(str(DATA / 'tiny_dates.tsv')),
        tree=str(DATA / 'tiny.nwk'),
        aln=str(DATA / 'tiny.fasta'),
        gtr=gtr,
        compress=False,
        branch_length_mode='marginal',
        verbose=0,
        rng_seed=3,
        **arguments,
    )
    tree.run(
        root=None,
        infer_gtr=False,
        resolve_polytomies=False,
        max_iter=0,
        fixed_clock_rate=0.001,
        branch_length_mode='marginal',
        time_marginal='never',
        raise_uncaught_exceptions=True,
    )
    return np.asarray([node.numdate for node in tree.tree.get_nonterminals()])


def test_uniform_tier_a_and_one_hot_tier_b_match_full_treetime_inference():
    uniform = SiteRateModel(
        mean_rates=np.ones(50),
        partition_index=np.zeros(50, dtype=int),
        partition_site=np.arange(1, 51),
        partition_names=('alignment',),
        partition_speeds=np.ones(1),
    )
    tier_a = load_iqtree_site_rates(DATA / 'tiny.rate', sequence_length=50)
    categories = np.unique(tier_a.mean_rates)
    category_index = np.searchsorted(categories, tier_a.mean_rates)
    weights = np.eye(len(categories))[category_index]
    tier_b = SiteRateModel(
        mean_rates=tier_a.mean_rates,
        posterior_weights=weights,
        category_rates=np.tile(categories, (50, 1)),
        category_prior_weights=np.tile(np.ones(len(categories)) / len(categories), (50, 1)),
        category_mask=np.ones((50, len(categories)), dtype=bool),
        partition_index=tier_a.partition_index,
        partition_site=tier_a.partition_site,
        partition_names=tier_a.partition_names,
        partition_speeds=tier_a.partition_speeds,
        evaluation_mode='posterior-elbo',
    )
    vanilla_dates = _run_tree()
    uniform_dates = _run_tree(uniform)
    tier_a_dates = _run_tree(tier_a)
    tier_b_dates = _run_tree(tier_b)

    np.testing.assert_allclose(uniform_dates, vanilla_dates, atol=0.03, rtol=0)
    np.testing.assert_allclose(tier_b_dates, tier_a_dates, atol=0.03, rtol=0)


def _clock_simulation_inputs():
    sequence_length = 1200
    clock_rate = 0.002
    tip_dates = {
        'a': 2000.0,
        'b': 2004.0,
        'c': 2006.0,
        'd': 2010.0,
        'e': 2012.0,
        'f': 2016.0,
    }
    internal_dates = {'root': 1990.0, 'left': 1995.0, 'right': 1997.0, 'middle': 2001.0, 'recent': 2003.0}

    def tip(name):
        return Phylo.BaseTree.Clade(name=name)

    left = Phylo.BaseTree.Clade(name='left', clades=[tip('a'), tip('b')])
    middle = Phylo.BaseTree.Clade(name='middle', clades=[tip('c'), tip('d')])
    recent = Phylo.BaseTree.Clade(name='recent', clades=[tip('e'), tip('f')])
    right = Phylo.BaseTree.Clade(name='right', clades=[middle, recent])
    root = Phylo.BaseTree.Clade(name='root', clades=[left, right])
    tree = Phylo.BaseTree.Tree(root=root, rooted=True)
    node_dates = {**tip_dates, **internal_dates}
    for parent in tree.get_nonterminals(order='preorder'):
        for child in parent:
            child.branch_length = (node_dates[child.name] - node_dates[parent.name]) * clock_rate

    mean_rates = np.tile([0.5, 1.5], sequence_length // 2)
    tier_a = SiteRateModel(
        mean_rates=mean_rates,
        partition_index=np.zeros(sequence_length, dtype=int),
        partition_site=np.arange(1, sequence_length + 1),
        partition_names=('alignment',),
        partition_speeds=np.ones(1),
    )
    category_rates = np.tile([0.5, 1.5], (sequence_length, 1))
    posterior_weights = np.zeros((sequence_length, 2))
    posterior_weights[np.arange(sequence_length), np.arange(sequence_length) % 2] = 1
    tier_b = SiteRateModel(
        mean_rates=mean_rates,
        posterior_weights=posterior_weights,
        category_rates=category_rates,
        category_prior_weights=np.full((sequence_length, 2), 0.5),
        category_mask=np.ones((sequence_length, 2), dtype=bool),
        partition_index=tier_a.partition_index,
        partition_site=tier_a.partition_site,
        partition_names=tier_a.partition_names,
        partition_speeds=tier_a.partition_speeds,
        evaluation_mode='posterior-elbo',
    )
    simulation_gtr, _ = build_site_rate_gtrs(tier_a, GTR.standard('JC69', alphabet='nuc'))
    generator = SeqGen(
        sequence_length,
        tree=deepcopy(tree),
        gtr=simulation_gtr,
        rng_seed=41,
        verbose=0,
    )
    root_sequence = ''.join(np.random.default_rng(42).choice(list('ACGT'), size=sequence_length))
    generator.evolve(root_seq=root_sequence)
    return tree, generator.get_aln(), tip_dates, internal_dates, clock_rate, tier_a, tier_b


def _guide_tree_site_rate_model(tree, alignment, guide_scale):
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    category_rates = np.array([0.5, 1.5])
    priors = np.array([0.5, 0.5])
    tip_profiles = {
        record.id: seq2prof(np.asarray(list(str(record.seq))), base_gtr.profile_map) for record in alignment
    }
    category_likelihoods = []
    for rate in category_rates:
        subtree_likelihood = {}
        for node in tree.find_clades(order='postorder'):
            if node.is_terminal():
                subtree_likelihood[node] = tip_profiles[node.name]
                continue
            profile = np.ones((len(alignment[0]), len(base_gtr.alphabet)))
            for child in node:
                transition = base_gtr.expQt(child.branch_length * guide_scale * rate)
                profile *= subtree_likelihood[child] @ transition
            subtree_likelihood[node] = profile
        category_likelihoods.append(subtree_likelihood[tree.root] @ base_gtr.Pi)

    category_likelihoods = np.column_stack(category_likelihoods)
    posterior = category_likelihoods * priors
    posterior /= posterior.sum(axis=1, keepdims=True)
    normalization = float(np.mean(posterior @ category_rates))
    normalized_rates = category_rates / normalization
    sequence_length = len(alignment[0])
    return SiteRateModel(
        mean_rates=posterior @ normalized_rates,
        posterior_weights=posterior,
        category_rates=np.tile(normalized_rates, (sequence_length, 1)),
        category_prior_weights=np.tile(priors, (sequence_length, 1)),
        category_mask=np.ones((sequence_length, 2), dtype=bool),
        partition_index=np.zeros(sequence_length, dtype=int),
        partition_site=np.arange(1, sequence_length + 1),
        partition_names=('alignment',),
        partition_speeds=np.ones(1),
        evaluation_mode='posterior-elbo',
    )


def _run_clock_simulation(tree, alignment, tip_dates, model):
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, GTR.standard('JC69', alphabet='nuc'))
    analysis = TreeTime(
        dates=tip_dates,
        tree=deepcopy(tree),
        aln=alignment,
        gtr=site_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
        verbose=0,
        rng_seed=43,
    )
    analysis.run(
        root=None,
        infer_gtr=False,
        resolve_polytomies=False,
        max_iter=0,
        branch_length_mode='marginal',
        time_marginal='never',
        raise_uncaught_exceptions=True,
    )
    internal_dates = {node.name: node.numdate for node in analysis.tree.get_nonterminals()}
    return analysis.date2dist.clock_rate, internal_dates


def test_clock_calibrated_simulation_recovers_scale_and_internal_dates():
    tree, alignment, tip_dates, true_internal_dates, true_clock_rate, tier_a, tier_b = _clock_simulation_inputs()

    tier_a_clock, tier_a_dates = _run_clock_simulation(tree, alignment, tip_dates, tier_a)
    tier_b_clock, tier_b_dates = _run_clock_simulation(tree, alignment, tip_dates, tier_b)

    assert tier_a_clock == pytest.approx(true_clock_rate, rel=0.2)
    assert tier_b_clock == pytest.approx(true_clock_rate, rel=0.2)
    for name, true_date in true_internal_dates.items():
        tolerance = 5.0 if name == 'root' else 3.0
        assert tier_a_dates[name] == pytest.approx(true_date, abs=tolerance)
        assert tier_b_dates[name] == pytest.approx(true_date, abs=tolerance)

    for guide_scale in (0.8, 1.0, 1.2):
        guide_model = _guide_tree_site_rate_model(tree, alignment, guide_scale)
        guide_clock, guide_dates = _run_clock_simulation(tree, alignment, tip_dates, guide_model)
        assert guide_clock == pytest.approx(true_clock_rate, rel=0.3)
        for name, true_date in true_internal_dates.items():
            tolerance = 7.0 if name == 'root' else 5.0
            assert guide_dates[name] == pytest.approx(true_date, abs=tolerance)


def test_dense_amino_acid_alignment_uses_site_rate_model():
    sequence_length = 200
    model = SiteRateModel(
        mean_rates=np.tile([0.5, 1.5], sequence_length // 2),
        partition_index=np.zeros(sequence_length, dtype=int),
        partition_site=np.arange(1, sequence_length + 1),
        partition_names=('alignment',),
        partition_speeds=np.ones(1),
    )
    base_gtr = GTR.standard('JTT92')
    simulation_gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
    tree = Phylo.read(str(DATA / 'tiny.nwk'), 'newick')
    generator = SeqGen(
        sequence_length,
        tree=deepcopy(tree),
        gtr=simulation_gtr,
        rng_seed=47,
        verbose=0,
    )
    amino_acids = [state for state in base_gtr.alphabet if state != '-']
    root_sequence = ''.join(np.random.default_rng(48).choice(amino_acids, size=sequence_length))
    generator.evolve(root_seq=root_sequence)
    analysis = TreeTime(
        dates=utils.parse_dates(str(DATA / 'tiny_dates.tsv')),
        tree=tree,
        aln=generator.get_aln(),
        gtr=simulation_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
        verbose=0,
        rng_seed=49,
    )

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

    assert np.array_equal(analysis.gtr.alphabet, base_gtr.alphabet)
    assert analysis.date2dist.clock_rate == pytest.approx(0.001)


@pytest.mark.parametrize('stochastic', [False, True], ids=['greedy', 'stochastic'])
def test_site_rate_model_survives_polytomy_resolution(tmp_path, stochastic):
    tree_path = tmp_path / 'polytomy.nwk'
    tree_path.write_text(
        '('
        'A_1977:0.015,B_1981:0.020,C_1989:0.025,D_1993:0.018,'
        'E_2001:0.030,F_2005:0.022,G_2010:0.035,H_2013:0.028,'
        'I_2017:0.040,J_2020:0.038'
        ');\n',
        encoding='utf-8',
    )
    model = load_iqtree_site_rate_posteriors(
        DATA / 'tiny.sitelh',
        report_file=DATA / 'tiny.iqtree',
        sequence_length=50,
    )
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, GTR.standard('JC69', alphabet='nuc'))
    tree = TreeTime(
        dates=utils.parse_dates(str(DATA / 'tiny_dates.tsv')),
        tree=str(tree_path),
        aln=str(DATA / 'tiny.fasta'),
        gtr=site_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
        verbose=0,
        rng_seed=5,
    )

    tree.run(
        root=None,
        infer_gtr=False,
        resolve_polytomies=True,
        stochastic_resolve=stochastic,
        max_iter=1,
        fixed_clock_rate=0.001,
        branch_length_mode='marginal',
        time_marginal='never',
        prune_short=False,
        raise_uncaught_exceptions=True,
    )
    if not stochastic and len(tree.tree.get_nonterminals()) == 1:
        tree.resolve_polytomies(stochastic_resolve=False, resolution_threshold=-np.inf)

    assert len(tree.tree.get_nonterminals()) > 1
    assert all(not hasattr(node, 'site_rate_model') for node in tree.tree.find_clades())
    interpolators = [
        node.branch_length_interpolator
        for node in tree.tree.find_clades()
        if node.up is not None and hasattr(node, 'branch_length_interpolator')
    ]
    assert interpolators
    assert all(interpolator.site_rate_model is model for interpolator in interpolators)
    assert all(interpolator.site_rate_base_gtr is scalar_gtr for interpolator in interpolators)


@pytest.mark.parametrize('mode', ['mean', 'posterior-elbo'])
def test_site_rate_cli_runs_end_to_end_and_writes_provenance(tmp_path, mode):
    output_directory = tmp_path / mode
    result = _run_cli(output_directory, mode)

    assert result.returncode == 0, result.stderr
    assert f'Using {mode} site-rate dating' in result.stdout
    assert (output_directory / 'dates.tsv').is_file()
    assert (output_directory / 'timetree.nexus').is_file()
    metadata = json.loads((output_directory / 'site_rate_model.json').read_text(encoding='utf-8'))
    assert metadata['evaluation_mode'] == mode
    assert metadata['sequence_length'] == 50
    audit_rows = (output_directory / 'site_rate_model.tsv').read_text(encoding='utf-8').splitlines()
    assert len(audit_rows) == 51
