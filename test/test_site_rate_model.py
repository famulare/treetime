from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from treetime import GTR, TreeAnc, TreeTime, UnknownMethodError
from treetime import argument_parser as argument_parser_module
from treetime.CLI_io import read_if_vcf
from treetime.branch_len_interpolator import BranchLenInterpolator
from treetime.gtr_site_specific import GTR_site_specific
from treetime.site_rate_model import SiteRateModel, build_site_rate_gtrs
from treetime.wrappers import create_gtr, run_timetree


def test_site_specific_gtr_rejects_compressed_likelihoods():
    gtr = GTR_site_specific(seq_len=3, alphabet='nuc')
    with pytest.raises(NotImplementedError, match='site-specific rates'):
        gtr.prob_t_compressed(None, None, 0.1)


@pytest.mark.parametrize(
    ('gtr_name', 'custom_gtr', 'expected'),
    [('infer', None, True), ('JC69', None, False), ('infer', 'model.txt', False)],
)
def test_run_timetree_passes_infer_gtr_explicitly(gtr_name, custom_gtr, expected):
    captured = {}

    class StopAfterRunCall(Exception):
        pass

    class FakeTree:
        one_mutation = 0.01

        def run(self, **kwargs):
            captured.update(kwargs)
            raise StopAfterRunCall

    params = SimpleNamespace(
        aa=False,
        aln=None,
        branch_length_mode='marginal',
        clock_filter=0,
        clock_filter_method='residual',
        clock_rate=None,
        clock_std_dev=None,
        coalescent=0.0,
        confidence=False,
        covariation=False,
        custom_gtr=custom_gtr,
        gtr=gtr_name,
        keep_polytomies=True,
        keep_root=True,
        max_iter=0,
        n_branches_posterior=False,
        n_skyline=20,
        reconstruct_tip_states=False,
        relax=None,
        reroot=None,
        sequence_length=None,
        stochastic_resolve=False,
        time_marginal='never',
        tip_slack=0,
    )

    with pytest.raises(StopAfterRunCall):
        run_timetree(FakeTree(), params, outdir='')

    assert captured['infer_gtr'] is expected


def test_create_gtr_does_not_mutate_inference_configuration(tmp_path):
    model_file = tmp_path / 'custom_gtr.txt'
    model_file.write_text(str(GTR.standard('JC69', alphabet='nuc')), encoding='utf-8')
    params = SimpleNamespace(
        aa=False,
        custom_gtr=str(model_file),
        gtr='infer',
        gtr_params=None,
    )

    loaded = create_gtr(params)

    assert isinstance(loaded, GTR)
    assert params.gtr == 'infer'


def test_custom_gtr_disables_vcf_fixed_frequency_inference(monkeypatch):
    monkeypatch.setattr(
        'treetime.CLI_io.read_vcf',
        lambda *_: {'sequences': {'sample': {0: 'C'}}, 'reference': 'ACGT'},
    )
    params = SimpleNamespace(
        aa=False,
        aln='input.vcf',
        custom_gtr='model.txt',
        gtr='infer',
        vcf_reference='reference.fasta',
    )

    _, _, fixed_pi = read_if_vcf(params)

    assert fixed_pi is None


def test_toplevel_cli_propagates_success_and_reports_missing_inputs(monkeypatch, capsys):
    parser = argument_parser_module.make_parser()
    missing = parser.parse_args([])
    assert missing.func(missing) == 1
    assert 'REQUIRED inputs' in capsys.readouterr().out

    monkeypatch.setattr(argument_parser_module, 'timetree', lambda params: 7)
    valid = parser.parse_args(['--tree', 'tree.nwk', '--dates', 'dates.tsv'])
    assert valid.func(valid) == 7


def test_posterior_model_validation_and_immutability():
    model = SiteRateModel(
        mean_rates=[0.5, 1.5],
        posterior_weights=[[1.0, 0.0], [0.25, 0.75]],
        category_rates=[[0.5, 1.5], [0.0, 2.0]],
        category_prior_weights=[[0.5, 0.5], [0.5, 0.5]],
        category_mask=[[True, True], [True, True]],
        partition_index=[0, 0],
        partition_site=[1, 2],
        partition_names=('alignment',),
        partition_speeds=[1.0],
        evaluation_mode='posterior-elbo',
        metadata={'source': 'test'},
    )

    assert model.sequence_length == 2
    assert model.posterior_weights.flags.writeable is False
    with pytest.raises(ValueError):
        model.mean_rates[0] = 1.0
    with pytest.raises(TypeError):
        model.metadata['source'] = 'changed'


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        ({'evaluation_mode': 'unknown'}, 'unknown site-rate evaluation mode'),
        ({'mean_rates': [0.4, 1.5]}, 'global mean one'),
        ({'posterior_weights': [[0.9, 0.0], [0.25, 0.75]]}, 'rows must sum'),
        ({'category_prior_weights': [[0.4, 0.5], [0.5, 0.5]]}, 'prior rows must sum'),
        ({'category_mask': [[True, False], [True, True]]}, 'padded category prior'),
        (
            {
                'posterior_weights': [[1.0, 0.0], [0.25, 0.75]],
                'category_prior_weights': [[1.0, 0.0], [0.0, 1.0]],
            },
            'positive posterior weight requires positive',
        ),
        ({'category_rates': [[0.6, 1.5], [0.0, 2.0]]}, 'do not match'),
    ],
)
def test_posterior_model_rejects_incoherent_arrays(changes, message):
    arguments = {
        'mean_rates': [0.5, 1.5],
        'posterior_weights': [[1.0, 0.0], [0.25, 0.75]],
        'category_rates': [[0.5, 1.5], [0.0, 2.0]],
        'category_prior_weights': [[0.5, 0.5], [0.5, 0.5]],
        'category_mask': [[True, True], [True, True]],
        'partition_index': [0, 0],
        'partition_site': [1, 2],
        'partition_names': ('alignment',),
        'partition_speeds': [1.0],
        'evaluation_mode': 'posterior-elbo',
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        SiteRateModel(**arguments)


def _one_hot_profiles(gtr, parent_states, child_states):
    states = np.eye(len(gtr.alphabet))
    return states[np.asarray(parent_states)], states[np.asarray(child_states)]


def test_elbo_uniform_category_matches_scalar_gtr():
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = SiteRateModel(
        mean_rates=[1.0, 1.0, 1.0],
        posterior_weights=[[1.0], [1.0], [1.0]],
        category_rates=[[1.0], [1.0], [1.0]],
        category_prior_weights=[[1.0], [1.0], [1.0]],
        category_mask=[[True], [True], [True]],
        partition_index=[0, 0, 0],
        partition_site=[1, 2, 3],
        partition_names=('alignment',),
        partition_speeds=[1.0],
        evaluation_mode='posterior-elbo',
    )
    profiles = _one_hot_profiles(base_gtr, [0, 1, 2], [1, 1, 3])
    multiplicity = np.array([1.0, 2.0, 1.0])

    observed = model.prob_t_profiles_elbo(base_gtr, profiles, multiplicity, 0.1, return_log=True)
    expected = base_gtr.prob_t_profiles(profiles, multiplicity, 0.1, return_log=True)

    assert observed == pytest.approx(expected, abs=1e-12)


def test_elbo_one_hot_posteriors_match_tier_a_site_gtr():
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = SiteRateModel(
        mean_rates=[0.5, 1.5],
        posterior_weights=[[1.0, 0.0], [0.0, 1.0]],
        category_rates=[[0.5, 1.5], [0.5, 1.5]],
        category_prior_weights=[[0.5, 0.5], [0.5, 0.5]],
        category_mask=[[True, True], [True, True]],
        partition_index=[0, 0],
        partition_site=[1, 2],
        partition_names=('alignment',),
        partition_speeds=[1.0],
        evaluation_mode='posterior-elbo',
    )
    tier_a_gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
    profiles = _one_hot_profiles(base_gtr, [0, 1], [1, 1])

    elbo = model.prob_t_profiles_elbo(scalar_gtr, profiles, np.ones(2), 0.15, return_log=True)
    tier_a = tier_a_gtr.prob_t_profiles(profiles, np.ones(2), 0.15, return_log=True)

    assert elbo == pytest.approx(tier_a, abs=1e-12)


def test_elbo_retains_zero_rate_category_and_is_permutation_invariant():
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    arguments = {
        'mean_rates': [0.0, 2.0],
        'posterior_weights': [[1.0, 0.0], [0.0, 1.0]],
        'category_rates': [[0.0, 2.0], [0.0, 2.0]],
        'category_prior_weights': [[0.25, 0.75], [0.25, 0.75]],
        'category_mask': [[True, True], [True, True]],
        'partition_index': [0, 0],
        'partition_site': [1, 2],
        'partition_names': ('alignment',),
        'partition_speeds': [1.0],
        'evaluation_mode': 'posterior-elbo',
    }
    model = SiteRateModel(**arguments)
    permuted = SiteRateModel(
        **{
            **arguments,
            'posterior_weights': np.asarray(arguments['posterior_weights'])[:, ::-1],
            'category_rates': np.asarray(arguments['category_rates'])[:, ::-1],
            'category_prior_weights': np.asarray(arguments['category_prior_weights'])[:, ::-1],
            'category_mask': np.asarray(arguments['category_mask'])[:, ::-1],
        }
    )
    profiles = _one_hot_profiles(base_gtr, [0, 0], [0, 1])

    observed = model.prob_t_profiles_elbo(base_gtr, profiles, np.ones(2), 0.3, return_log=True)
    permuted_value = permuted.prob_t_profiles_elbo(base_gtr, profiles, np.ones(2), 0.3, return_log=True)

    assert np.isfinite(observed)
    assert observed == pytest.approx(permuted_value, abs=1e-12)


def test_elbo_large_rate_transition_matches_matrix_exponential():
    from scipy.linalg import expm

    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = SiteRateModel(
        mean_rates=[1.0],
        posterior_weights=[[1.0]],
        category_rates=[[1.0]],
        category_prior_weights=[[1.0]],
        category_mask=[[True]],
        partition_index=[0],
        partition_site=[1],
        partition_names=('alignment',),
        partition_speeds=[1.0],
        evaluation_mode='posterior-elbo',
    )
    profiles = _one_hot_profiles(base_gtr, [0], [1])
    branch_length = 8.0
    observed = model.prob_t_profiles_elbo(base_gtr, profiles, [1.0], branch_length, return_log=True)
    reference = expm(base_gtr.Q * base_gtr.mu * branch_length)
    expected = np.log(reference[1, 0] + 1e-24)

    assert observed == pytest.approx(expected, abs=1e-12)


def test_site_specific_large_rate_transition_matches_matrix_exponential():
    from scipy.linalg import expm

    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = SiteRateModel(
        mean_rates=[0.2, 1.8],
        partition_index=[0, 0],
        partition_site=[1, 2],
        partition_names=('alignment',),
        partition_speeds=[1.0],
    )
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
    branch_length = 8.0
    observed = site_gtr.expQt(branch_length)

    for site, rate in enumerate(model.mean_rates):
        reference = expm(scalar_gtr.Q * scalar_gtr.mu * branch_length * rate)
        np.testing.assert_allclose(observed[:, :, site], reference, atol=1e-12)
    assert np.all(observed >= 0)


def test_variational_identity_is_tight_at_guide_posterior():
    prior = np.array([0.4, 0.6])
    likelihood = np.array([0.2, 0.8])
    marginal = float(np.dot(prior, likelihood))
    posterior = prior * likelihood / marginal
    bound = np.sum(posterior * np.log(likelihood)) + np.sum(posterior * (np.log(prior) - np.log(posterior)))

    assert bound == pytest.approx(np.log(marginal), abs=1e-12)

    diffuse = np.array([0.5, 0.5])
    diffuse_bound = np.sum(diffuse * np.log(likelihood)) + np.sum(diffuse * (np.log(prior) - np.log(diffuse)))
    assert diffuse_bound < np.log(marginal)


def test_frozen_elbo_matches_guide_gradient_but_omits_score_curvature():
    prior = np.array([0.5, 0.5])
    category_scores = np.array([-1.0, 2.0])
    guide_parameter = 0.3
    guide_likelihood = np.exp(category_scores * guide_parameter)
    guide_posterior = prior * guide_likelihood / np.dot(prior, guide_likelihood)

    def exact_objective(parameter):
        return np.log(np.dot(prior, np.exp(category_scores * parameter)))

    constant = np.sum(guide_posterior * (np.log(prior) - np.log(guide_posterior)))

    def frozen_objective(parameter):
        return np.dot(guide_posterior, category_scores * parameter) + constant

    step = 1e-4
    exact_gradient = (exact_objective(guide_parameter + step) - exact_objective(guide_parameter - step)) / (2 * step)
    frozen_gradient = (frozen_objective(guide_parameter + step) - frozen_objective(guide_parameter - step)) / (2 * step)
    exact_curvature = (
        exact_objective(guide_parameter + step)
        - 2 * exact_objective(guide_parameter)
        + exact_objective(guide_parameter - step)
    ) / step**2
    frozen_curvature = (
        frozen_objective(guide_parameter + step)
        - 2 * frozen_objective(guide_parameter)
        + frozen_objective(guide_parameter - step)
    ) / step**2

    assert frozen_gradient == pytest.approx(exact_gradient, abs=1e-8)
    assert exact_curvature > 0
    assert frozen_curvature == pytest.approx(0.0, abs=1e-8)


def test_guide_scale_perturbation_quantifies_bias_and_conditional_coverage():
    """Predeclared two-rate Gaussian analogue of the frozen-responsibility handoff."""
    rng = np.random.default_rng(20260705)
    rates = np.array([0.5, 1.5])
    prior = np.array([0.5, 0.5])
    true_scale = 1.0
    noise = 0.35
    site_count = 200
    replicates = 400
    metrics = {}

    for relative_guide_error in (-0.1, 0.0, 0.1):
        estimates = []
        covered = []
        guide_scale = true_scale * (1 + relative_guide_error)
        for _ in range(replicates):
            categories = rng.choice(2, size=site_count, p=prior)
            observations = true_scale * rates[categories] + rng.normal(0, noise, size=site_count)
            log_responsibility = (
                np.log(prior)[None, :] - 0.5 * ((observations[:, None] - guide_scale * rates[None, :]) / noise) ** 2
            )
            log_responsibility -= log_responsibility.max(axis=1, keepdims=True)
            responsibility = np.exp(log_responsibility)
            responsibility /= responsibility.sum(axis=1, keepdims=True)

            curvature = np.sum(responsibility * rates[None, :] ** 2) / noise**2
            estimate = np.sum(responsibility * rates[None, :] * observations[:, None]) / noise**2 / curvature
            standard_error = 1 / np.sqrt(curvature)
            estimates.append(estimate)
            covered.append(abs(estimate - true_scale) <= 1.96 * standard_error)

        metrics[relative_guide_error] = (
            float(np.mean(estimates) - true_scale),
            float(np.mean(covered)),
        )

    guide_bias, guide_coverage = metrics[0.0]
    low_bias, low_coverage = metrics[-0.1]
    high_bias, high_coverage = metrics[0.1]
    assert abs(guide_bias) < 0.005
    assert 0.95 <= guide_coverage <= 1.0
    assert -0.05 < low_bias < -0.03
    assert 0.03 < high_bias < 0.05
    assert 0.5 < low_coverage < 0.7
    assert 0.5 < high_coverage < 0.7


def test_elbo_gap_weighting_and_negative_length_match_tree_time_contract():
    from treetime import config as ttconf

    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = SiteRateModel(
        mean_rates=[1.0, 1.0],
        posterior_weights=[[1.0], [1.0]],
        category_rates=[[1.0], [1.0]],
        category_prior_weights=[[1.0], [1.0]],
        category_mask=[[True], [True]],
        partition_index=[0, 0],
        partition_site=[1, 2],
        partition_names=('alignment',),
        partition_speeds=[1.0],
        evaluation_mode='posterior-elbo',
    )
    profiles = _one_hot_profiles(base_gtr, [base_gtr.gap_index, 0], [1, 1])
    observed = model.prob_t_profiles_elbo(base_gtr, profiles, [1.0, 1.0], 0.1, return_log=True)
    expected = base_gtr.prob_t_profiles(
        (profiles[0][1:], profiles[1][1:]),
        np.ones(1),
        0.1,
        return_log=True,
    )

    assert observed == pytest.approx(expected, abs=1e-12)
    assert model.prob_t_profiles_elbo(base_gtr, profiles, [1.0, 1.0], -0.1, return_log=True) == -ttconf.BIG_NUMBER
