from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from treetime import GTR
from treetime.gtr_site_specific import GTR_site_specific
from treetime.iqtree_site_rates import load_iqtree_site_rates, parse_iqtree_partitions
from treetime.site_rate_model import SiteRateModel
from treetime.wrappers import create_gtr, run_timetree


DATA = Path(__file__).parent / 'data' / 'site_rate_model'


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


def test_unpartitioned_rates_are_reordered_and_normalized(tmp_path):
    source = tmp_path / 'reordered.rate'
    source.write_text('Site\tRate\n3\t1.5\n1\t0.5\n2\t1.0\n', encoding='utf-8')

    model = load_iqtree_site_rates(source, sequence_length=3)

    np.testing.assert_allclose(model.mean_rates, [0.5, 1.0, 1.5])
    np.testing.assert_array_equal(model.partition_site, [1, 2, 3])
    assert model.mean_rates.flags.writeable is False


def test_partition_coordinates_preserve_interleaved_global_axis():
    names, coordinates = parse_iqtree_partitions(DATA / 'partitioned.best_model.nex', sequence_length=6)

    assert names == ('alpha_sites', 'beta-block')
    np.testing.assert_array_equal(coordinates[0], [0, 2, 4])
    np.testing.assert_array_equal(coordinates[1], [1, 3, 5])


def test_partition_names_follow_iqtree_plus_normalization(tmp_path):
    partition_file = tmp_path / 'plus.nex'
    partition_file.write_text(
        (DATA / 'partitioned.best_model.nex').read_text(encoding='utf-8').replace('beta-block', 'beta+block'),
        encoding='utf-8',
    )
    report_file = tmp_path / 'plus.iqtree'
    report_file.write_text(
        (DATA / 'partitioned.iqtree').read_text(encoding='utf-8').replace('beta-block', 'beta+block'),
        encoding='utf-8',
    )

    model = load_iqtree_site_rates(
        DATA / 'partitioned.rate',
        partition_file=partition_file,
        report_file=report_file,
    )

    assert model.partition_names == ('alpha_sites', 'beta_block')


def test_partition_name_normalization_collisions_are_rejected(tmp_path):
    partition_file = tmp_path / 'collision.nex'
    partition_file.write_text(
        '#nexus\nbegin sets;\ncharset a+b = 1;\ncharset a_b = 2;\nend;\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='after IQ-TREE normalization'):
        parse_iqtree_partitions(partition_file, sequence_length=2)


def test_partitioned_rates_map_local_sites_and_apply_speeds():
    model = load_iqtree_site_rates(
        DATA / 'partitioned.rate',
        sequence_length=6,
        partition_file=DATA / 'partitioned.best_model.nex',
        report_file=DATA / 'partitioned.iqtree',
    )

    scaled_before_normalization = np.array([1.0, 1.0, 2.0, 0.5, 3.0, 0.25])
    expected_constant = scaled_before_normalization.mean()
    np.testing.assert_allclose(model.mean_rates, scaled_before_normalization / expected_constant)
    np.testing.assert_array_equal(model.partition_index, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(model.partition_site, [1, 1, 2, 2, 3, 3])
    np.testing.assert_allclose(model.partition_speeds, [2.0, 0.5])
    assert model.metadata['normalization_constant'] == pytest.approx(expected_constant)


def test_edge_equal_model_uses_reported_unit_speeds():
    model = load_iqtree_site_rates(
        DATA / 'partitioned.rate',
        partition_file=DATA / 'partitioned.best_model.nex',
        report_file=DATA / 'edge_equal.iqtree',
    )

    np.testing.assert_allclose(model.partition_speeds, [1.0, 1.0])
    assert model.metadata['partition_model'] == 'edge-linked-equal'


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        ('Site\tRate\n1\tnan\n', 'finite decimal'),
        ('Site\tRate\n1\t-0.1\n', 'finite decimal'),
        ('Site\tRate\n1\t1\n1\t2\n', 'duplicate site'),
        ('Site\tRate\n2\t1\n', 'missing alignment site'),
        ('Rate\n1\n', 'missing required column'),
    ],
)
def test_malformed_rate_rows_fail_without_fallback(tmp_path, text, message):
    source = tmp_path / 'invalid.rate'
    source.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        load_iqtree_site_rates(source)


def test_partitioned_rates_require_coordinate_and_speed_sources():
    with pytest.raises(ValueError, match='partition/model'):
        load_iqtree_site_rates(DATA / 'partitioned.rate')
    with pytest.raises(ValueError, match='partition speeds'):
        load_iqtree_site_rates(
            DATA / 'partitioned.rate',
            partition_file=DATA / 'partitioned.best_model.nex',
        )


def test_overlapping_partition_coordinates_are_rejected(tmp_path):
    source = tmp_path / 'overlap.nex'
    source.write_text(
        '#nexus\nbegin sets;\ncharset a = 1-3;\ncharset b = 3-4;\nend;\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='overlap'):
        parse_iqtree_partitions(source, sequence_length=4)


def test_edge_unlinked_partition_model_is_rejected(tmp_path):
    report = tmp_path / 'unlinked.iqtree'
    report.write_text(
        'SUBSTITUTION PROCESS\n--------------------\n\n'
        'Edge-unlinked partition model with separate substitution models\n\n'
        '  ID Model TreeLen Parameters\n'
        '  1 GTR+R2 0.1 -\n'
        '  2 HKY+R2 0.2 -\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='cannot be represented'):
        load_iqtree_site_rates(
            DATA / 'partitioned.rate',
            partition_file=DATA / 'partitioned.best_model.nex',
            report_file=report,
        )


@pytest.mark.parametrize(
    ('old', 'new', 'message'),
    [
        ('Input data: 4 taxa with 2 partitions and 6 total sites', '', 'summary is missing'),
        ('beta-block', 'wrong_partition', 'sizes or names'),
        ('\t4\t3\t3\t1\t1\t1\tbeta-block', '\t4\t2\t3\t1\t1\t1\tbeta-block', 'sizes or names'),
    ],
)
def test_partition_report_cross_checks_are_mandatory(tmp_path, old, new, message):
    report = tmp_path / 'mismatched.iqtree'
    text = (DATA / 'partitioned.iqtree').read_text(encoding='utf-8')
    report.write_text(text.replace(old, new), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        load_iqtree_site_rates(
            DATA / 'partitioned.rate',
            partition_file=DATA / 'partitioned.best_model.nex',
            report_file=report,
        )
