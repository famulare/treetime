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
from treetime.iqtree_site_rates import (
    load_iqtree_site_rate_posteriors,
    load_iqtree_site_rates,
    parse_iqtree_partitions,
)
from treetime.site_rate_model import SiteRateModel, build_site_rate_gtrs
from treetime.wrappers import create_gtr, run_timetree


DATA = Path(__file__).parent / 'data' / 'site_rate_model'
REAL_IQTREE_DATA = Path(__file__).parent / 'data' / 'site_rate_iqtree_2_4_0'


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


# GTR captured from pre-refactor `master` infer_gtr on the fixture below (numpy 2.5.0).
# Proves the optional site_rate_weights refactor reproduces master's inference for the
# unweighted path. rtol accommodates cross-platform BLAS in the eigendecomposition.
_MASTER_UNWEIGHTED_GTR_GOLDEN = {
    False: {
        'mu': 1.0000000000000007,
        'Pi': [0.2502723163528559, 0.26865883203377944, 0.23447694169224087, 0.23447694169224087, 0.012114968228883045],
        'W': [
            [0.0, 1.375561021007432, 1.2696787634975544, 1.2696787634975544, 1.2713777824842694],
            [1.375561021007432, 0.0, 1.3858962314067746, 1.3858962314067746, 1.262119440667105],
            [1.2696787634975544, 1.3858962314067746, 0.0, 1.2771192170104886, 1.2784111704610566],
            [1.2696787634975544, 1.3858962314067746, 1.2771192170104886, 0.0, 1.2784111704610566],
            [1.2713777824842694, 1.262119440667105, 1.2784111704610566, 1.2784111704610566, 0.0],
        ],
    },
    True: {
        'mu': 0.9999999999999999,
        'Pi': [0.24737878512904185, 0.2751698302610791, 0.2312895302314521, 0.23404992748961032, 0.012111926888816658],
        'W': [
            [0.0, 1.3350431388766457, 1.3515239726582715, 1.2705840639416728, 1.2723305922898558],
            [1.3350431388766457, 0.0, 1.3451326185800545, 1.393659710923357, 1.2667938761507975],
            [1.3515239726582715, 1.3451326185800545, 0.0, 1.2780136605361254, 1.2793501056297625],
            [1.2705840639416728, 1.393659710923357, 1.2780136605361254, 0.0, 1.2793872927818202],
            [1.2723305922898558, 1.2667938761507975, 1.2793501056297625, 1.2793872927818202, 0.0],
        ],
    },
}


@pytest.mark.parametrize('marginal', [False, True])
def test_unweighted_gtr_inference_matches_explicit_unit_weights(marginal):
    def analysis():
        tree = Phylo.read(StringIO('(a:0.1,b:0.1,c:0.1);'), 'newick')
        alignment = MultipleSeqAlignment(
            [
                SeqRecord(Seq('AACC'), id='a'),
                SeqRecord(Seq('ACCC'), id='b'),
                SeqRecord(Seq('AGCT'), id='c'),
            ]
        )
        result = TreeAnc(
            tree=tree,
            aln=alignment,
            gtr=GTR.standard('JC69', alphabet='nuc'),
            compress=False,
            verbose=0,
            rng_seed=3,
        )
        result.infer_ancestral_sequences('probabilistic', marginal=marginal)
        return result

    legacy = analysis().infer_gtr(marginal=marginal)
    weighted = analysis().infer_gtr(marginal=marginal, site_rate_weights=np.ones(4))

    # The default (weights-absent) path is byte-identical to explicit unit weights:
    # multiplying exposure by 1.0 is IEEE-exact.
    assert legacy.mu == weighted.mu
    np.testing.assert_array_equal(legacy.Pi, weighted.Pi)
    np.testing.assert_array_equal(legacy.W, weighted.W)
    assert str(legacy) == str(weighted)

    # The refactored inference reproduces pre-refactor `master` (regression golden).
    golden = _MASTER_UNWEIGHTED_GTR_GOLDEN[marginal]
    np.testing.assert_allclose(legacy.mu, golden['mu'], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(legacy.Pi, golden['Pi'], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(legacy.W, golden['W'], rtol=1e-8, atol=1e-10)


def test_shared_gtr_inference_weights_exposure_by_known_site_rate(monkeypatch):
    tree = Phylo.read(StringIO('(a:0.1,b:0.1,c:0.1);'), 'newick')
    alignment = MultipleSeqAlignment(
        [
            SeqRecord(Seq('AACC'), id='a'),
            SeqRecord(Seq('AACC'), id='b'),
            SeqRecord(Seq('AACC'), id='c'),
        ]
    )
    tree_anc = TreeAnc(
        tree=tree,
        aln=alignment,
        gtr=GTR.standard('JC69', alphabet='nuc'),
        compress=False,
        verbose=0,
    )
    tree_anc.infer_ancestral_sequences('probabilistic', marginal=False)
    captured = {}
    original_infer = GTR.infer.__func__

    def capture_exposure(cls, nij, Ti, root_state, **kwargs):
        captured['exposure'] = Ti.copy()
        return original_infer(cls, nij, Ti, root_state, **kwargs)

    monkeypatch.setattr(GTR, 'infer', classmethod(capture_exposure))
    tree_anc.infer_gtr(
        marginal=False,
        site_rate_weights=np.array([0.5, 0.5, 1.5, 1.5]),
    )

    exposure = captured['exposure']
    assert exposure[tree_anc.gtr.state_index['C']] == pytest.approx(3 * exposure[tree_anc.gtr.state_index['A']])


def test_weighted_joint_gtr_inference_scales_mutation_midpoint_exposure(monkeypatch):
    tree = Phylo.read(StringIO('(a:0.1,b:0.1,c:0.1);'), 'newick')
    alignment = MultipleSeqAlignment(
        [
            SeqRecord(Seq('A'), id='a'),
            SeqRecord(Seq('A'), id='b'),
            SeqRecord(Seq('C'), id='c'),
        ]
    )
    tree_anc = TreeAnc(
        tree=tree,
        aln=alignment,
        gtr=GTR.standard('JC69', alphabet='nuc'),
        compress=False,
        verbose=0,
    )
    tree_anc.infer_ancestral_sequences('probabilistic', marginal=False)
    captured = {}
    original_infer = GTR.infer.__func__

    def capture_exposure(cls, nij, Ti, root_state, **kwargs):
        captured['exposure'] = Ti.copy()
        return original_infer(cls, nij, Ti, root_state, **kwargs)

    monkeypatch.setattr(GTR, 'infer', classmethod(capture_exposure))
    tree_anc.infer_gtr(
        marginal=False,
        site_rate_weights=np.array([0.25]),
    )

    exposure = captured['exposure']
    assert exposure[tree_anc.gtr.state_index['A']] == pytest.approx(0.0625)
    assert exposure[tree_anc.gtr.state_index['C']] == pytest.approx(0.0125)
    assert np.all(exposure >= 0)


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


def test_partition_mapping_invariant_uses_each_source_row_exactly_once(tmp_path):
    source_rates = {
        (1, 1): 0.4,
        (1, 2): 0.9,
        (1, 3): 1.4,
        (2, 1): 2.2,
        (2, 2): 1.3,
        (2, 3): 0.6,
    }
    rate_file = tmp_path / 'distinct.rate'
    rate_file.write_text(
        'Part\tSite\tRate\n'
        + ''.join(f'{partition}\t{site}\t{rate}\n' for (partition, site), rate in source_rates.items()),
        encoding='utf-8',
    )
    model = load_iqtree_site_rates(
        rate_file,
        sequence_length=6,
        partition_file=DATA / 'partitioned.best_model.nex',
        report_file=DATA / 'partitioned.iqtree',
    )
    _, coordinates = parse_iqtree_partitions(DATA / 'partitioned.best_model.nex', sequence_length=6)
    scaled_rates = np.empty(6)
    for (partition, local_site), raw_rate in source_rates.items():
        global_coordinate = coordinates[partition - 1][local_site - 1]
        scaled_rates[global_coordinate] = raw_rate * model.partition_speeds[partition - 1]

    assert len(np.unique(scaled_rates)) == 6
    np.testing.assert_allclose(model.mean_rates, scaled_rates / scaled_rates.mean())


def test_real_iqtree_partition_outputs_preserve_writer_units_and_coordinates():
    tier_a = load_iqtree_site_rates(
        REAL_IQTREE_DATA / 'generated.rate',
        sequence_length=50,
        partition_file=REAL_IQTREE_DATA / 'generated.best_model.nex',
        report_file=REAL_IQTREE_DATA / 'generated.iqtree',
    )
    tier_b = load_iqtree_site_rate_posteriors(
        REAL_IQTREE_DATA / 'generated.siteprob',
        sequence_length=50,
        partition_file=REAL_IQTREE_DATA / 'generated.best_model.nex',
        report_file=REAL_IQTREE_DATA / 'generated.iqtree',
        rate_file=REAL_IQTREE_DATA / 'generated.rate',
    )
    _, coordinates = parse_iqtree_partitions(REAL_IQTREE_DATA / 'generated.best_model.nex', sequence_length=50)
    assert tuple(len(partition) for partition in coordinates) == (35, 15)
    np.testing.assert_allclose(tier_a.partition_speeds, [0.1246, 3.0425])
    np.testing.assert_allclose(tier_b.partition_speeds, [0.1246, 3.0425])

    raw_rows = []
    for line in (REAL_IQTREE_DATA / 'generated.rate').read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            raw_rows.append((int(fields[0]), int(fields[1]), float(fields[2])))
    assert len(raw_rows) == 50
    raw_partition_means = [
        np.mean([rate for partition, _, rate in raw_rows if partition == partition_id]) for partition_id in (1, 2)
    ]
    np.testing.assert_allclose(raw_partition_means, [1.0, 1.0], atol=1e-2)
    assert not np.allclose(raw_partition_means, tier_a.partition_speeds, atol=1e-2)

    scaled_rates = np.empty(50)
    for partition, local_site, raw_rate in raw_rows:
        global_coordinate = coordinates[partition - 1][local_site - 1]
        scaled_rates[global_coordinate] = raw_rate * tier_a.partition_speeds[partition - 1]
        assert tier_a.partition_index[global_coordinate] == partition - 1
        assert tier_a.partition_site[global_coordinate] == local_site
    np.testing.assert_allclose(tier_a.mean_rates, scaled_rates / scaled_rates.mean())


def test_real_iqtree_interleaved_outputs_map_each_source_row_to_global_axis():
    data = Path(__file__).parent / 'data' / 'site_rate_iqtree_2_4_0_interleaved'
    tier_a = load_iqtree_site_rates(
        data / 'generated.rate',
        sequence_length=50,
        partition_file=data / 'generated.best_model.nex',
        report_file=data / 'generated.iqtree',
    )
    tier_b = load_iqtree_site_rate_posteriors(
        data / 'generated.siteprob',
        report_file=data / 'generated.iqtree',
        sequence_length=50,
        partition_file=data / 'generated.best_model.nex',
    )
    _, coordinates = parse_iqtree_partitions(data / 'generated.best_model.nex', sequence_length=50)

    # interleaved codon charsets: partition rows do not occupy contiguous global blocks
    assert tuple(len(part) for part in coordinates) == (17, 17, 16)
    assert (coordinates[0][0], coordinates[1][0], coordinates[2][0]) == (0, 1, 2)
    np.testing.assert_allclose(tier_a.partition_speeds, [0.1936, 2.5699, 0.1888])
    np.testing.assert_allclose(tier_b.partition_speeds, [0.1936, 2.5699, 0.1888])

    raw_rows = []
    for line in (data / 'generated.rate').read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            raw_rows.append((int(fields[0]), int(fields[1]), float(fields[2])))
    assert len(raw_rows) == 50
    raw_partition_means = [
        np.mean([rate for part, _, rate in raw_rows if part == partition_id]) for partition_id in (1, 2, 3)
    ]
    np.testing.assert_allclose(raw_partition_means, [1.0, 1.0, 1.0], atol=1e-2)
    assert not np.allclose(raw_partition_means, tier_a.partition_speeds, atol=1e-2)

    scaled_rates = np.empty(50)
    for partition, local_site, raw_rate in raw_rows:
        global_coordinate = coordinates[partition - 1][local_site - 1]
        scaled_rates[global_coordinate] = raw_rate * tier_a.partition_speeds[partition - 1]
        assert tier_a.partition_index[global_coordinate] == partition - 1
        assert tier_a.partition_site[global_coordinate] == local_site
    np.testing.assert_allclose(tier_a.mean_rates, scaled_rates / scaled_rates.mean())


def test_edge_equal_model_uses_reported_unit_speeds():
    model = load_iqtree_site_rates(
        DATA / 'partitioned.rate',
        partition_file=DATA / 'partitioned.best_model.nex',
        report_file=DATA / 'edge_equal.iqtree',
    )

    np.testing.assert_allclose(model.partition_speeds, [1.0, 1.0])
    assert model.metadata['partition_model'] == 'edge-linked-equal'


def test_unpartitioned_posteriors_build_normalized_category_model():
    model = load_iqtree_site_rate_posteriors(
        DATA / 'unpartitioned.siteprob',
        report_file=DATA / 'unpartitioned.iqtree',
        rate_file=DATA / 'unpartitioned.rate',
    )

    np.testing.assert_allclose(model.mean_rates, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(model.category_rates, [[0.5, 1.5]] * 3)
    np.testing.assert_allclose(model.category_prior_weights, [[0.5, 0.5]] * 3)
    assert model.evaluation_mode == 'posterior-elbo'


def test_unpartitioned_report_length_must_match_posteriors(tmp_path):
    report = tmp_path / 'wrong_length.iqtree'
    report.write_text(
        (DATA / 'unpartitioned.iqtree')
        .read_text(encoding='utf-8')
        .replace('with 3 nucleotide sites', 'with 4 nucleotide sites'),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='alignment length'):
        load_iqtree_site_rate_posteriors(
            DATA / 'unpartitioned.siteprob',
            report_file=report,
        )


def test_unpartitioned_report_must_identify_freerate_model(tmp_path):
    report = tmp_path / 'gamma.iqtree'
    report.write_text(
        (DATA / 'unpartitioned.iqtree').read_text(encoding='utf-8').replace('GTR+F+R2', 'GTR+F+G2'),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=r'requires one IQ-TREE \+R'):
        load_iqtree_site_rate_posteriors(
            DATA / 'unpartitioned.siteprob',
            report_file=report,
        )


def test_mixture_class_siteprob_is_rejected_even_when_width_matches_freerate():
    with pytest.raises(ValueError, match=r'substitution-mixture.*ambiguous'):
        load_iqtree_site_rate_posteriors(
            DATA / 'unpartitioned.siteprob',
            report_file=DATA / 'mixture_wspm.iqtree',
        )


def test_partition_mixture_model_is_rejected_even_when_width_matches_freerate(tmp_path):
    model_file = tmp_path / 'mixture.best_model.nex'
    model_file.write_text(
        (DATA / 'partitioned.best_model.nex')
        .read_text(encoding='utf-8')
        .replace(
            'GTR+F+R2{0.5,0.5,0.5,1.5}',
            'C2+R2{0.5,0.5,0.5,1.5}',
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=r'substitution-mixture.*ambiguous'):
        load_iqtree_site_rate_posteriors(
            DATA / 'partitioned.siteprob',
            report_file=DATA / 'partitioned.iqtree',
            partition_file=model_file,
        )


def test_partitioned_posteriors_map_unequal_category_counts_and_speeds():
    model = load_iqtree_site_rate_posteriors(
        DATA / 'partitioned.siteprob',
        report_file=DATA / 'partitioned.iqtree',
        partition_file=DATA / 'partitioned.best_model.nex',
        rate_file=DATA / 'partitioned.rate',
    )

    scaled_before_normalization = np.array([1.0, 1.0, 2.0, 0.5, 3.0, 0.25])
    normalization = scaled_before_normalization.mean()
    np.testing.assert_allclose(model.mean_rates, scaled_before_normalization / normalization)
    np.testing.assert_array_equal(model.partition_index, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(
        model.category_mask,
        [
            [True, True, False],
            [True, True, True],
            [True, True, False],
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ],
    )
    np.testing.assert_allclose(model.posterior_weights[:, 2], [0, 1, 0, 0, 0, 0])
    assert model.metadata['category_counts'] == (2, 3)


def test_elbo_reuses_transition_matrices_by_partition_rate_group(monkeypatch):
    model = load_iqtree_site_rate_posteriors(
        DATA / 'partitioned.siteprob',
        report_file=DATA / 'partitioned.iqtree',
        sequence_length=6,
        partition_file=DATA / 'partitioned.best_model.nex',
    )
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    original_expqt = base_gtr.expQt
    calls = []

    def counted_expqt(branch_length):
        calls.append(branch_length)
        return original_expqt(branch_length)

    monkeypatch.setattr(base_gtr, 'expQt', counted_expqt)
    profiles = np.full((2, 6, len(base_gtr.alphabet)), 1 / len(base_gtr.alphabet))
    model.prob_t_profiles_elbo(
        base_gtr,
        profiles,
        np.ones(6),
        0.1,
        return_log=True,
    )

    expected_calls = sum(len(valid_categories) for _, valid_categories in model._rate_groups)
    assert len(calls) == expected_calls
    assert len(calls) < model.sequence_length * model.posterior_weights.shape[1]


def test_elbo_is_invariant_to_partition_row_order_after_mapping():
    model = load_iqtree_site_rate_posteriors(
        DATA / 'partitioned.siteprob',
        report_file=DATA / 'partitioned.iqtree',
        partition_file=DATA / 'partitioned.best_model.nex',
    )
    permuted = replace(
        model,
        partition_index=1 - model.partition_index,
        partition_names=model.partition_names[::-1],
        partition_speeds=model.partition_speeds[::-1],
    )
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    rng = np.random.default_rng(29)
    profiles = (
        rng.dirichlet(np.ones(len(base_gtr.alphabet)), size=model.sequence_length),
        rng.dirichlet(np.ones(len(base_gtr.alphabet)), size=model.sequence_length),
    )
    observed = model.prob_t_profiles_elbo(base_gtr, profiles, np.ones(model.sequence_length), 0.1, return_log=True)
    permuted_value = permuted.prob_t_profiles_elbo(
        base_gtr,
        profiles,
        np.ones(model.sequence_length),
        0.1,
        return_log=True,
    )
    assert observed == pytest.approx(permuted_value, abs=1e-12)


def test_rate_and_posterior_cross_check_rejects_category_mismatch(tmp_path):
    rate_file = tmp_path / 'wrong.rate'
    rate_file.write_text(
        (DATA / 'partitioned.rate').read_text(encoding='utf-8').replace('1\t2\t1.00000', '1\t2\t1.20000'),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='cross-check failed'):
        load_iqtree_site_rate_posteriors(
            DATA / 'partitioned.siteprob',
            report_file=DATA / 'partitioned.iqtree',
            partition_file=DATA / 'partitioned.best_model.nex',
            rate_file=rate_file,
        )


def test_censored_rate_cross_check_fails_with_focused_message(tmp_path):
    rate_file = tmp_path / 'censored.rate'
    rate_file.write_text(
        (DATA / 'unpartitioned.rate').read_text(encoding='utf-8').replace('1\t0.50000', '1\t100.00000'),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='censors posterior mean rates'):
        load_iqtree_site_rate_posteriors(
            DATA / 'unpartitioned.siteprob',
            report_file=DATA / 'unpartitioned.iqtree',
            rate_file=rate_file,
        )


def test_siteprob_rows_allow_small_rounding_only(tmp_path):
    report = tmp_path / 'one_site.iqtree'
    report.write_text(
        (DATA / 'unpartitioned.iqtree')
        .read_text(encoding='utf-8')
        .replace('with 3 nucleotide sites', 'with 1 nucleotide sites'),
        encoding='utf-8',
    )
    rounded = tmp_path / 'rounded.siteprob'
    rounded.write_text(
        'Site\tp1\tp2\n1\t0.500001\t0.500001\n',
        encoding='utf-8',
    )
    model = load_iqtree_site_rate_posteriors(
        rounded,
        report_file=report,
        sequence_length=1,
    )
    assert model.metadata['renormalized_posterior_rows'] == 1

    malformed = tmp_path / 'malformed.siteprob'
    malformed.write_text('Site\tp1\tp2\n1\t0.7\t0.7\n', encoding='utf-8')
    with pytest.raises(ValueError, match='expected one'):
        load_iqtree_site_rate_posteriors(
            malformed,
            report_file=report,
        )


def test_unverified_invariant_output_is_rejected(tmp_path):
    report = tmp_path / 'invariant.iqtree'
    report.write_text(
        'Input data: 4 sequences with 3 nucleotide sites\n\n'
        'Model of substitution: GTR+F+I+R1\n\n'
        ' Category  Relative_rate  Proportion\n'
        '  0         0.0            0.2\n'
        '  1         1.25           0.8\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=r'\+I\+R'):
        load_iqtree_site_rate_posteriors(
            DATA / 'unpartitioned.siteprob',
            report_file=report,
        )


def _one_hot_profiles(gtr, parent_states, child_states):
    states = np.eye(len(gtr.alphabet))
    return states[np.asarray(parent_states)], states[np.asarray(child_states)]


def test_elbo_evaluator_matches_hand_calculation():
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = load_iqtree_site_rate_posteriors(
        DATA / 'unpartitioned.siteprob',
        report_file=DATA / 'unpartitioned.iqtree',
    )
    profiles = _one_hot_profiles(base_gtr, [0, 0, 0], [0, 1, 0])
    branch_length = 0.2

    observed = model.prob_t_profiles_elbo(
        base_gtr,
        profiles,
        np.ones(3),
        branch_length,
        return_log=True,
    )
    expected = 0.0
    for site in range(3):
        for category in range(2):
            weight = model.posterior_weights[site, category]
            if weight == 0:
                continue
            transition = base_gtr.expQt(branch_length * model.category_rates[site, category])
            expected += weight * np.log(transition[[0, 1, 0][site], 0] + 1e-24)

    assert observed == pytest.approx(expected)


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


def test_branch_interpolator_uses_elbo_without_node_model_state():
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = load_iqtree_site_rate_posteriors(
        DATA / 'unpartitioned.siteprob',
        report_file=DATA / 'unpartitioned.iqtree',
    )
    tier_a_gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
    profiles = _one_hot_profiles(base_gtr, [0, 0, 0], [0, 1, 0])
    node = SimpleNamespace(
        up=object(),
        mutation_length=0.1,
        profile_pair=profiles,
    )

    interpolator = BranchLenInterpolator(
        node,
        tier_a_gtr,
        one_mutation=1 / 3,
        branch_length_mode='marginal',
        pattern_multiplicity=np.ones(3),
        n_grid_points=20,
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
    )
    expected = np.asarray(
        [
            -model.prob_t_profiles_elbo(scalar_gtr, profiles, np.ones(3), value, return_log=True)
            for value in interpolator.x
        ]
    )

    np.testing.assert_allclose(interpolator.y, expected)
    assert not hasattr(node, 'site_rate_model')
    assert not hasattr(node, 'site_rate_posteriors')


def _tiny_treetime_inputs():
    from io import StringIO

    tree = Phylo.read(StringIO('((a:0.1,b:0.1):0.1,c:0.2);'), 'newick')
    alignment = MultipleSeqAlignment(
        [
            SeqRecord(Seq('AA'), id='a'),
            SeqRecord(Seq('AC'), id='b'),
            SeqRecord(Seq('CC'), id='c'),
        ]
    )
    return tree, alignment, {'a': 2000.0, 'b': 2001.0, 'c': 2002.0}


def test_treetime_owns_site_rate_model_and_forces_marginal_mode():
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
    tree, alignment, dates = _tiny_treetime_inputs()
    tree_time = TreeTime(
        tree=tree,
        aln=alignment,
        dates=dates,
        gtr=tier_a_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
    )

    tree_time._set_branch_length_mode('auto')
    assert tree_time.branch_length_mode == 'marginal'
    assert tree_time.site_rate_model is model
    assert all(not hasattr(node, 'site_rate_model') for node in tree_time.tree.find_clades())
    with pytest.raises(UnknownMethodError, match='require'):
        tree_time._set_branch_length_mode('joint')


def test_treetime_builds_date_constraints_with_analysis_owned_elbo_model():
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
    tree, alignment, dates = _tiny_treetime_inputs()
    tree_time = TreeTime(
        tree=tree,
        aln=alignment,
        dates=dates,
        gtr=tier_a_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
        verbose=0,
    )

    tree_time._set_branch_length_mode('marginal')
    tree_time.init_date_constraints(clock_rate=0.01)

    assert all(
        node.branch_length_interpolator.site_rate_model is model
        for node in tree_time.tree.find_clades()
        if node.up is not None
    )


def test_custom_gtr_scaling_is_compared_by_effective_generator():
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
    base_gtr = GTR.custom(
        mu=2.5,
        pi=pi,
        W=exchangeability,
        alphabet='nuc',
    )
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
    tree, alignment, dates = _tiny_treetime_inputs()

    tree_time = TreeTime(
        tree=tree,
        aln=alignment,
        dates=dates,
        gtr=tier_a_gtr,
        compress=False,
        branch_length_mode='marginal',
        site_rate_model=model,
        site_rate_base_gtr=scalar_gtr,
    )

    assert tree_time.site_rate_base_gtr is scalar_gtr


def test_mean_mode_rejects_global_site_gtr_scaling_that_confounds_the_clock():
    model = SiteRateModel(
        mean_rates=[0.5, 1.5],
        partition_index=[0, 0],
        partition_site=[1, 2],
        partition_names=('alignment',),
        partition_speeds=[1.0],
    )
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, GTR.standard('JC69', alphabet='nuc'))
    site_gtr._mu *= 2
    tree, alignment, dates = _tiny_treetime_inputs()

    with pytest.raises(ValueError, match='generators do not match'):
        TreeTime(
            tree=tree,
            aln=alignment,
            dates=dates,
            gtr=site_gtr,
            compress=False,
            branch_length_mode='marginal',
            site_rate_model=model,
            site_rate_base_gtr=scalar_gtr,
        )


def test_treetime_rejects_site_rate_length_mismatch():
    base_gtr = GTR.standard('JC69', alphabet='nuc')
    model = SiteRateModel(
        mean_rates=[1.0],
        partition_index=[0],
        partition_site=[1],
        partition_names=('alignment',),
        partition_speeds=[1.0],
    )
    tier_a_gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
    tree, alignment, dates = _tiny_treetime_inputs()
    with pytest.raises(ValueError, match='length'):
        TreeTime(
            tree=tree,
            aln=alignment,
            dates=dates,
            gtr=tier_a_gtr,
            compress=False,
            branch_length_mode='marginal',
            site_rate_model=model,
            site_rate_base_gtr=scalar_gtr,
        )


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


def test_polytomy_rate_uses_gap_excluding_scalar_normalization():
    model = load_iqtree_site_rates(DATA / 'unpartitioned.rate')
    alphabet = GTR.standard('JC69', alphabet='nuc').alphabet
    exchangeability = np.ones((len(alphabet), len(alphabet)))
    np.fill_diagonal(exchangeability, 0)
    gap_index = int(np.flatnonzero(alphabet == '-')[0])
    exchangeability[gap_index, :] = 100
    exchangeability[:, gap_index] = 100
    exchangeability[gap_index, gap_index] = 0
    base_gtr = GTR.custom(
        mu=1.0,
        pi=[0.24, 0.24, 0.24, 0.24, 0.04],
        W=exchangeability,
        alphabet=alphabet,
    )
    site_gtr, scalar_gtr = build_site_rate_gtrs(model, base_gtr)
    tree_time = TreeTime.__new__(TreeTime)
    tree_time.site_rate_model = model
    tree_time.site_rate_base_gtr = scalar_gtr
    tree_time._gtr = site_gtr
    tree_time.data = SimpleNamespace(full_length=model.sequence_length)

    assert not np.isclose(site_gtr.mu.sum(), model.sequence_length)
    assert tree_time._alignment_mutation_rate() == pytest.approx(model.sequence_length)


def test_polytomy_rate_matches_legacy_scalar_expression_exactly():
    tree_time = TreeTime.__new__(TreeTime)
    tree_time.site_rate_model = None
    tree_time._gtr = SimpleNamespace(mu=np.float64(0.125))
    tree_time.data = SimpleNamespace(full_length=37)

    expected = tree_time.gtr.mu * tree_time.data.full_length

    assert tree_time._alignment_mutation_rate() == expected


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
