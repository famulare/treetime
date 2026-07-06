from types import SimpleNamespace

import pytest

from treetime import GTR
from treetime import argument_parser as argument_parser_module
from treetime.CLI_io import read_if_vcf
from treetime.gtr_site_specific import GTR_site_specific
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
