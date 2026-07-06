"""Validated site-rate data used by TreeTime date inference."""

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import config as ttconf


_NORMALIZATION_TOLERANCE = 1e-8
_PROBABILITY_TOLERANCE = 1e-8


def _readonly_array(value, *, dtype, ndim, name):
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions, got shape {array.shape}')
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SiteRateModel:
    """Analysis-owned relative site-rate model.

    The first axis of every array is the original, uncompressed alignment
    coordinate. Rates are globally normalized so that ``mean_rates.mean()`` is
    one. Posterior/category arrays are optional and are required only for the
    frozen-responsibility ELBO evaluation mode.
    """

    mean_rates: np.ndarray
    partition_index: np.ndarray
    partition_site: np.ndarray
    partition_names: tuple
    partition_speeds: np.ndarray
    evaluation_mode: str = 'mean'
    posterior_weights: np.ndarray = None
    category_rates: np.ndarray = None
    category_prior_weights: np.ndarray = None
    category_mask: np.ndarray = None
    metadata: dict = None
    _rate_groups: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        mean_rates = _readonly_array(self.mean_rates, dtype=float, ndim=1, name='mean_rates')
        partition_index = _readonly_array(self.partition_index, dtype=int, ndim=1, name='partition_index')
        partition_site = _readonly_array(self.partition_site, dtype=int, ndim=1, name='partition_site')
        partition_speeds = _readonly_array(self.partition_speeds, dtype=float, ndim=1, name='partition_speeds')
        partition_names = tuple(self.partition_names)

        length = len(mean_rates)
        if length == 0:
            raise ValueError('site-rate model must contain at least one alignment site')
        if len(partition_index) != length or len(partition_site) != length:
            raise ValueError('partition arrays must have one entry per alignment site')
        if not np.all(np.isfinite(mean_rates)) or np.any(mean_rates < 0):
            raise ValueError('mean_rates must be finite and nonnegative')
        if not np.isclose(mean_rates.mean(), 1.0, atol=_NORMALIZATION_TOLERANCE, rtol=0):
            raise ValueError(f'mean_rates must have global mean one, got {mean_rates.mean():.12g}')
        if len(partition_names) == 0:
            raise ValueError('partition_names must not be empty')
        if len(partition_speeds) != len(partition_names):
            raise ValueError('partition_speeds must have one entry per partition')
        if not np.all(np.isfinite(partition_speeds)) or np.any(partition_speeds <= 0):
            raise ValueError('partition_speeds must be finite and positive')
        if np.any(partition_index < 0) or np.any(partition_index >= len(partition_names)):
            raise ValueError('partition_index contains an unknown partition')
        if np.any(partition_site < 1):
            raise ValueError('partition_site uses one-based positive coordinates')
        if self.evaluation_mode not in {'mean', 'posterior-elbo'}:
            raise ValueError(f'unknown site-rate evaluation mode: {self.evaluation_mode!r}')

        posterior_fields = (
            self.posterior_weights,
            self.category_rates,
            self.category_prior_weights,
            self.category_mask,
        )
        if any(value is not None for value in posterior_fields):
            if not all(value is not None for value in posterior_fields):
                raise ValueError('posterior site-rate models require all category arrays')
            posterior_weights = _readonly_array(self.posterior_weights, dtype=float, ndim=2, name='posterior_weights')
            category_rates = _readonly_array(self.category_rates, dtype=float, ndim=2, name='category_rates')
            category_prior_weights = _readonly_array(
                self.category_prior_weights,
                dtype=float,
                ndim=2,
                name='category_prior_weights',
            )
            category_mask = _readonly_array(self.category_mask, dtype=bool, ndim=2, name='category_mask')
            expected_shape = posterior_weights.shape
            if expected_shape[0] != length or expected_shape[1] == 0:
                raise ValueError('category arrays must have shape (alignment length, Kmax)')
            if any(array.shape != expected_shape for array in (category_rates, category_prior_weights, category_mask)):
                raise ValueError('all category arrays must have the same shape')
            for name, array in (
                ('posterior_weights', posterior_weights),
                ('category_rates', category_rates),
                ('category_prior_weights', category_prior_weights),
            ):
                if not np.all(np.isfinite(array)) or np.any(array < 0):
                    raise ValueError(f'{name} must be finite and nonnegative')
            if np.any(np.sum(category_mask, axis=1) == 0):
                raise ValueError('every site must have at least one valid rate category')
            if np.any(posterior_weights[~category_mask] != 0):
                raise ValueError('padded posterior weights must be zero')
            if np.any(category_prior_weights[~category_mask] != 0):
                raise ValueError('padded category prior weights must be zero')
            if np.any((posterior_weights > 0) & (category_prior_weights == 0)):
                raise ValueError('positive posterior weight requires positive category prior weight')
            posterior_sums = posterior_weights.sum(axis=1)
            prior_sums = category_prior_weights.sum(axis=1)
            if not np.allclose(posterior_sums, 1.0, atol=_PROBABILITY_TOLERANCE, rtol=0):
                raise ValueError('posterior weight rows must sum to one')
            if not np.allclose(prior_sums, 1.0, atol=_PROBABILITY_TOLERANCE, rtol=0):
                raise ValueError('category prior rows must sum to one')
            posterior_means = np.sum(posterior_weights * category_rates, axis=1)
            if not np.allclose(posterior_means, mean_rates, atol=_NORMALIZATION_TOLERANCE, rtol=1e-8):
                raise ValueError('mean_rates do not match posterior-weighted category rates')
        else:
            posterior_weights = None
            category_rates = None
            category_prior_weights = None
            category_mask = None
            if self.evaluation_mode != 'mean':
                raise ValueError('posterior-elbo mode requires category arrays')

        object.__setattr__(self, 'mean_rates', mean_rates)
        object.__setattr__(self, 'partition_index', partition_index)
        object.__setattr__(self, 'partition_site', partition_site)
        object.__setattr__(self, 'partition_names', partition_names)
        object.__setattr__(self, 'partition_speeds', partition_speeds)
        object.__setattr__(self, 'posterior_weights', posterior_weights)
        object.__setattr__(self, 'category_rates', category_rates)
        object.__setattr__(self, 'category_prior_weights', category_prior_weights)
        object.__setattr__(self, 'category_mask', category_mask)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata or {})))
        if category_rates is None:
            rate_groups = ()
        else:
            grouping_values = np.concatenate(
                [category_rates, category_mask.astype(float)],
                axis=1,
            )
            _, inverse = np.unique(grouping_values, axis=0, return_inverse=True)
            groups = []
            for group_index in range(int(inverse.max()) + 1):
                sites = np.flatnonzero(inverse == group_index)
                sites.setflags(write=False)
                valid_categories = tuple(np.flatnonzero(category_mask[sites[0]]))
                groups.append((sites, valid_categories))
            rate_groups = tuple(groups)
        object.__setattr__(self, '_rate_groups', rate_groups)

    @property
    def sequence_length(self):
        """Number of sites on the original alignment axis."""
        return len(self.mean_rates)

    def prob_t_profiles_elbo(
        self,
        base_gtr,
        profile_pair,
        multiplicity,
        t,
        *,
        return_log=False,
        ignore_gaps=True,
    ):
        """Evaluate the frozen-responsibility branch ELBO.

        This is the date-dependent term
        ``sum_i sum_k p_ik log P(profile_pair_i | t * rate_ik)``. Category
        responsibilities are fixed across branches; they are not remixed inside
        a branch likelihood.
        """
        if self.evaluation_mode != 'posterior-elbo':
            raise ValueError('prob_t_profiles_elbo requires posterior-elbo mode')
        if getattr(base_gtr, 'is_site_specific', False):
            raise ValueError('the ELBO evaluator requires a scalar base GTR')

        parent = np.asarray(profile_pair[0], dtype=float)
        child = np.asarray(profile_pair[1], dtype=float)
        expected_shape = (self.sequence_length, len(base_gtr.alphabet))
        if parent.shape != expected_shape or child.shape != expected_shape:
            raise ValueError(f'profile arrays must have shape {expected_shape}, got {parent.shape} and {child.shape}')
        if multiplicity is None:
            multiplicity = np.ones(self.sequence_length, dtype=float)
        else:
            multiplicity = np.asarray(multiplicity, dtype=float)
        if multiplicity.shape != (self.sequence_length,):
            raise ValueError('multiplicity must have one entry per alignment site')
        if not np.all(np.isfinite(multiplicity)) or np.any(multiplicity < 0):
            raise ValueError('multiplicity must be finite and nonnegative')

        if t < 0:
            log_probability = -ttconf.BIG_NUMBER
        else:
            site_log_probability = np.zeros(self.sequence_length, dtype=float)
            for sites, valid_categories in self._rate_groups:
                representative = sites[0]
                for category in valid_categories:
                    weights = self.posterior_weights[sites, category]
                    positive = weights > 0
                    if not np.any(positive):
                        continue
                    transition = base_gtr.expQt(float(t) * float(self.category_rates[representative, category]))
                    if transition.ndim != 2 or not np.all(np.isfinite(transition)) or np.any(transition < 0):
                        raise ValueError('base GTR produced an invalid transition matrix')
                    active_sites = sites[positive]
                    probabilities = np.einsum(
                        'ai,ij,aj->a',
                        child[active_sites],
                        transition,
                        parent[active_sites],
                    )
                    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
                        raise ValueError('profile transition probability is invalid')
                    site_log_probability[active_sites] += weights[positive] * np.log(
                        np.maximum(probabilities, ttconf.SUPERTINY_NUMBER)
                    )

            if ignore_gaps and base_gtr.gap_index is not None:
                gap_weight = (1 - parent[:, base_gtr.gap_index]) * (1 - child[:, base_gtr.gap_index])
            else:
                gap_weight = 1.0
            log_probability = float(np.sum(multiplicity * gap_weight * site_log_probability))

        return log_probability if return_log else np.exp(log_probability)


def build_site_rate_gtrs(site_rate_model, base_gtr):
    """Build the Tier A ancestral GTR and normalized scalar ELBO GTR."""
    if getattr(base_gtr, 'is_site_specific', False):
        raise ValueError('base_gtr must be a scalar GTR')

    from .gtr import GTR
    from .gtr_site_specific import GTR_site_specific

    exchangeability = 0.5 * (base_gtr.W + base_gtr.W.T)
    scalar_gtr = GTR.custom(
        mu=1.0,
        pi=base_gtr.Pi,
        W=exchangeability,
        alphabet=base_gtr.alphabet,
        prof_map=base_gtr.profile_map,
    )
    site_gtr = GTR_site_specific(
        seq_len=site_rate_model.sequence_length,
        approximate=False,
        alphabet=scalar_gtr.alphabet,
        prof_map=scalar_gtr.profile_map,
    )
    site_gtr.assign_rates(
        mu=site_rate_model.mean_rates,
        pi=scalar_gtr.Pi,
        W=scalar_gtr.W,
    )
    return site_gtr, scalar_gtr


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def write_site_rate_audit(site_rate_model, output_directory):
    """Write inspectable site mapping and metadata for one site-rate run."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    table_path = output_directory / 'site_rate_model.tsv'
    metadata_path = output_directory / 'site_rate_model.json'

    with table_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['site', 'partition', 'partition_site', 'partition_speed', 'posterior_mean_rate'])
        for coordinate in range(site_rate_model.sequence_length):
            partition = int(site_rate_model.partition_index[coordinate])
            writer.writerow(
                [
                    coordinate + 1,
                    site_rate_model.partition_names[partition],
                    int(site_rate_model.partition_site[coordinate]),
                    f'{site_rate_model.partition_speeds[partition]:.12g}',
                    f'{site_rate_model.mean_rates[coordinate]:.12g}',
                ]
            )

    from . import version

    category_counts = None
    if site_rate_model.category_mask is not None:
        category_counts = []
        for partition in range(len(site_rate_model.partition_names)):
            counts = np.unique(site_rate_model.category_mask[site_rate_model.partition_index == partition].sum(axis=1))
            category_counts.append(int(counts[0]) if len(counts) == 1 else counts.tolist())
    metadata = {
        'schema_version': 1,
        'treetime_version': version,
        'evaluation_mode': site_rate_model.evaluation_mode,
        'sequence_length': site_rate_model.sequence_length,
        'partition_names': site_rate_model.partition_names,
        'partition_speeds': site_rate_model.partition_speeds,
        'category_counts': category_counts,
        'global_mean_rate': float(site_rate_model.mean_rates.mean()),
        'validation_tolerances': {
            'model_normalization_absolute': _NORMALIZATION_TOLERANCE,
            'probability_row_absolute': _PROBABILITY_TOLERANCE,
            'iqtree_probability_sum_absolute': 1e-5,
            'rate_crosscheck_absolute': 2e-5,
            'rate_crosscheck_relative': 2e-4,
        },
        'source_metadata': site_rate_model.metadata,
    }
    metadata_path.write_text(json.dumps(_json_value(metadata), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return table_path, metadata_path
