"""Validated site-rate data used by TreeTime date inference."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


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

    @property
    def sequence_length(self):
        """Number of sites on the original alignment axis."""
        return len(self.mean_rates)
