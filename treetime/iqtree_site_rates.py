"""Load IQ-TREE site-rate output onto the original alignment coordinate axis."""

from pathlib import Path
import re

import numpy as np
from scipy.special import gammainc, gammaincinv

from .site_rate_model import SiteRateModel


_INTEGER = re.compile(r'[1-9][0-9]*\Z')
_NUMBER = re.compile(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_COORDINATE_TOKEN = re.compile(r'\s*([0-9]+|[-,\\/])')
_NAMED_MIXTURE_MODELS = {
    'CF4',
    'EHO',
    'EX2',
    'EX3',
    'EX_EHO',
    'JTTCF4G',
    'LG4',
    'LG4M',
    'LG4X',
    'UL2',
    'UL3',
}


def _is_mixture_model_expression(expression):
    """Recognize IQ-TREE substitution-mixture syntax and built-in names."""
    expression = expression.upper()
    if 'MIX{' in expression:
        return True
    for component in re.split(r'[+*]', expression):
        name = component.split('{', maxsplit=1)[0]
        if name in _NAMED_MIXTURE_MODELS or re.fullmatch(r'C[0-9]+(?:OPT|TEST)?', name):
            return True
    return False


def _read_text(path, description):
    source = Path(path)
    if not source.is_file():
        raise ValueError(f'{description} does not exist: {source}')
    return source.read_text(encoding='utf-8')


def _parse_positive_integer(value, *, field, line_number):
    if not _INTEGER.fullmatch(value):
        raise ValueError(f'line {line_number}: {field} must be a positive integer, got {value!r}')
    return int(value)


def _parse_nonnegative_number(value, *, field, line_number):
    if not _NUMBER.fullmatch(value):
        raise ValueError(f'line {line_number}: {field} is not a finite decimal number: {value!r}')
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f'line {line_number}: {field} must be finite and nonnegative')
    return number


def _read_iqtree_table(path, required_columns):
    text = _read_text(path, 'IQ-TREE table')
    header = None
    records = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        columns = line.split()
        if header is None:
            header = columns
            missing = set(required_columns) - set(header)
            if missing:
                raise ValueError(f'{path}: header is missing required column(s): {", ".join(sorted(missing))}')
            if len(set(header)) != len(header):
                raise ValueError(f'{path}: header contains duplicate column names')
            continue
        if len(columns) != len(header):
            raise ValueError(f'{path}: line {line_number} has {len(columns)} fields; expected {len(header)}')
        records.append((line_number, dict(zip(header, columns))))
    if header is None:
        raise ValueError(f'{path}: no header found')
    if not records:
        raise ValueError(f'{path}: no data rows found')
    return header, records


def _nexus_statements(text):
    """Split NEXUS text into statements while respecting comments and quotes."""
    statements = []
    current = []
    quote = None
    comment_depth = 0
    index = 0
    while index < len(text):
        character = text[index]
        if comment_depth:
            if character == '[':
                comment_depth += 1
            elif character == ']':
                comment_depth -= 1
            index += 1
            continue
        if quote:
            current.append(character)
            if character == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    current.append(text[index + 1])
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {'"', "'"}:
            quote = character
            current.append(character)
        elif character == '[':
            comment_depth = 1
        elif character == ';':
            statement = ''.join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(character)
        index += 1
    if quote:
        raise ValueError('unterminated quote in NEXUS partition file')
    if comment_depth:
        raise ValueError('unterminated comment in NEXUS partition file')
    if ''.join(current).strip():
        raise ValueError('unterminated statement in NEXUS partition file')
    return statements


def _unquote_nexus_name(name):
    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in {'"', "'"}:
        quote = name[0]
        return name[1:-1].replace(quote * 2, quote)
    if not name or any(character.isspace() for character in name):
        raise ValueError(f'invalid unquoted NEXUS charset name: {name!r}')
    return name


def _normalize_partition_name(name):
    """Apply IQ-TREE's best-model charset name normalization."""
    return name.replace('+', '_')


def _coordinate_tokens(expression):
    tokens = []
    position = 0
    while position < len(expression):
        match = _COORDINATE_TOKEN.match(expression, position)
        if not match:
            if expression[position:].strip():
                raise ValueError(f'unsupported charset coordinate syntax: {expression!r}')
            break
        tokens.append(match.group(1))
        position = match.end()
    return tokens


def _parse_charset_coordinates(expression):
    """Parse IQ-TREE/NEXUS ranges into ordered one-based coordinates."""
    rhs = expression.strip()
    if ':' in rhs:
        rhs = rhs.rsplit(':', maxsplit=1)[1].strip()
    if ',' in rhs:
        prefix, remainder = rhs.split(',', maxsplit=1)
        if any(character.isalpha() for character in prefix):
            rhs = remainder.strip()

    tokens = _coordinate_tokens(rhs)
    coordinates = []
    index = 0
    while index < len(tokens):
        if tokens[index] == ',':
            index += 1
            continue
        if not tokens[index].isdigit():
            raise ValueError(f'expected charset coordinate, got {tokens[index]!r}')
        start = int(tokens[index])
        index += 1
        end = start
        step = 1
        if index < len(tokens) and tokens[index] == '-':
            index += 1
            if index >= len(tokens) or not tokens[index].isdigit():
                raise ValueError('charset range is missing its end coordinate')
            end = int(tokens[index])
            index += 1
            if index < len(tokens) and tokens[index] in {'\\', '/'}:
                index += 1
                if index >= len(tokens) or not tokens[index].isdigit():
                    raise ValueError('charset range is missing its step')
                step = int(tokens[index])
                index += 1
        if start < 1 or end < start or step < 1:
            raise ValueError(f'invalid charset range {start}-{end}/{step}')
        coordinates.extend(range(start, end + 1, step))
    if not coordinates:
        raise ValueError('charset contains no coordinates')
    if len(set(coordinates)) != len(coordinates):
        raise ValueError('charset contains duplicate coordinates')
    return np.asarray(coordinates, dtype=int)


def parse_iqtree_partitions(path, *, sequence_length=None):
    """Return ordered charset names and global zero-based coordinates."""
    text = _read_text(path, 'IQ-TREE partition/model file')
    partitions = []
    for statement in _nexus_statements(text):
        match = re.match(
            r'\s*charset\s+(.+?)\s*=\s*(.+)\Z',
            statement,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            source_name = _unquote_nexus_name(match.group(1))
            name = _normalize_partition_name(source_name)
            if any(existing_name == name for existing_name, _ in partitions):
                raise ValueError(f'duplicate charset name after IQ-TREE normalization: {source_name!r}')
            partitions.append((name, _parse_charset_coordinates(match.group(2))))
    if not partitions:
        raise ValueError(f'{path}: no charset statements found')

    all_coordinates = np.concatenate([coordinates for _, coordinates in partitions])
    unique_coordinates, counts = np.unique(all_coordinates, return_counts=True)
    duplicates = unique_coordinates[counts > 1]
    if duplicates.size:
        raise ValueError(f'partition charsets overlap at alignment site {int(duplicates[0])}')
    inferred_length = int(all_coordinates.max())
    length = inferred_length if sequence_length is None else int(sequence_length)
    if length < 1:
        raise ValueError('sequence_length must be positive')
    expected = np.arange(1, length + 1)
    observed = np.sort(all_coordinates)
    if not np.array_equal(observed, expected):
        missing = np.setdiff1d(expected, observed)
        extra = np.setdiff1d(observed, expected)
        detail = []
        if missing.size:
            detail.append(f'missing site {int(missing[0])}')
        if extra.size:
            detail.append(f'out-of-range site {int(extra[0])}')
        raise ValueError('partition charsets are not a bijection over the alignment: ' + ', '.join(detail))
    return tuple(name for name, _ in partitions), tuple(coordinates - 1 for _, coordinates in partitions)


def _parse_partition_report(path, *, partition_names, partition_coordinates, sequence_length):
    lines = _read_text(path, 'IQ-TREE report').splitlines()
    mode_index = None
    mode = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('Edge-linked-proportional partition model'):
            mode_index = index
            mode = 'edge-linked-proportional'
            break
        if stripped.startswith('Edge-linked-equal partition model'):
            mode_index = index
            mode = 'edge-linked-equal'
            break
        if stripped.startswith(('Edge-unlinked partition model', 'Topology-unlinked partition model')):
            raise ValueError('edge-unlinked partition models cannot be represented by one TreeTime branch-time process')
    if mode_index is None:
        raise ValueError(f'{path}: supported partition model not found in SUBSTITUTION PROCESS section')

    header_index = None
    header = None
    for index in range(mode_index + 1, len(lines)):
        fields = lines[index].split()
        if fields[:2] == ['ID', 'Model']:
            header_index = index
            header = fields
            break
        if lines[index].strip() == 'SUBSTITUTION PROCESS':
            break
    if header_index is None or 'Speed' not in header:
        raise ValueError(f'{path}: partition Speed table is missing')

    id_column = header.index('ID')
    speed_column = header.index('Speed')
    speeds = {}
    for line_number in range(header_index + 1, len(lines)):
        fields = lines[line_number].split()
        if not fields:
            if speeds:
                break
            continue
        if len(fields) <= speed_column or not fields[id_column].isdigit():
            if speeds:
                break
            continue
        partition_id = int(fields[id_column])
        speed = _parse_nonnegative_number(fields[speed_column], field='partition Speed', line_number=line_number + 1)
        if speed == 0:
            raise ValueError(f'line {line_number + 1}: partition Speed must be positive')
        if partition_id in speeds:
            raise ValueError(f'{path}: duplicate partition ID {partition_id} in Speed table')
        speeds[partition_id] = speed

    expected_ids = set(range(1, len(partition_names) + 1))
    if set(speeds) != expected_ids:
        raise ValueError(f'{path}: partition Speed table IDs do not match partition definitions')

    input_data = next(
        (line.strip() for line in lines if line.strip().startswith('Input data:')),
        None,
    )
    if input_data is None:
        raise ValueError(f'{path}: SEQUENCE ALIGNMENT summary is missing')
    summary = re.search(
        r'with\s+([0-9]+)\s+partitions\s+and\s+([0-9]+)\s+total sites',
        input_data,
    )
    if summary is None:
        raise ValueError(f'{path}: could not parse partition count and total sites')
    report_partition_count, report_length = map(int, summary.groups())
    if report_partition_count != len(partition_names) or report_length != sequence_length:
        raise ValueError(f'{path}: alignment summary does not match partition definitions')

    size_header_index = next(
        (index for index, line in enumerate(lines) if line.split()[:4] == ['ID', 'Type', 'Seq', 'Site']),
        None,
    )
    if size_header_index is None:
        raise ValueError(f'{path}: partition size table is missing')
    size_header = lines[size_header_index].split()
    if 'Name' not in size_header:
        raise ValueError(f'{path}: partition size table has no Name column')
    name_column = size_header.index('Name')
    report_partitions = {}
    for line in lines[size_header_index + 1 :]:
        fields = line.split()
        if not fields:
            break
        if len(fields) <= name_column or not fields[0].isdigit() or not fields[3].isdigit():
            break
        partition_id = int(fields[0])
        report_name = _normalize_partition_name(' '.join(fields[name_column:]))
        report_partitions[partition_id] = (int(fields[3]), report_name)
    expected_partitions = {
        partition_id: (
            len(partition_coordinates[partition_id - 1]),
            partition_names[partition_id - 1],
        )
        for partition_id in expected_ids
    }
    if report_partitions != expected_partitions:
        raise ValueError(f'{path}: reported partition sizes or names do not match partition definitions')

    return mode, np.asarray([speeds[partition_id] for partition_id in sorted(speeds)], dtype=float)


def load_iqtree_site_rates(
    rate_file,
    *,
    sequence_length=None,
    partition_file=None,
    report_file=None,
):
    """Load posterior-mean IQ-TREE rates as a globally normalized model."""
    header, records = _read_iqtree_table(rate_file, required_columns={'Site', 'Rate'})
    partitioned = 'Part' in header
    if partitioned != (partition_file is not None):
        if partitioned:
            raise ValueError('partitioned .rate input requires an IQ-TREE partition/model file')
        raise ValueError('partition_file was provided for an unpartitioned .rate file')

    if partitioned:
        if report_file is None:
            raise ValueError('partitioned .rate input requires an IQ-TREE report with partition speeds')
        partition_names, partition_coordinates = parse_iqtree_partitions(
            partition_file, sequence_length=sequence_length
        )
        length = sum(len(coordinates) for coordinates in partition_coordinates)
        mode, partition_speeds = _parse_partition_report(
            report_file,
            partition_names=partition_names,
            partition_coordinates=partition_coordinates,
            sequence_length=length,
        )
    else:
        sites = [
            _parse_positive_integer(record['Site'], field='Site', line_number=line_number)
            for line_number, record in records
        ]
        inferred_length = max(sites)
        length = inferred_length if sequence_length is None else int(sequence_length)
        if length < 1:
            raise ValueError('sequence_length must be positive')
        partition_names = ('alignment',)
        partition_coordinates = (np.arange(length, dtype=int),)
        partition_speeds = np.ones(1)
        mode = 'unpartitioned'

    rates = np.full(length, np.nan)
    partition_index = np.full(length, -1, dtype=int)
    partition_site = np.full(length, -1, dtype=int)
    seen_source_coordinates = set()
    for line_number, record in records:
        local_site = _parse_positive_integer(record['Site'], field='Site', line_number=line_number)
        rate = _parse_nonnegative_number(record['Rate'], field='Rate', line_number=line_number)
        if partitioned:
            partition_id = _parse_positive_integer(record['Part'], field='Part', line_number=line_number)
        else:
            partition_id = 1
        if partition_id > len(partition_names):
            raise ValueError(f'line {line_number}: unknown partition ID {partition_id}')
        coordinates = partition_coordinates[partition_id - 1]
        if local_site > len(coordinates):
            raise ValueError(f'line {line_number}: site {local_site} is out of range for partition {partition_id}')
        source_coordinate = (partition_id, local_site)
        if source_coordinate in seen_source_coordinates:
            raise ValueError(f'line {line_number}: duplicate site {local_site} in partition {partition_id}')
        seen_source_coordinates.add(source_coordinate)
        global_coordinate = int(coordinates[local_site - 1])
        if np.isfinite(rates[global_coordinate]):
            raise ValueError(f'line {line_number}: duplicate global alignment coordinate')
        rates[global_coordinate] = rate * partition_speeds[partition_id - 1]
        partition_index[global_coordinate] = partition_id - 1
        partition_site[global_coordinate] = local_site

    if len(records) != length or np.any(~np.isfinite(rates)):
        missing = np.flatnonzero(~np.isfinite(rates))
        detail = f'; first missing alignment site is {int(missing[0]) + 1}' if missing.size else ''
        raise ValueError(f'{rate_file}: expected exactly {length} mapped rows, got {len(records)}{detail}')
    normalization_constant = float(rates.mean())
    if not np.isfinite(normalization_constant) or normalization_constant <= 0:
        raise ValueError('site rates must have a positive global mean')
    rates /= normalization_constant

    metadata = {
        'source': 'IQ-TREE .rate',
        'rate_file': str(Path(rate_file)),
        'partition_file': str(Path(partition_file)) if partition_file is not None else None,
        'report_file': str(Path(report_file)) if report_file is not None else None,
        'partition_model': mode,
        'normalization_constant': normalization_constant,
    }
    return SiteRateModel(
        mean_rates=rates,
        partition_index=partition_index,
        partition_site=partition_site,
        partition_names=partition_names,
        partition_speeds=partition_speeds,
        evaluation_mode='mean',
        metadata=metadata,
    )


def _split_top_level(text, separator):
    """Split one IQ-TREE model expression outside quotes and nested groups."""
    parts = []
    current = []
    quote = None
    depths = {'{': 0, '(': 0, '[': 0}
    closing = {'}': '{', ')': '(', ']': '['}
    for character in text:
        if quote:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
            current.append(character)
        elif character in depths:
            depths[character] += 1
            current.append(character)
        elif character in closing:
            opener = closing[character]
            depths[opener] -= 1
            if depths[opener] < 0:
                raise ValueError(f'unbalanced {character!r} in IQ-TREE model expression')
            current.append(character)
        elif character == separator and not any(depths.values()):
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(character)
    if quote or any(depths.values()):
        raise ValueError('unterminated quote or group in IQ-TREE model expression')
    parts.append(''.join(current).strip())
    return parts


def _top_level_partition(entry):
    pieces = _split_top_level(entry, ':')
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f'invalid IQ-TREE charpartition entry: {entry!r}')
    return pieces


_DEFAULT_GAMMA_CATEGORIES = 4


def _parse_freerate_expression(expression):
    """Parse a single top-level ``+Rk{...}`` or ``+Gk{alpha}`` rate component.

    ``+R`` reads explicit alternating ``weight,rate`` pairs. ``+G`` (discrete
    Gamma) writes only the shape in ``best_model.nex`` (no per-category rate
    table survives partition output), so the category rates are recomputed from
    ``alpha`` with the mean method and equal ``1/K`` priors.
    """
    if '+I{' in expression:
        raise ValueError(
            'IQ-TREE +I+R site posteriors do not expose an independently verified '
            'invariant-category convention; use +R for posterior-elbo mode'
        )

    matches = []
    depth = 0
    index = 0
    while index < len(expression):
        character = expression[index]
        if character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth < 0:
                raise ValueError('unbalanced braces in IQ-TREE model expression')
        elif depth == 0 and (expression.startswith('+R', index) or expression.startswith('+G', index)):
            kind = expression[index + 1]
            cursor = index + 2
            while cursor < len(expression) and expression[cursor].isdigit():
                cursor += 1
            bare = cursor == index + 2
            if cursor >= len(expression) or expression[cursor] != '{' or (bare and kind == 'R'):
                raise ValueError(f'malformed +{kind} rate component in {expression!r}')
            category_count = _DEFAULT_GAMMA_CATEGORIES if bare else int(expression[index + 2 : cursor])
            end = cursor + 1
            nested = 1
            while end < len(expression) and nested:
                if expression[end] == '{':
                    nested += 1
                elif expression[end] == '}':
                    nested -= 1
                end += 1
            if nested:
                raise ValueError(f'unterminated +{kind} parameter group')
            matches.append((kind, category_count, expression[cursor + 1 : end - 1]))
            index = end
            continue
        index += 1
    if depth != 0:
        raise ValueError('unbalanced braces in IQ-TREE model expression')
    if len(matches) != 1:
        raise ValueError(f'expected exactly one top-level IQ-TREE +R or +G component, found {len(matches)}')

    kind, category_count, parameter_text = matches[0]
    if kind == 'G':
        alpha_tokens = [token.strip() for token in _split_top_level(parameter_text, ',')]
        if len(alpha_tokens) != 1:
            raise ValueError(f'+G requires a single shape parameter, got {parameter_text!r}')
        alpha = _parse_nonnegative_number(alpha_tokens[0], field='Gamma shape alpha', line_number=0)
        category_rates = _discrete_gamma_mean_rates(alpha, category_count)
        prior_weights, deviation = _normalize_probability_row(
            np.full(category_count, 1.0 / category_count), description='discrete-Gamma category prior'
        )
        return category_rates, prior_weights, deviation

    values = [
        _parse_nonnegative_number(value.strip(), field='FreeRate parameter', line_number=0)
        for value in _split_top_level(parameter_text, ',')
    ]
    if len(values) != 2 * category_count:
        raise ValueError(
            f'+R{category_count} requires {2 * category_count} alternating weight/rate values, got {len(values)}'
        )
    prior_weights = np.asarray(values[0::2], dtype=float)
    category_rates = np.asarray(values[1::2], dtype=float)
    prior_weights, deviation = _normalize_probability_row(prior_weights, description='FreeRate category prior')
    return category_rates, prior_weights, deviation


def _normalize_probability_row(values, *, description, tolerance=1e-5):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f'{description} must be finite and nonnegative')
    total = float(values.sum())
    deviation = abs(total - 1.0)
    if total <= 0 or deviation > tolerance:
        raise ValueError(f'{description} sums to {total:.12g}, expected one')
    return values / total, deviation


def _discrete_gamma_mean_rates(alpha, category_count):
    """Discretize a mean-one Gamma(shape=rate=alpha) into ``K`` mean-method rates.

    IQ-TREE reports "Relative rates are computed as MEAN of the portion of the
    Gamma distribution falling in the category" over ``K`` equal-probability
    categories (Yang 1994). The category boundaries satisfy
    ``alpha * q_k = gammaincinv(alpha, k / K)`` and the mean rate of category
    ``k`` is ``K * (F(q_k; alpha+1) - F(q_{k-1}; alpha+1))`` where
    ``F(q; alpha+1) = gammainc(alpha+1, alpha*q)``. The returned rates have mean
    exactly one by construction (the ``F(.; alpha+1)`` telescopes to one).
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f'discrete-Gamma shape alpha must be finite and positive, got {alpha!r}')
    if category_count < 1:
        raise ValueError('discrete-Gamma category count must be positive')
    edges = [gammaincinv(alpha, k / category_count) for k in range(1, category_count)]
    cumulative = [0.0] + [gammainc(alpha + 1.0, edge) for edge in edges] + [1.0]
    rates = np.asarray(
        [category_count * (cumulative[k + 1] - cumulative[k]) for k in range(category_count)],
        dtype=float,
    )
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError(f'discrete-Gamma rates are not finite and nonnegative for alpha {alpha!r}')
    return rates


def _parse_partition_freerate_models(path, partition_names):
    text = _read_text(path, 'IQ-TREE partition/model file')
    charpartition = None
    for statement in _nexus_statements(text):
        match = re.match(
            r'\s*charpartition\s+\S+\s*=\s*(.+)\Z',
            statement,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            if charpartition is not None:
                raise ValueError(f'{path}: multiple charpartition statements are ambiguous')
            charpartition = match.group(1)
    if charpartition is None:
        raise ValueError(f'{path}: no charpartition model statement found')

    models = {}
    maximum_prior_deviation = 0.0
    for entry in _split_top_level(charpartition, ','):
        model_expression, target = _top_level_partition(entry)
        if _is_mixture_model_expression(model_expression):
            raise ValueError(
                f'{path}: substitution-mixture models have ambiguous .sitelh columns; '
                'posterior-elbo requires a single-matrix IQ-TREE model'
            )
        target_name = target
        for index, character in enumerate(target):
            if character == '{':
                target_name = target[:index]
                break
        name = _normalize_partition_name(_unquote_nexus_name(target_name))
        if name in models:
            raise ValueError(f'{path}: duplicate model assignment for partition {name!r}')
        rates, priors, deviation = _parse_freerate_expression(model_expression)
        models[name] = (rates, priors)
        maximum_prior_deviation = max(maximum_prior_deviation, deviation)

    if set(models) != set(partition_names):
        raise ValueError(f'{path}: charpartition model names do not match charset names')
    return tuple(models[name] for name in partition_names), maximum_prior_deviation


def _parse_report_freerate_model(path):
    lines = _read_text(path, 'IQ-TREE report').splitlines()
    if any('Mixture model of substitution:' in line for line in lines):
        raise ValueError(
            f'{path}: substitution-mixture models have ambiguous .sitelh columns; '
            'posterior-elbo requires a single-matrix IQ-TREE model'
        )
    model_lines = [
        line.split(':', maxsplit=1)[1].strip()
        for line in lines
        if line.strip().startswith('Model of substitution:') and ':' in line
    ]
    if len(model_lines) != 1:
        raise ValueError(f'{path}: expected one reported substitution model')
    model_name = model_lines[0]
    if '+I' in model_name:
        raise ValueError(
            'IQ-TREE +I+R category output is not supported until its .sitelh '
            'category convention is independently verified'
        )
    # +R and +G share the report's Category/Relative_rate/Proportion table
    # (equal 1/K proportions and MEAN category rates for +G); accept exactly one.
    category_counts = []
    index = 0
    while index < len(model_name):
        if model_name.startswith('+R', index) or model_name.startswith('+G', index):
            kind = model_name[index + 1]
            cursor = index + 2
            while cursor < len(model_name) and model_name[cursor].isdigit():
                cursor += 1
            bare = cursor == index + 2
            if bare and kind == 'R':
                raise ValueError(f'{path}: malformed reported FreeRate model')
            category_counts.append(_DEFAULT_GAMMA_CATEGORIES if bare else int(model_name[index + 2 : cursor]))
            index = cursor
        else:
            index += 1
    if len(category_counts) != 1:
        raise ValueError(f'{path}: posterior-elbo input requires one IQ-TREE +R or +G model')

    headers = [index for index, line in enumerate(lines) if line.split() == ['Category', 'Relative_rate', 'Proportion']]
    if len(headers) != 1:
        raise ValueError(f'{path}: expected one Category/Relative_rate/Proportion table, found {len(headers)}')
    rows = []
    for line_number in range(headers[0] + 1, len(lines)):
        fields = lines[line_number].split()
        if not fields:
            if rows:
                break
            continue
        if len(fields) != 3 or not fields[0].isdigit():
            if rows:
                break
            continue
        category_id = int(fields[0])
        rate = _parse_nonnegative_number(fields[1], field='FreeRate category rate', line_number=line_number + 1)
        weight = _parse_nonnegative_number(fields[2], field='FreeRate category prior', line_number=line_number + 1)
        rows.append((category_id, rate, weight))
    if not rows:
        raise ValueError(f'{path}: FreeRate category table has no rows')
    if rows[0][0] == 0:
        raise ValueError(
            'IQ-TREE +I+R category output is not supported until its .sitelh '
            'category convention is independently verified'
        )
    expected_ids = list(range(1, len(rows) + 1))
    if [row[0] for row in rows] != expected_ids:
        raise ValueError(f'{path}: FreeRate category IDs must be consecutive from one')
    if len(rows) != category_counts[0]:
        raise ValueError(f'{path}: reported +R/+G category count does not match category table')
    rates = np.asarray([row[1] for row in rows], dtype=float)
    priors, deviation = _normalize_probability_row([row[2] for row in rows], description='FreeRate category prior')
    return rates, priors, deviation


def _validate_unpartitioned_report_length(path, sequence_length):
    lines = _read_text(path, 'IQ-TREE report').splitlines()
    input_data = next(
        (line.strip() for line in lines if line.strip().startswith('Input data:')),
        None,
    )
    if input_data is None:
        raise ValueError(f'{path}: SEQUENCE ALIGNMENT summary is missing')
    summary = re.search(
        r'Input data:\s+[0-9]+\s+sequences\s+with\s+([0-9]+)\s+\S+\s+sites',
        input_data,
    )
    if summary is None:
        raise ValueError(f'{path}: could not parse unpartitioned alignment length')
    if int(summary.group(1)) != sequence_length:
        raise ValueError(f'{path}: reported alignment length does not match .sitelh')


_SIGNED_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')


def _parse_finite_number(value, *, field, line_number):
    if not _SIGNED_NUMBER.fullmatch(value):
        raise ValueError(f'line {line_number}: {field} is not a finite decimal number: {value!r}')
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f'line {line_number}: {field} must be finite')
    return number


def _read_sitelh_table(path):
    """Parse IQ-TREE ``-wslr`` ``.sitelh`` output into per-site category posteriors.

    Rows carry ``LnL`` (total site log-likelihood) followed by consecutive
    ``LnLW_k = log(category-k site-likelihood * category-k weight)``. The per-site
    category responsibility is reconstructed as ``q_k = exp(LnLW_k - LnL)``; for
    non-invariant models (+R) these reproduce IQ-TREE's ``-wspr`` ``.siteprob``
    columns to writer precision (~1e-4). Partitioned files are detected by a
    leading ``Part`` column and may emit ragged rows: partitions with fewer
    categories print fewer ``LnLW`` values (the file's own R hint uses fill=TRUE).
    """
    text = _read_text(path, 'IQ-TREE .sitelh file')
    header = None
    records = []
    category_count = None
    partitioned = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if header is None:
            header = fields
            partitioned = header[:1] == ['Part']
            site_index = 1 if partitioned else 0
            if header[site_index : site_index + 2] != ['Site', 'LnL']:
                expected = 'Part Site LnL LnLW_1..LnLW_K' if partitioned else 'Site LnL LnLW_1..LnLW_K'
                raise ValueError(f'{path}: expected header {expected!r}')
            category_names = header[site_index + 2 :]
            expected_names = [f'LnLW_{index}' for index in range(1, len(category_names) + 1)]
            if not category_names or category_names != expected_names:
                raise ValueError(f'{path}: category columns must be consecutive LnLW_1..LnLW_K')
            category_count = len(category_names)
            continue
        minimum_fields = 4 if partitioned else 3
        if len(fields) < minimum_fields or len(fields) > len(header):
            raise ValueError(f'{path}: line {line_number} has an invalid number of fields')
        if partitioned:
            partition_id = _parse_positive_integer(fields[0], field='Part', line_number=line_number)
            site_value = fields[1]
            log_likelihood_value = fields[2]
            category_values = fields[3:]
        else:
            partition_id = 1
            site_value = fields[0]
            log_likelihood_value = fields[1]
            category_values = fields[2:]
        local_site = _parse_positive_integer(site_value, field='Site', line_number=line_number)
        log_likelihood = _parse_finite_number(log_likelihood_value, field='site LnL', line_number=line_number)
        category_log_weights = np.asarray(
            [_parse_finite_number(value, field='category LnLW', line_number=line_number) for value in category_values],
            dtype=float,
        )
        probabilities = np.exp(category_log_weights - log_likelihood)
        records.append((line_number, partition_id, local_site, probabilities))
    if header is None or not records:
        raise ValueError(f'{path}: no site-likelihood rows found')
    return partitioned, category_count, records


def _map_raw_rate_file(rate_file, *, partition_names, partition_coordinates, partitioned):
    header, records = _read_iqtree_table(rate_file, required_columns={'Site', 'Rate'})
    if ('Part' in header) != partitioned:
        raise ValueError(f'{rate_file}: partition structure does not match .sitelh input')
    length = sum(len(coordinates) for coordinates in partition_coordinates)
    rates = np.full(length, np.nan)
    for line_number, record in records:
        partition_id = (
            _parse_positive_integer(record['Part'], field='Part', line_number=line_number) if partitioned else 1
        )
        local_site = _parse_positive_integer(record['Site'], field='Site', line_number=line_number)
        if partition_id > len(partition_names):
            raise ValueError(f'{rate_file}: unknown partition ID {partition_id}')
        coordinates = partition_coordinates[partition_id - 1]
        if local_site > len(coordinates):
            raise ValueError(f'{rate_file}: partition-local site is out of range')
        global_coordinate = int(coordinates[local_site - 1])
        if np.isfinite(rates[global_coordinate]):
            raise ValueError(f'{rate_file}: duplicate mapped rate row')
        rates[global_coordinate] = _parse_nonnegative_number(record['Rate'], field='Rate', line_number=line_number)
    if len(records) != length or np.any(~np.isfinite(rates)):
        raise ValueError(f'{rate_file}: rate rows do not map bijectively to the alignment')
    return rates


def load_iqtree_site_rate_posteriors(
    sitelh_file,
    *,
    report_file,
    sequence_length=None,
    partition_file=None,
    rate_file=None,
):
    """Load IQ-TREE ``-wslr`` ``.sitelh`` output for frozen-responsibility ELBO dating.

    The ``.sitelh`` file carries per-site, per-category log-likelihoods; the
    category responsibilities used here are reconstructed as
    ``q_k = exp(LnLW_k - LnL)``. For the currently supported +R models this is a
    strict superset of the older ``-wspr`` ``.siteprob`` posteriors and matches
    them to writer precision (~1e-4).
    """
    partitioned, maximum_columns, records = _read_sitelh_table(sitelh_file)
    if partitioned != (partition_file is not None):
        if partitioned:
            raise ValueError('partitioned .sitelh input requires an IQ-TREE best-model file')
        raise ValueError('partition_file was provided for an unpartitioned .sitelh file')

    if partitioned:
        partition_names, partition_coordinates = parse_iqtree_partitions(
            partition_file, sequence_length=sequence_length
        )
        length = sum(len(coordinates) for coordinates in partition_coordinates)
        partition_mode, partition_speeds = _parse_partition_report(
            report_file,
            partition_names=partition_names,
            partition_coordinates=partition_coordinates,
            sequence_length=length,
        )
        partition_models, maximum_prior_deviation = _parse_partition_freerate_models(partition_file, partition_names)
    else:
        inferred_length = max(record[2] for record in records)
        length = inferred_length if sequence_length is None else int(sequence_length)
        if length < 1:
            raise ValueError('sequence_length must be positive')
        partition_names = ('alignment',)
        partition_coordinates = (np.arange(length, dtype=int),)
        partition_speeds = np.ones(1)
        partition_mode = 'unpartitioned'
        _validate_unpartitioned_report_length(report_file, length)
        rates, priors, maximum_prior_deviation = _parse_report_freerate_model(report_file)
        partition_models = ((rates, priors),)

    category_counts = tuple(len(model[0]) for model in partition_models)
    if max(category_counts) != maximum_columns:
        raise ValueError(f'{sitelh_file}: posterior header category count does not match rate models')
    category_width = max(category_counts)
    posterior_weights = np.zeros((length, category_width), dtype=float)
    category_rates = np.zeros((length, category_width), dtype=float)
    category_priors = np.zeros((length, category_width), dtype=float)
    category_mask = np.zeros((length, category_width), dtype=bool)
    partition_index = np.full(length, -1, dtype=int)
    partition_site = np.full(length, -1, dtype=int)
    maximum_posterior_deviation = 0.0
    renormalized_posterior_rows = 0

    for line_number, partition_id, local_site, probabilities in records:
        if partition_id > len(partition_names):
            raise ValueError(f'line {line_number}: unknown partition ID {partition_id}')
        coordinates = partition_coordinates[partition_id - 1]
        if local_site > len(coordinates):
            raise ValueError(f'line {line_number}: site {local_site} is out of range for partition {partition_id}')
        expected_count = category_counts[partition_id - 1]
        if len(probabilities) != expected_count:
            raise ValueError(
                f'line {line_number}: partition {partition_id} requires '
                f'{expected_count} posterior columns, got {len(probabilities)}'
            )
        # -wslr prints ~4-decimal log-likelihoods, so a reconstructed
        # q_k = exp(LnLW_k - LnL) row can deviate from sum-1 by ~1e-4; loosen the
        # per-row normalization tolerance accordingly (the strict 1e-5 tolerance is
        # retained for the full-precision category-prior rows from the model files).
        normalized_probabilities, deviation = _normalize_probability_row(
            probabilities, description=f'line {line_number} posterior row', tolerance=1e-3
        )
        maximum_posterior_deviation = max(maximum_posterior_deviation, deviation)
        if deviation > 0:
            renormalized_posterior_rows += 1
        global_coordinate = int(coordinates[local_site - 1])
        if partition_index[global_coordinate] >= 0:
            raise ValueError(f'line {line_number}: duplicate mapped posterior row')
        raw_rates, priors = partition_models[partition_id - 1]
        posterior_weights[global_coordinate, :expected_count] = normalized_probabilities
        category_rates[global_coordinate, :expected_count] = raw_rates * partition_speeds[partition_id - 1]
        category_priors[global_coordinate, :expected_count] = priors
        category_mask[global_coordinate, :expected_count] = True
        partition_index[global_coordinate] = partition_id - 1
        partition_site[global_coordinate] = local_site

    if len(records) != length or np.any(partition_index < 0):
        raise ValueError(f'{sitelh_file}: posterior rows do not map bijectively to the alignment')

    scaled_mean_rates = np.sum(posterior_weights * category_rates, axis=1)
    normalization_constant = float(scaled_mean_rates.mean())
    if not np.isfinite(normalization_constant) or normalization_constant <= 0:
        raise ValueError('posterior-weighted category rates must have a positive global mean')
    category_rates /= normalization_constant
    mean_rates = np.sum(posterior_weights * category_rates, axis=1)

    if rate_file is not None:
        raw_rate_means = _map_raw_rate_file(
            rate_file,
            partition_names=partition_names,
            partition_coordinates=partition_coordinates,
            partitioned=partitioned,
        )
        if np.any(raw_rate_means == 100.0):
            raise ValueError(
                f'{rate_file}: IQ-TREE censors posterior mean rates at 100; '
                'an exact .rate/.sitelh cross-check is impossible. Omit the '
                'optional .rate cross-check or supply uncensored rates.'
            )
        posterior_raw_means = mean_rates * normalization_constant / partition_speeds[partition_index]
        # -wslr reconstructs the responsibilities from ~4-decimal log-likelihoods,
        # so the posterior mean rate can differ from IQ-TREE's own .rate row by
        # ~1e-4; loosen atol from the .siteprob-era 2e-5 accordingly.
        if not np.allclose(
            raw_rate_means,
            posterior_raw_means,
            atol=2e-4,
            rtol=2e-4,
        ):
            difference = np.abs(raw_rate_means - posterior_raw_means)
            coordinate = int(np.argmax(difference))
            raise ValueError(f'{rate_file}: posterior mean cross-check failed at alignment site {coordinate + 1}')

    metadata = {
        'source': 'IQ-TREE .sitelh',
        'sitelh_file': str(Path(sitelh_file)),
        'rate_file': str(Path(rate_file)) if rate_file is not None else None,
        'partition_file': str(Path(partition_file)) if partition_file is not None else None,
        'report_file': str(Path(report_file)),
        'partition_model': partition_mode,
        'sitelh_mode': 'rate-category (single-matrix IQ-TREE model)',
        'category_counts': category_counts,
        'normalization_constant': normalization_constant,
        'renormalized_posterior_rows': renormalized_posterior_rows,
        'maximum_posterior_deviation': maximum_posterior_deviation,
        'maximum_prior_deviation': maximum_prior_deviation,
    }
    return SiteRateModel(
        mean_rates=mean_rates,
        posterior_weights=posterior_weights,
        category_rates=category_rates,
        category_prior_weights=category_priors,
        category_mask=category_mask,
        partition_index=partition_index,
        partition_site=partition_site,
        partition_names=partition_names,
        partition_speeds=partition_speeds,
        evaluation_mode='posterior-elbo',
        metadata=metadata,
    )
