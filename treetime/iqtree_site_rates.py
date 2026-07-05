"""Load IQ-TREE site-rate output onto the original alignment coordinate axis."""

from pathlib import Path
import re

import numpy as np

from .site_rate_model import SiteRateModel


_INTEGER = re.compile(r'[1-9][0-9]*\Z')
_NUMBER = re.compile(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_COORDINATE_TOKEN = re.compile(r'\s*([0-9]+|[-,\\/])')


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
