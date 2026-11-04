"""Admit a freshly captured native cache without reusing historical calibration.

Receipts describe checksummed data/source inputs. Actual native teacher parity
is always rerun by the lifecycle before its first update; receipt text cannot
assert or bypass that gate.
"""
import json
import math
from pathlib import Path

from .shared import file_sha

SCHEMA = 'supra-native-pg094-cache-v1'
_KEYS = {'response_scale', 'cache_receipt', 'cache_receipt_sha256'}


def selected(options):
    return bool(_KEYS & set(options))


def validate_options(options):
    if not selected(options):
        return
    if not _KEYS <= set(options):
        raise ValueError('fresh cache requires response_scale, cache_receipt and cache_receipt_sha256 together')
    scale = options['response_scale']
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('fresh cache response_scale must be finite and positive')
    path = options['cache_receipt']
    if not isinstance(path, str) or not Path(path).is_file():
        raise ValueError('fresh cache_receipt must be an existing local JSON file')
    expected = options['cache_receipt_sha256']
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in '0123456789abcdef' for char in expected)
            or file_sha(path) != expected):
        raise ValueError('fresh cache receipt SHA256 differs')


def current_sources():
    """Exact executed product modules plus the separately verified core receipt."""
    from .native_io import LOCK
    names = ('runtime.py', 'native_model.py', 'native_io.py', 'cache.py', 'campaign.py',
             'lifecycle.py', 'fresh_cache.py', 'learning_rates.py', 'shared.py')
    from importlib.metadata import distribution
    import inspect
    from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter
    core = distribution('particle-sliders-core')
    relative = 'particle_sliders/pg09_backends.py'
    actual = Path(inspect.getfile(SupraRuntimeResponseAdapter)).resolve()
    if actual != Path(core.locate_file(relative)).resolve():
        raise RuntimeError('executed shared Supra response adapter differs from distribution ownership')
    core_receipt = json.loads(LOCK.read_text())
    return dict(product_sha256={name: file_sha(Path(__file__).with_name(name)) for name in names},
                core_receipt_sha256=file_sha(LOCK),
                core_origin=dict(distribution='particle-sliders-core', version=core.version,
                    commit=core_receipt['source_commit'], module=relative,
                    source_sha256=file_sha(actual)))


def receipt_fields(options, rows):
    """Describe real staged data; this is not a native parity attestation.

    Capture tools may write this object as their receipt. Lifecycle admission
    recomputes the fields and every distinct teacher response before updates.
    """
    from .cache import canonical_digest
    from .native_io import read_context_receipt
    context_receipt = read_context_receipt(options['contexts'])
    return dict(schema=SCHEMA,
        cache_sha256=file_sha(options['cache']),
        contexts_sha256=file_sha(options['contexts']),
        contexts_receipt_sha256=file_sha(Path(options['contexts']).with_suffix('.json')),
        native_config_sha256=canonical_digest(options['native_config']),
        frozen_base_sha256=options['frozen_base_sha256'],
        runtime_identity=context_receipt['runtime_identity'],
        runtime_source_sha256=context_receipt['runtime_source_sha256'],
        source_identity=current_sources(), response_scale=rows.response_scale(),
        edited_states=len(rows.records), preservation_states=len(rows.holds),
        parity_policy=dict(coverage='all unique cache records', relative_rms_limit=.001,
                           absolute_rms_limit=1e-6))


def admit(options, rows):
    validate_options(options)
    try:
        receipt = json.loads(Path(options['cache_receipt']).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('readable fresh cache receipt JSON required') from error
    expected = receipt_fields(options, rows)
    if receipt != expected:
        raise ValueError('fresh cache receipt data/source/calibration binding differs')
    if not math.isclose(rows.response_scale(), options['response_scale'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('fresh cache response_scale differs from actual calibrated data')
    if (expected['cache_sha256'] != options['cache_sha256']
            or expected['contexts_sha256'] != options['contexts_sha256']):
        raise ValueError('fresh cache option checksums differ from actual data')
    return receipt


def parity_indices(rows):
    """Each underlying edit/preservation row once, excluding sampling repeats."""
    indices = {}
    for index, address in enumerate(rows.routes):
        indices.setdefault(tuple(address), index)
    return tuple(indices.values())
