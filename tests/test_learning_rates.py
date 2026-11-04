"""LR admission and exact shared recipe wiring; no pretrained/GPU work."""
import pytest
from supra.learning_rates import recipe_overrides

@pytest.mark.parametrize('key', ['lr', 'd_lr_mult', 'prior_lr_mult'])
@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '0.001'])
def test_invalid_rates(key, value):
    with pytest.raises(ValueError):
        recipe_overrides(**{key: value})


def test_optional_rates_keep_preset_defaults():
    assert recipe_overrides() == {}
    assert recipe_overrides(lr=2e-5, d_lr_mult=3) == {'lr': 2e-5, 'd_lr_mult': 3.0}


def test_adapter_only_rejects_prior_rate():
    with pytest.raises(ValueError, match='adapter-only'):
        recipe_overrides(prior_lr_mult=2)


def test_closed_hndl_borrowed_runtime_bytes_are_preserved(tmp_path):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('lr_archive_builder', root / 'scripts/build_hndl_profile.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    manifest = json.loads((root/'cohorts/pre-lr-hndl/manifest.json').read_text())
    for relative in manifest['files']:
        (tmp_path/relative).parent.mkdir(parents=True, exist_ok=True)
    builder.restore_pre_lr_runtime(root,tmp_path)
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((tmp_path/relative).read_bytes()).hexdigest() == expected
    assert not list(tmp_path.rglob('learning_rates.py'))
