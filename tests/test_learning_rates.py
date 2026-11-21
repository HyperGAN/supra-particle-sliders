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
