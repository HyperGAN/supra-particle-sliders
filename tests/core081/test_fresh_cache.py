"""Fresh native cache admission derives scale from data and reruns every teacher."""
import json

import pytest
import torch
from particle_sliders.lifecycle import SessionConfig

from supra import cache, campaign, fresh_cache, lifecycle
from supra.shared import file_sha

from .test_lifecycle_closure import configured as _configured, assets  # noqa: F401

configured = _configured


def new_config(original, tmp_path, *, patch_data=None):
    options = original.options
    if patch_data:
        data = torch.load(options['cache'], weights_only=True)
        patch_data(data)
        path = tmp_path / 'fresh-cache.pt'
        torch.save(data, path)
        options.update(cache=str(path), cache_sha256=file_sha(path))
    contexts = cache.load_supra_contexts(options['contexts'], options['contexts_sha256'], options['native_config'])
    rows = cache.SupraCampaignRows(options['cache'], cache_sha256=options['cache_sha256'],
        config=options['native_config'], contexts=contexts)
    receipt = fresh_cache.receipt_fields(options, rows)
    path = tmp_path / 'fresh-cache.json'
    path.write_text(json.dumps(receipt))
    return SessionConfig('supra', 'fresh-native-cache-lr-v1', **options,
        response_scale=rows.response_scale(), cache_receipt=str(path), cache_receipt_sha256=file_sha(path))


def test_fresh_receipt_binds_actual_scale_and_every_unique_row(configured, tmp_path, monkeypatch):
    runtime, original = configured
    config = new_config(original, tmp_path)
    real = campaign.teacher_parity
    seen = []
    def checked(run, indices, **kwargs):
        seen.extend(indices)
        return real(run, indices, **kwargs)
    monkeypatch.setattr(campaign, 'teacher_parity', checked)
    with lifecycle.BACKEND.open(config, runtime=runtime) as session:
        rows = session.game.rows
        assert len(seen) == len(rows.records) + len(rows.holds)
        assert len({tuple(rows.routes[index]) for index in seen}) == len(seen)
        assert session.game.response_scale == rows.response_scale()
        session.step()
        receipt = json.loads(open(config.options['cache_receipt']).read())
        assert receipt['source_identity']['core_origin']['module'] == 'particle_sliders/pg09_backends.py'
        assert receipt['source_identity']['core_origin']['commit'] == '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423'


@pytest.mark.parametrize('key', ['response_scale', 'source_identity', 'cache_sha256',
                                  'contexts_sha256', 'frozen_base_sha256', 'edited_states'])
def test_receipt_claims_cannot_replace_measured_inputs(configured, tmp_path, key):
    _, original = configured
    config = new_config(original, tmp_path)
    options = config.options
    path = options['cache_receipt']
    receipt = json.loads(open(path).read())
    if key == 'response_scale':
        receipt[key] *= 2
        options['response_scale'] = receipt[key]
    elif key == 'source_identity':
        receipt[key]['core_origin']['source_sha256'] = '0'*64
    elif key == 'edited_states':
        receipt[key] += 1
    else:
        receipt[key] = '0'*64
    with open(path, 'w') as stream:
        json.dump(receipt, stream)
    options['cache_receipt_sha256'] = file_sha(path)
    with pytest.raises(ValueError, match='fresh cache receipt'):
        lifecycle._assets(options)


def test_late_teacher_row_is_checked_even_with_self_consistent_receipt(configured, tmp_path):
    runtime, original = configured
    def corrupt(data):
        # The old representative-per-prompt gate would not inspect this second
        # timestep for the first prompt/conditioning role.
        data['records'][1]['target'].add_(.1)
    config = new_config(original, tmp_path, patch_data=corrupt)
    with pytest.raises(ValueError, match='native teacher/cache parity failed'):
        lifecycle.BACKEND.open(config, runtime=runtime)


def test_old_scale_gate_remains_without_fresh_receipt(configured, monkeypatch):
    _, original = configured
    monkeypatch.setattr(cache, 'SUPRA_SCALE', cache.SUPRA_SCALE * 2)
    with pytest.raises(ValueError, match='Reviewed native cache response scale differs'):
        lifecycle._assets(original.options)


@pytest.mark.parametrize('values', [{'response_scale':1.},
    {'response_scale':True,'cache_receipt':'x','cache_receipt_sha256':'0'*64},
    {'response_scale':float('inf'),'cache_receipt':'x','cache_receipt_sha256':'0'*64}])
def test_incomplete_and_nonpositive_fresh_options_refuse(values):
    with pytest.raises(ValueError):
        fresh_cache.validate_options(values)
