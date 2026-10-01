"""Editing-only sampling law, unchanged native game, and strict recovery.

The small public two-site128x4 fixture uses an explicit synthetic native clock
at1600; it does not claim to reproduce1600 training updates. Its application
Generator7 state is independently advanced through all1600 historical draws.
The production runner separately qualifies the real full-native1600 boundary.
"""
from copy import deepcopy

import pytest
import torch

from scripts.e22_supra_editing_only import (
    BASE_SAMPLING, EDITING_ONLY, SCHEDULE_KEYS, START_STEP,
    editing_config, make_from_saved, opt_in, training_update, validate_editing_config,
)
from supra.particle_game import update
from supra.particle_adapter import GATED_PARTICLE_V3
from supra.particle_pilot import checkpoint, restore, state_digest
from supra.particle_training import training_update as historical_update
from test_experimental_e22_bank_game_trust import native_loop


@pytest.fixture(autouse=True)
def shared_serial_native_fixture(monkeypatch):
    import test_experimental_e22_bank_game_trust as fixture
    original = fixture.get_recipe
    monkeypatch.setattr(fixture, "get_recipe", lambda name, **kw:
                        original(name, birth_death_backend="auto", reopen_guard="settled", **kw))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(threads)


def boundary():
    current = native_loop()
    current.fit_context = torch.linspace(-.3, .4, 26).reshape(13, 2)
    current.fit_targets = torch.zeros(13, 4, 32, 32)
    current.hold_context = torch.linspace(-.2, .5, 14).reshape(7, 2)
    current.config.update(batch_size=4, sampling=BASE_SAMPLING,
        preservation_game_weight=.1, architecture=GATED_PARTICLE_V3,
        particle_profile="pr223_shared_routed_v1", branch_lr=5e-5, probe_interval=100)
    # Resolve the shared backend through a genuine first native update. Its
    # public restore correctly refuses a pending auto backend at a later clock.
    historical_update(current)
    expected = torch.Generator().manual_seed(7)
    for step in range(1, START_STEP + 1):
        size = len(current.hold_context if step % 5 == 0 else current.fit_context)
        torch.randint(size, (4,), generator=expected)
    current.data_rng.set_state(expected.get_state())
    current.policy.completed_steps = START_STEP  # Explicit synthetic clock.
    return current


def direct_edit_update(current):
    """Reference calls the unchanged native game directly on the same fit draw."""
    pool = current.fit_context
    ids = torch.randint(len(pool), (4,), generator=current.data_rng)
    row = update(current, context=pool[ids], target=torch.zeros(4, 4, 32, 32),
                 batch_indices=ids, game_weight=1.)
    row.update(hold=False, game_weight=1.)
    return row


def test_config_versions_only_sampling_and_rejects_malformed_laws():
    base = boundary().config
    original = deepcopy(base)
    tagged = editing_config(base)
    assert base == original and tagged is not base
    assert tagged["training_schedule"] == EDITING_ONLY
    assert tagged["schedule_start_step"] == START_STEP
    assert tagged["pre_switch_edit_updates"] == 1280
    assert tagged["pre_switch_preservation_updates"] == 320
    assert validate_editing_config(tagged) == base
    for key in base:
        if key != "sampling":
            assert tagged[key] == base[key]
    for key, value in (("training_schedule", "editing_only_v2"),
                       ("schedule_start_step", 1599),
                       ("pre_switch_edit_updates", 1600),
                       ("pre_switch_preservation_updates", 0),
                       ("sampling", BASE_SAMPLING), ("preservation_game_weight", 0)):
        malformed = dict(tagged, **{key: value})
        with pytest.raises(ValueError):
            validate_editing_config(malformed)
    for key in SCHEDULE_KEYS:
        malformed = dict(tagged)
        malformed.pop(key)
        with pytest.raises(ValueError):
            validate_editing_config(malformed)
    with pytest.raises(ValueError):
        editing_config(tagged)


def test_opt_in_changes_only_config_at_declared_boundary():
    current = boundary()
    before = deepcopy(checkpoint(current))
    for wrong_step in (START_STEP - 1, START_STEP + 1):
        current.policy.completed_steps = wrong_step
        digest = state_digest(checkpoint(current))
        with pytest.raises(ValueError):
            opt_in(current)
        assert state_digest(checkpoint(current)) == digest
    current.policy.completed_steps = START_STEP
    opt_in(current)
    after = checkpoint(current)
    assert validate_editing_config(after["config"]) == before["config"]
    assert state_digest({key: value for key, value in after.items() if key != "config"}) == \
           state_digest({key: value for key, value in before.items() if key != "config"})
    with pytest.raises(ValueError):
        opt_in(current)
    current.policy.completed_steps = START_STEP - 1
    digest = state_digest(checkpoint(current))
    with pytest.raises(ValueError):
        training_update(current)
    assert state_digest(checkpoint(current)) == digest


def test_five_actual_updates_exact_native_game_with_fit_only_sampling():
    edited = boundary()
    opt_in(edited)
    initial = deepcopy(checkpoint(edited))
    expected_stream = torch.Generator().set_state(initial["data_rng"])
    expected_ids = [torch.randint(len(edited.fit_context), (4,), generator=expected_stream).tolist()
                    for _ in range(5)]
    actual = [training_update(edited) for _ in range(5)]
    final = state_digest(checkpoint(edited))
    assert [row["step"] for row in actual] == list(range(1601, 1606))
    assert [row["batch_indices"] for row in actual] == expected_ids
    assert all(row["hold"] is False and row["game_weight"] == 1. for row in actual)
    assert [row["penalty_calls"] for row in actual] == list(range(2, 7))
    assert all(row["dense_gradient_rows"] == 128 for row in actual)
    assert torch.equal(edited.data_rng.get_state(), expected_stream.get_state())
    reference = boundary()
    opt_in(reference)
    restore(reference, initial)
    assert [direct_edit_update(reference) for _ in range(5)] == actual
    assert state_digest(checkpoint(reference)) == final
    historical = boundary()
    # The counterfactual differs on1605 by the explicitly requested sampling
    # law, while both still draw the same two private Gaussian panels per step.
    old = [historical_update(historical) for _ in range(5)]
    assert old[-1]["hold"] and old[-1]["game_weight"] == .1
    assert all(new["base_noise_sums"] == previous["base_noise_sums"]
               and new["paired_rng_digest"] == previous["paired_rng_digest"]
               and new["dv12_rng_digest"] == previous["dv12_rng_digest"]
               for new, previous in zip(actual, old))
    assert actual[-1]["batch_indices"] != old[-1]["batch_indices"]


def test_exact_two_update_recovery_across_former_preservation_boundary():
    current = boundary()
    legacy = deepcopy(checkpoint(current))
    opt_in(current)
    tagged_initial = deepcopy(checkpoint(current))
    first = [training_update(current) for _ in range(3)]
    mid = deepcopy(checkpoint(current))
    last = [training_update(current) for _ in range(2)]
    final = state_digest(checkpoint(current))
    restore(current, mid)
    assert [training_update(current) for _ in range(2)] == last
    assert state_digest(checkpoint(current)) == final
    assert last[-1]["step"] == 1605 and not last[-1]["hold"]
    digest = state_digest(checkpoint(current))
    with pytest.raises(ValueError):
        restore(current, legacy)
    assert state_digest(checkpoint(current)) == digest
    restore(current, tagged_initial)
    assert [training_update(current) for _ in range(3)] == first


def test_saved_factory_rebuilds_explicit_law_and_refuses_other_config(monkeypatch):
    import supra.particle_training as source
    plain = boundary()
    base = deepcopy(plain.config)
    opt_in(plain)
    saved = deepcopy(checkpoint(plain))
    calls = []

    def factory(base_model, data, **options):
        calls.append(options)
        result = boundary()
        result.config = deepcopy(base)
        return result

    monkeypatch.setattr(source, "make_training_loop", factory)
    rebuilt = make_from_saved(object(), {}, saved, device="cpu")
    restore(rebuilt, saved)
    assert state_digest(checkpoint(rebuilt)) == state_digest(saved)
    assert calls == [dict(device="cpu", architecture=base["architecture"], branch_lr=5e-5,
                         probe_interval=100, profile="pr223_shared_routed_v1")]
    wrong = deepcopy(saved)
    wrong["config"]["training_schedule"] = "editing_only_v2"
    with pytest.raises(ValueError):
        make_from_saved(object(), {}, wrong, device="cpu")
    assert len(calls) == 1  # Reject the law before constructing native owners.
    wrong = deepcopy(saved)
    wrong["config"]["unrecognized_law"] = True
    with pytest.raises(ValueError):
        make_from_saved(object(), {}, wrong, device="cpu")
