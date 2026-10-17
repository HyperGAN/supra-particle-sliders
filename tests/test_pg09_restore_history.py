"""Qualify shared optimizer-history restore through the native Supra product IO."""
from copy import deepcopy
import json

import pytest
import torch

from supra.pg09 import frozen_digest, load_checkpoint, save_checkpoint
from supra.runtime import adapter_state
from supra.shared import file_sha, shared_module
from test_pg09_integration import FixtureRuntime, admitted, equal, run_for


def _rewrite_checkpoint(directory, mutate):
    """Keep the product checksum valid so the shared history validator must run."""
    path = directory / "checkpoint.pt"
    packet = torch.load(path, map_location="cpu", weights_only=True)
    mutate(packet["game"])
    torch.save(packet, path)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    previous = receipt["files"][path.name]
    receipt["files"][path.name] = file_sha(path)
    assert receipt["files"][path.name] != previous
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    assert all(file_sha(directory / name) == digest
               for name, digest in receipt["files"].items())


def _live_state(run):
    # Full resumable state includes both modules, both optimizers/regularizers,
    # every sampling stream/order/cursor/exposure, and the model RNG streams.
    from particlegan.bispf import _LAST_THERMAL_SCALE

    return deepcopy(dict(
        game=run.game.state_dict(), runtime=run.runtime.model.state_dict(),
        frozen=frozen_digest(run.runtime),
        flags=(run.game._failed, run.game._export_loaded),
        scales=[module.multiplier for module in run.runtime.loras],
        modes=[module.training for module in run.game.critic.modules()],
        trainable=[parameter.requires_grad for parameter in run.game.critic.parameters()],
        cpu_rng=torch.get_rng_state(), thermal=_LAST_THERMAL_SCALE,
    ))


def _assert_product_rejects_atomically(run, directory, monkeypatch, *, match):
    before = _live_state(run)
    original = run.game.load_state_dict
    assert original.__func__.__module__ == "particle_sliders.pg09"
    calls = []

    def shared_validator(state):
        calls.append(state["completed_steps"])
        return original(state)

    monkeypatch.setattr(run.game, "load_state_dict", shared_validator)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(run, directory)
    assert calls == [1], "The product must reach shared restore, beyond checksum validation"
    assert equal(_live_state(run), before)


def _started_checkpoint(admitted, tmp_path):
    source = run_for(admitted)
    source.step()
    directory = save_checkpoint(source, tmp_path / "step-1")
    receiver = run_for(admitted, FixtureRuntime(admitted[1]))
    assert receiver.game.completed_steps == 0
    return source, receiver, directory


@pytest.mark.parametrize("optimizer", ["opt_g", "opt_d"])
@pytest.mark.parametrize("moment", ["exp_avg_sq", "max_exp_avg_sq"])
def test_product_rejects_started_negative_second_moment_atomically(
        admitted, tmp_path, monkeypatch, optimizer, moment):
    source, receiver, directory = _started_checkpoint(admitted, tmp_path)

    def corrupt(game):
        saved = game[optimizer]
        key = next(iter(saved["state"]))
        parameter = next(parameter
                         for packet_group, live_group in zip(
                             saved["param_groups"], getattr(source.game, optimizer).param_groups)
                         for identifier, parameter in zip(packet_group["params"], live_group["params"])
                         if identifier == key)
        # The product selects zero-momentum DualNorm, not Adam. Inject a finite,
        # native-shape moment and require the shared negative-moment guard itself
        # to reject it before the additional-field/history-schema checks.
        saved["state"][key][moment] = torch.full_like(parameter, -1.)
        assert torch.isfinite(saved["state"][key][moment]).all()

    _rewrite_checkpoint(directory, corrupt)
    _assert_product_rejects_atomically(receiver, directory, monkeypatch,
                                      match="second moment is negative")


@pytest.mark.parametrize("optimizer", ["opt_g", "opt_d"])
@pytest.mark.parametrize("corruption", [
    "missing_slot", "extra_slot", "history_deleted", "history_emptied",
    "extra_parameter_history", "all_histories_deleted",
])
def test_product_rejects_started_history_corruption_atomically(
        admitted, tmp_path, monkeypatch, optimizer, corruption):
    _, receiver, directory = _started_checkpoint(admitted, tmp_path)

    def corrupt(game):
        saved = game[optimizer]
        key = next(iter(saved["state"]))
        assert saved["state"][key]  # This is an actual started parameter history.
        if corruption == "missing_slot":
            del saved["state"][key]["step"]
        elif corruption == "extra_slot":
            saved["state"][key]["unexpected"] = 1
        elif corruption == "history_deleted":
            del saved["state"][key]
        elif corruption == "history_emptied":
            saved["state"][key] = {}
        elif corruption == "extra_parameter_history":
            identifiers = [identifier for group in saved["param_groups"] for identifier in group["params"]]
            saved["state"][max(identifiers) + 1] = deepcopy(saved["state"][key])
        else:
            saved["state"] = {}
        # Retain the genuine presence schema; dropping optimizer history must
        # never be misrepresented as the valid never-used-parameter case.

    _rewrite_checkpoint(directory, corrupt)
    _assert_product_rejects_atomically(receiver, directory, monkeypatch,
                                      match="history|histories")


def test_product_requires_optimizer_history_presence_schema(admitted, tmp_path, monkeypatch):
    _, receiver, directory = _started_checkpoint(admitted, tmp_path)
    _rewrite_checkpoint(directory, lambda game: game.pop("optimizer_history_schema"))
    _assert_product_rejects_atomically(receiver, directory, monkeypatch,
                                      match="history presence")


def test_product_initial_empty_histories_resume_exact_native_update(admitted, tmp_path):
    source = run_for(admitted)
    initial = source.game.state_dict()
    assert initial["opt_g"]["state"] == initial["opt_d"]["state"] == {}
    assert all(value is None for phase in initial["optimizer_history_schema"].values()
               for value in phase.values())
    directory = save_checkpoint(source, tmp_path / "step-0")
    receiver = run_for(admitted, FixtureRuntime(admitted[1]))
    assert load_checkpoint(receiver, directory)["step"] == 0
    assert equal(receiver.game.state_dict(), initial)
    expected = source.step()
    assert receiver.step() == expected
    assert equal(receiver.game.state_dict(), source.game.state_dict())


class CriticWithUnusedParameter(torch.nn.Module):
    """Delegate native critic math; one truthful unused leaf tests absent state."""
    def __init__(self, channels, condition_width, width):
        super().__init__()
        native = shared_module("particle_sliders.pg09_moss_campaign").CampaignFieldCritic
        self.field_critic = native(channels, condition_width, width)
        self.unused = torch.nn.Parameter(torch.ones(2))

    def forward(self, response, condition):
        return self.field_critic(response, condition)


def test_product_unused_critic_history_exports_and_resumes_exact_native_update(
        admitted, tmp_path, monkeypatch):
    bridge = shared_module("particle_sliders.pg09_remaining_campaigns")
    monkeypatch.setattr(bridge, "CampaignFieldCritic", CriticWithUnusedParameter)
    source = run_for(admitted)
    original_frozen = frozen_digest(source.runtime)
    initial_adapter = deepcopy(adapter_state(source.runtime.model))
    initial_critic = deepcopy(source.game.critic.state_dict())
    source.step()
    assert not equal(adapter_state(source.runtime.model), initial_adapter)
    assert not equal(source.game.critic.state_dict(), initial_critic)
    directory = save_checkpoint(source, tmp_path / "step-1-unused-d")
    packet = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)["game"]
    unused = next(identifier
                  for saved_group, live_group in zip(packet["opt_d"]["param_groups"],
                                                    source.game.opt_d.param_groups)
                  for identifier, parameter in zip(saved_group["params"], live_group["params"])
                  if parameter is source.game.critic.unused)
    assert packet["opt_d"]["state"]
    assert unused not in packet["opt_d"]["state"]
    assert packet["optimizer_history_schema"]["d"][unused] is None

    # Exercise the actual native adapter artifact, including its product CFG
    # policy and lock receipt, rather than only a shared in-memory export.
    exported = FixtureRuntime(admitted[1])
    exported.load(directory / "adapter.safetensors")
    assert exported.cfg_adapter_branch == "conditional_only"
    assert equal(adapter_state(exported.model), adapter_state(source.runtime.model))
    record = source.rows.records[0]
    with torch.no_grad():
        expected_response = source.runtime.velocity(record["prompt"], record["z"], record["t"], scale=.75)
        exported_response = exported.velocity(record["prompt"], record["z"], record["t"], scale=.75)
    assert torch.equal(exported_response, expected_response)

    expected = source.step()
    expected_state = source.game.state_dict()
    receiver = run_for(admitted, FixtureRuntime(admitted[1]))
    assert load_checkpoint(receiver, directory)["step"] == 1
    assert equal(receiver.game.state_dict(), packet)
    assert receiver.step() == expected
    assert equal(receiver.game.state_dict(), expected_state)
    assert receiver.game.state_dict()["optimizer_history_schema"]["d"][unused] is None
    assert frozen_digest(receiver.runtime) == original_frozen
