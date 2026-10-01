"""Fresh initialization ownership and recovery; no training or GPU required."""
from copy import deepcopy

import pytest
import torch

from supra.particle_adapter import GATED_PARTICLE_V3, LINEAR_MODULATED_V2
from supra.particle_pilot import checkpoint, restore, state_digest
from supra.particle_training import (
    LEGACY_PARTICLE_INIT, NEUTRAL_PARTICLE_INIT, SAMPLED_PARTICLE_INIT,
    SHARED_ROUTED_PROFILE, make_training_loop, resolve_particle_init,
)
from supra.runtime import TARGETS, model_module


@pytest.fixture(autouse=True)
def cpu_scope():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def tiny_inputs(depth=14):
    module = model_module()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = module.SupraDiT(d_model=8, depth=depth, n_heads=2, ctx_dim=768,
                               mlp_ratio=2, num_tokens=256, patch=2)
        module.attach_supra_lora(model, rank=16, alpha=16, targets=TARGETS)
    context = torch.zeros(4, 4100)
    context[:, :4096] = torch.linspace(-.2, .3, 4 * 4096).reshape(4, 4096)
    context[:, 4096] = torch.tensor([.1, .35, .6, .85])
    context[:, 4097:4099] = torch.tensor([1., 2.])
    context[:, 4099] = 1
    data = dict(text_contexts=torch.linspace(-.1, .1, 3 * 3 * 768).reshape(3, 3, 768),
                text_masks=torch.ones(3, 3), coordinate_scale=torch.ones(16),
                fit=dict(context=context, targets=torch.zeros(4, 4, 32, 32)),
                guard=dict(context=context.clone()), holds=dict(context=context.clone()))
    return model, data


def loop_for(model, data, mode):
    return make_training_loop(model, data, device="cpu", architecture=GATED_PARTICLE_V3,
                              profile=SHARED_ROUTED_PROFILE, particle_init=mode)


def test_sampled_initialization_only_neutralizes_h_b_before_fast_and_ema(monkeypatch):
    from particlegan import init
    original_api, calls = init.initialize_, []

    def observed(module, **kwargs):
        streams = kwargs["parameter_generators"]
        assert kwargs["method"] == "sample_distributions_v1"
        assert set(streams) == {name for name, p in module.named_parameters() if p.requires_grad and p.numel()}
        assert len({id(stream) for stream in streams.values()}) == len(streams)
        assert all(stream.device.type == "cpu" and stream is not torch.default_generator for stream in streams.values())
        calls.append(module)
        return original_api(module, **kwargs)

    monkeypatch.setattr(init, "initialize_", observed)
    model, data = tiny_inputs()
    model_before, data_before = state_digest(model.state_dict()), state_digest(data)
    rng = torch.get_rng_state().clone()
    control = loop_for(model, data, SAMPLED_PARTICLE_INIT)
    neutral = loop_for(model, data, NEUTRAL_PARTICLE_INIT)
    assert len(calls) == 8
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(neutral.policy.G.sites) == 71
    assert neutral.config["initialization"] == "particlegan.init.initialize_"
    assert control.config["parameter_seeds"] == neutral.config["parameter_seeds"]
    assert {key: value for key, value in control.config.items() if key != "particle_init"} == {
        key: value for key, value in neutral.config.items() if key != "particle_init"}
    for actual, expected in ((neutral.policy.G, control.policy.G), (neutral.policy.ema_G, control.policy.ema_G)):
        for branch, sampled in zip(actual.particle_branches(), expected.particle_branches()):
            rank = actual.rank
            assert branch.bridge.weight[:, :rank].count_nonzero() == branch.bridge.bias.count_nonzero() == 0
            assert torch.equal(branch.bridge.weight[:, rank:], sampled.bridge.weight[:, rank:])
            assert branch.bridge.weight[:, rank:].count_nonzero() > 0
            assert torch.equal(branch.down.weight, sampled.down.weight)
            assert torch.equal(branch.up.weight, sampled.up.weight) and branch.up.weight.count_nonzero() == 0
            assert torch.equal(branch.base.weight, sampled.base.weight)
    assert all(branch.bridge.weight.requires_grad and branch.bridge.bias.requires_grad
               for branch in neutral.policy.G.particle_branches())
    owners = [deepcopy(loop.policy.state_dict()) for loop in (control, neutral)]
    for state in owners:
        state["models"].pop("generator")
        state["averages"].pop("generator")
    assert state_digest(owners[0]) == state_digest(owners[1])
    assert state_digest(model.state_dict()) == model_before and state_digest(data) == data_before
    with torch.no_grad():
        assert torch.equal(control.policy.routed_generate(data["fit"]["context"], sigma=0, perturb=False),
                           neutral.policy.routed_generate(data["fit"]["context"], sigma=0, perturb=False))


def test_neutral_checkpoint_loading_preserves_learned_h_b_and_rejects_cross_mode():
    model, data = tiny_inputs(depth=1)
    neutral = loop_for(model, data, NEUTRAL_PARTICLE_INIT)
    # Stand-ins for learned values exercise loading without spending on training.
    with torch.no_grad():
        for branch in neutral.policy.G.particle_branches():
            branch.bridge.weight[:, :neutral.policy.G.rank].fill_(.123)
            branch.bridge.bias.fill_(.234)
        for branch in neutral.policy.ema_G.particle_branches():
            branch.bridge.weight[:, :neutral.policy.G.rank].fill_(.345)
            branch.bridge.bias.fill_(.456)
    saved = checkpoint(neutral)
    resumed = loop_for(model, data, NEUTRAL_PARTICLE_INIT)
    restore(resumed, saved)
    assert state_digest(checkpoint(resumed)) == state_digest(saved)
    control = loop_for(model, data, SAMPLED_PARTICLE_INIT)
    before = state_digest(checkpoint(control))
    with pytest.raises(ValueError, match="configuration mismatch"):
        restore(control, saved)
    assert state_digest(checkpoint(control)) == before
    with pytest.raises(ValueError, match="fresh zero-up"):
        with torch.no_grad():
            resumed.policy.G.particle_branches()[0].up.weight.fill_(.1)
        resumed.policy.G.neutralize_hidden_initialization()


def test_retained_particles_and_h_b_receive_native_game_gradients_without_an_update():
    model, data = tiny_inputs(depth=2)
    loop = loop_for(model, data, NEUTRAL_PARTICLE_INIT)
    p = loop.policy
    # A nonzero read-only probe exposes paths masked by the fresh zero-up start.
    with torch.no_grad():
        for branch in p.G.particle_branches():
            branch.up.weight.fill_(.01)
    prediction = p.routed_generate(data["fit"]["context"], sigma=0, perturb=False)
    from supra.particle_game import patchify
    residual = patchify(prediction) / p.D.scale
    condition = p.encoder.condition(data["fit"]["context"])
    zero = torch.zeros_like(residual)
    game = p.recipe.make_loss().g_loss(p.D(residual, condition), p.D(zero, condition))
    game.backward()
    assert p.completed_steps == 0
    assert p.table.grad is not None and p.table.grad.norm(dim=-1).gt(0).all()
    assert all(query.weight.grad is not None and query.weight.grad.norm() > 0
               for query in p.router.site_queries.values())
    assert all(branch.bridge.weight.grad[:, :p.G.rank].norm() > 0
               and branch.bridge.weight.grad[:, p.G.rank:].norm() > 0
               and branch.bridge.bias.grad.norm() > 0 for branch in p.G.particle_branches())


def test_default_legacy_law_and_saved_mode_resolution_remain_exact(monkeypatch):
    from particlegan import init
    monkeypatch.setattr(init, "initialize_", lambda *args, **kwargs: pytest.fail("legacy initialization changed"))
    model, data = tiny_inputs(depth=1)
    implicit = make_training_loop(model, data, device="cpu", architecture=GATED_PARTICLE_V3)
    explicit = make_training_loop(model, data, device="cpu", architecture=GATED_PARTICLE_V3,
                                  particle_init=LEGACY_PARTICLE_INIT)
    assert implicit.config == explicit.config and "particle_init" not in implicit.config
    assert state_digest(checkpoint(implicit)) == state_digest(checkpoint(explicit))
    assert resolve_particle_init() == resolve_particle_init({}) == LEGACY_PARTICLE_INIT
    assert resolve_particle_init({"particle_init": NEUTRAL_PARTICLE_INIT}) == NEUTRAL_PARTICLE_INIT
    with pytest.raises(ValueError, match="differs"):
        resolve_particle_init({}, NEUTRAL_PARTICLE_INIT)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_particle_init({"particle_init": "unknown"})
    with pytest.raises(ValueError, match="requires gated_particle_v3"):
        make_training_loop(model, data, device="cpu", architecture=LINEAR_MODULATED_V2,
                           particle_init=NEUTRAL_PARTICLE_INIT)
