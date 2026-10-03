"""Tiny public-API precision/ownership qualification; no actual Supra assets.

Exactly three native updates are confined to the final split/recovery case:
0->1->2 and a fresh owner's saved1->2 replay. All other cases are zero-update.
The actual full12800 two-step GPU qualifier is a separate prerequisite.
"""
from copy import deepcopy
import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import Recipe, RoutedRows, init
from supra import particle_final_precision as precision
from supra import particle_final_precision_export as export
from supra.particle_game import update
from supra.particle_pilot import state_digest
from supra.particle_training import raw_model_forward


MODES = (precision.BF16_FINAL, precision.FP32_FINAL)
SOFTWARE_NATIVE_UPDATES = 3


class TinyOrdinaryLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(16, 16)
        self.down = nn.Linear(16, 2, bias=False)
        self.up = nn.Linear(2, 16, bias=False)
        self.multiplier = 1.

    def forward(self, x):
        return self.base(x) + self.multiplier * self.up(self.down(x))


class TinyFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


class TinyBackbone(nn.Module):
    """One routed site, native physical patch layout, no downloaded backbone."""
    def __init__(self):
        super().__init__()
        self.projection = TinyOrdinaryLoRA()
        self.final = TinyFinal()

    def forward(self, z, t, context, mask):
        batch = len(z)
        patches = z.reshape(batch, 4, 16, 2, 16, 2).permute(0, 2, 4, 1, 3, 5).reshape(batch, 256, 16)
        text = (context[:, :, :16] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        hidden = (patches + .125 * text[:, None] + .05 * t[:, None, None]).tanh()
        output = self.final(self.projection(hidden))
        return output.reshape(batch, 16, 16, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(batch, 4, 32, 32)


def fresh_inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        model = TinyBackbone()
        streams = {name: torch.Generator(device="cpu").manual_seed(
            int.from_bytes(hashlib.sha256(("final-precision-tiny:" + name).encode()).digest()[:8], "little") % (2**63-1))
            for name, parameter in model.named_parameters() if parameter.requires_grad}
        init.initialize_(model, method="sample_distributions_v1", parameter_generators=streams)
        model.requires_grad_(False).eval()
    context = torch.zeros(4, 4100)
    context[:, :4096] = torch.linspace(-.17, .23, 4 * 4096).reshape(4, 4096)
    context[:, 4096] = torch.tensor([.1, .35, .6, .85])
    context[:, 4097:4099] = torch.tensor([1., 2.])
    context[:, 4099] = 1.
    guard = context.clone(); guard[:, 0] += .01
    holds = context.clone(); holds[:, 4098] = holds[:, 4097]
    data = dict(text_contexts=torch.linspace(-.13, .19, 3 * 3 * 768).reshape(3, 3, 768),
                text_masks=torch.ones(3, 3), coordinate_scale=torch.ones(16),
                fit=dict(context=context, targets=torch.zeros(4, 4, 32, 32)),
                guard=dict(context=guard), holds=dict(context=holds))
    return model, data


def fresh_loop(mode):
    base, data = fresh_inputs()
    loop = precision.make_precision_training_loop(base, data, precision=mode, device="cpu", probe_interval=1000)
    return base, data, loop


def expected_head_class(mode):
    return {precision.BF16_FINAL: precision.BF16StudentFinalLinear,
            precision.FP32_FINAL: precision.FP32StudentFinalLinear}[mode]


def assert_owners(loop, mode):
    p = loop.policy
    precision.assert_policy_precision(p, mode)
    for generator in (p.G, p.ema_G):
        head = generator.model.final.linear
        assert type(head) is expected_head_class(mode)
        assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in head.parameters())
    assert type(p.encoder.teacher.final.linear) is nn.Linear
    assert type(p.ema_encoder.teacher.final.linear) is nn.Linear
    for fast, average in ((p.G, p.ema_G), (p.router, p.ema_router), (p.encoder, p.ema_encoder)):
        fast_p, average_p = dict(fast.named_parameters()), dict(average.named_parameters())
        assert fast_p.keys() == average_p.keys()
        assert all(fast_p[key] is not average_p[key] and fast_p[key].data_ptr() != average_p[key].data_ptr() for key in fast_p)
    student = p.G.model.final.linear; teacher = p.encoder.teacher.final.linear
    assert all(a is not b and a.data_ptr() != b.data_ptr() for a, b in zip(student.parameters(), teacher.parameters()))


def public_raw(served, context):
    def unused_features(*args):
        raise AssertionError("clean reload qualification does not query a critic")
    routing = RoutedRows(model_forward=raw_model_forward, features=unused_features, sites=served.generator.sites)
    candidate = routing.candidate_for(served.models, served.table, averaged=served.source == "averaged")
    with torch.no_grad():
        return routing.forward(served.models, context, candidate)


@pytest.fixture(autouse=True)
def caller_scope():
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


@pytest.mark.parametrize("mode", MODES)
def test_fresh_public_initialization_precedes_optimizers_and_owned_EMA(monkeypatch, mode):
    events, initialized, generators, priors = [], set(), [], []
    original_init = init.initialize_
    original_g_optimizer = Recipe.make_generator_optimizer
    original_d_optimizer = Recipe.make_critic_optimizer

    def observe_init(module, **kwargs):
        result = original_init(module, **kwargs)
        initialized.update(id(p) for p in module.parameters() if p.requires_grad)
        if hasattr(module, "model") and hasattr(module.model, "final"):
            generators.append(module)
        if set(dict(module.named_parameters())) == {"z"}:
            priors.append(module)
            assert kwargs["method"] == "sample_distributions_v1"
            assert set(kwargs["parameter_generators"]) == set(kwargs["distributions"]) == {"z"}
            assert isinstance(kwargs["distributions"]["z"], init.Normal)
            events.append("prior_initialize")
        else:
            events.append("initialize")
        return result

    def observe_g_optimizer(recipe, parameters, *args, **kwargs):
        assert generators and type(generators[-1].model.final.linear) is expected_head_class(mode)
        assert len(priors) == 1 and kwargs["latent_table"] is priors[0].z
        for group in parameters:
            assert all(id(p) in initialized for p in group["params"])
        events.append("G_optimizer")
        return original_g_optimizer(recipe, parameters, *args, **kwargs)

    def observe_d_optimizer(recipe, critic, *args, **kwargs):
        assert all(id(p) in initialized for p in critic.parameters() if p.requires_grad)
        assert state_digest(critic.state_dict()) == state_digest(kwargs["ema_critic"].state_dict())
        events.append("D_optimizer")
        return original_d_optimizer(recipe, critic, *args, **kwargs)

    monkeypatch.setattr(init, "initialize_", observe_init)
    monkeypatch.setattr(Recipe, "make_generator_optimizer", observe_g_optimizer)
    monkeypatch.setattr(Recipe, "make_critic_optimizer", observe_d_optimizer)
    _, _, loop = fresh_loop(mode)
    assert events.count("G_optimizer") == events.count("D_optimizer") == 1
    assert events.count("prior_initialize") == 1
    assert events.index("initialize") < events.index("G_optimizer") < events.index("D_optimizer")
    assert events.index("prior_initialize") < events.index("G_optimizer")
    assert_owners(loop, mode)
    assert state_digest(loop.policy.G.state_dict()) == state_digest(loop.policy.ema_G.state_dict())
    for branch in loop.policy.G.particle_branches():
        assert not bool(branch.up.weight.count_nonzero())
        assert not bool(branch.bridge.weight[:, :loop.policy.G.rank].count_nonzero())
        assert not bool(branch.bridge.bias.count_nonzero())
        assert bool(branch.bridge.weight[:, loop.policy.G.rank:].count_nonzero())


def test_declared_BF16_and_FP32_math_preserves_frozen_F32_values_and_original_teacher():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(903)
        ordinary = nn.Linear(5, 3)
        init.initialize_(ordinary, method="sample_distributions_v1", parameter_generators={
            name: torch.Generator(device="cpu").manual_seed(904 + index)
            for index, (name, parameter) in enumerate(ordinary.named_parameters())})
        ordinary.requires_grad_(False)
    teacher = deepcopy(ordinary); teacher_state = state_digest(teacher.state_dict())
    x = torch.linspace(-.37129, .49817, 20).reshape(4, 5)
    outputs = {}
    for mode in MODES:
        generator = nn.Module(); generator.model = nn.Module(); generator.model.final = nn.Module()
        generator.model.final.linear = deepcopy(ordinary)
        head = generator.model.final.linear
        parameter_ids = {name: id(p) for name, p in head.named_parameters()}
        storage = {name: p.data_ptr() for name, p in head.named_parameters()}
        before = state_digest(head.state_dict())
        precision.install_student_final_(generator, mode, teacher_head=teacher)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs[mode] = generator.model.final.linear(x).float()
            teacher_output = teacher(x).float()
        oracle = F.linear(x.to(torch.bfloat16), ordinary.weight.to(torch.bfloat16), ordinary.bias.to(torch.bfloat16)).float()
        if mode == precision.FP32_FINAL:
            oracle = F.linear(x.float(), ordinary.weight.float(), ordinary.bias.float())
        assert torch.equal(outputs[mode], oracle)
        assert {name: id(p) for name, p in head.named_parameters()} == parameter_ids
        assert {name: p.data_ptr() for name, p in head.named_parameters()} == storage
        assert state_digest(head.state_dict()) == before
        assert torch.equal(teacher_output, F.linear(x.bfloat16(), teacher.weight.bfloat16(), teacher.bias.bfloat16()).float())
        assert type(teacher) is nn.Linear and state_digest(teacher.state_dict()) == teacher_state
    assert not torch.equal(outputs[precision.BF16_FINAL], outputs[precision.FP32_FINAL])


@pytest.mark.parametrize("mode", MODES)
def test_public_checkpoint_recovers_both_head_classes_without_reinitializing(tmp_path, monkeypatch, mode):
    _, _, loop = fresh_loop(mode)
    with torch.no_grad():
        loop.policy.G.particle_branches()[0].up.weight.fill_(.017)
        loop.policy.ema_G.particle_branches()[0].up.weight.fill_(.023)
    saved = precision.checkpoint_precision(loop)
    path = tmp_path / "owned-checkpoint.pt"; torch.save(saved, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    _, _, resumed = fresh_loop(mode)
    def forbidden(*args, **kwargs):
        raise AssertionError("restored learned owners must never initialize")
    monkeypatch.setattr(init, "initialize_", forbidden)
    precision.restore_precision(resumed, reloaded)
    assert_owners(resumed, mode)
    assert state_digest(precision.checkpoint_precision(resumed)) == state_digest(saved)
    assert not torch.equal(resumed.policy.G.particle_branches()[0].up.weight,
                           resumed.policy.ema_G.particle_branches()[0].up.weight)


def test_cross_precision_checkpoint_and_export_are_rejected_without_relabeling(tmp_path):
    base, _, fp32 = fresh_loop(precision.FP32_FINAL)
    _, _, bf16 = fresh_loop(precision.BF16_FINAL)
    saved = precision.checkpoint_precision(fp32)
    before = state_digest(precision.checkpoint_precision(bf16))
    with pytest.raises(ValueError, match="precision|arithmetic|mode"):
        precision.restore_precision(bf16, saved)
    assert state_digest(precision.checkpoint_precision(bf16)) == before
    served = fp32.policy.served_model()
    with pytest.raises(ValueError, match="precision|arithmetic|mode"):
        export.export_served_adapter(served, tmp_path / "wrong.safetensors", precision=precision.BF16_FINAL)
    assert not (tmp_path / "wrong.safetensors").exists()
    path = tmp_path / "correct.safetensors"; export.export_particle_adapter(fp32, path)
    with pytest.raises(ValueError, match="precision|arithmetic|mode"):
        export.load_particle_adapter(base, path, precision=precision.BF16_FINAL)


def test_offline_precision_exception_restores_FAST_EMA_teacher_and_training_tag():
    _, _, loop = fresh_loop(precision.FP32_FINAL)
    before = state_digest(precision.checkpoint_precision(loop))
    with pytest.raises(RuntimeError, match="offline evaluator failed"):
        with precision.serving_precision_scope(loop, precision.BF16_FINAL):
            assert type(loop.policy.G.model.final.linear) is precision.BF16StudentFinalLinear
            assert type(loop.policy.ema_G.model.final.linear) is precision.FP32StudentFinalLinear
            assert type(loop.policy.encoder.teacher.final.linear) is nn.Linear
            assert loop.config[precision.PRECISION_KEY] == precision.FP32_FINAL
            raise RuntimeError("offline evaluator failed")
    assert_owners(loop, precision.FP32_FINAL)
    assert state_digest(precision.checkpoint_precision(loop)) == before


@pytest.mark.parametrize("mode", MODES)
def test_serialized_clean_export_reload_is_exact_at_strength_zero_and_one(tmp_path, monkeypatch, mode):
    base, data, loop = fresh_loop(mode)
    with torch.no_grad():
        loop.policy.G.particle_branches()[0].up.weight.fill_(.031)
        loop.policy.ema_G.particle_branches()[0].up.weight.fill_(.031)
    served = loop.policy.served_model()
    expected = {}
    for strength in (0., 1.):
        context = data["fit"]["context"].clone(); context[:, 4099] = strength
        expected[strength] = public_raw(served, context)
    assert not torch.equal(expected[0.], expected[1.])
    path = tmp_path / "precision.safetensors"
    receipt = export.export_served_adapter(served, path, precision=mode)
    assert receipt["config"]["sampling"] == "clean"
    def forbidden(*args, **kwargs):
        raise AssertionError("serving loader must not initialize restored weights")
    monkeypatch.setattr(init, "initialize_", forbidden)
    adapter = export.load_particle_adapter(base, path, precision=mode)
    assert type(adapter.generator.model.final.linear) is expected_head_class(mode)
    context = data["fit"]["context"]
    z, t, text, mask, uncond, uncond_mask, _ = served.encoder.unpack(context)
    for strength in (0., 1.):
        actual = adapter.velocity(z, t, text, mask, uncond, uncond_mask, strength=strength)
        assert torch.equal(actual, expected[strength])
    assert not any(p.requires_grad for p in adapter.parameters())


def test_actual_tiny_next_step_replay_restores_optimizer_streams_and_precision_owners(tmp_path, monkeypatch):
    _, _, uninterrupted = fresh_loop(precision.FP32_FINAL)
    with precision.precision_rng_scope(uninterrupted):
        update(uninterrupted)
    midpoint = precision.checkpoint_precision(uninterrupted)
    path = tmp_path / "midpoint.pt"; torch.save(midpoint, path)
    with precision.precision_rng_scope(uninterrupted):
        row = update(uninterrupted)
    expected = state_digest(precision.checkpoint_precision(uninterrupted))
    _, _, resumed = fresh_loop(precision.FP32_FINAL)
    def forbidden(*args, **kwargs):
        raise AssertionError("restore must preserve the learned midpoint")
    monkeypatch.setattr(init, "initialize_", forbidden)
    precision.restore_precision(resumed, torch.load(path, map_location="cpu", weights_only=False))
    with precision.precision_rng_scope(resumed):
        replay = update(resumed)
    assert replay == row and row["step"] == 2
    assert state_digest(precision.checkpoint_precision(resumed)) == expected
    assert_owners(resumed, precision.FP32_FINAL)
