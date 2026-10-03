"""Explicit final-head arithmetic around the unchanged public routed game.

Fresh owners use the existing named public initializer. Trained restoration
never initializes tensors. Native checkpoints alone do not encode Python
forward classes, so this module adds an explicit arithmetic envelope.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import E22Policy, RoutedRows, get_recipe, init
from .particle_adapter import (GATED_PARTICLE_V3, SupraParticleHost,
                               SupraParticleRouter, adapter_input_dims, adapter_sites)
from .particle_game import ConditionalTokenCritic, features_for_rows
from .particle_pilot import PilotLoop, checkpoint, restore, state_digest
from .particle_training import (NEUTRAL_PARTICLE_INIT, SAMPLED_SEED_NAMESPACE,
                                SHARED_ROUTED_PROFILE, initialize_sampled,
                                residual_model_forward)
from .particle_training_data import FrozenSliderContexts

BF16_FINAL = "student_final_bf16_v1"
FP32_FINAL = "student_final_fp32_v1"
PRECISIONS = (BF16_FINAL, FP32_FINAL)
CHECKPOINT_SCHEMA = "supra_final_precision_checkpoint_v1"
PRECISION_KEY = "student_final_precision"


class BF16StudentFinalLinear(nn.Linear):
    """Original Linear arithmetic in the host's enclosing CUDA BF16 autocast.

CPU fixture forwards retain the native host's CPU behavior. This class does
not introduce a new autocast context or quantize stored frozen Parameters.
"""
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class FP32StudentFinalLinear(nn.Linear):
    """FP32 input/matmul/return with the unchanged frozen FP32 Parameters."""
    def forward(self, x):
        if self.weight.dtype != torch.float32 or (
                self.bias is not None and self.bias.dtype != torch.float32):
            raise TypeError("student final head requires FP32 weight/bias storage")
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight, self.bias)


def validate_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError("unsupported student final precision")
    return precision


def precision_spec(precision):
    validate_precision(precision)
    return dict(mode=precision, frozen_storage="torch.float32", teacher="original enclosing BF16 arithmetic",
                arithmetic=("native enclosing CUDA BF16 autocast F.linear" if precision == BF16_FINAL else
                            "autocast-disabled FP32 input/weight/bias F.linear and FP32 return"),
                scope="student final.linear only; shared-Up particle branches, query, bridge and teacher unchanged",
                strength_zero_student_arithmetic_changes=(precision == FP32_FINAL))


def _head_class(precision):
    validate_precision(precision)
    return BF16StudentFinalLinear if precision == BF16_FINAL else FP32StudentFinalLinear


def _frozen_head(head):
    if not isinstance(head, nn.Linear):
        raise TypeError("student/teacher final head must be Linear")
    if any(p.dtype != torch.float32 or p.requires_grad or not bool(torch.isfinite(p).all())
           for p in head.parameters()):
        raise ValueError("final head requires finite frozen FP32 storage")


def _disjoint(left, right):
    if left is right or any(p is q or p.data_ptr() == q.data_ptr()
                            for p in left.parameters() for q in right.parameters()):
        raise ValueError("student/teacher/EMA final-head ownership aliases")


def install_student_final_(generator, precision, *, teacher_head=None):
    """Class installation only; no tensor copy, replacement or initialization.

Training constructors supply the independently frozen live teacher head.
Serving reconstruction has no teacher callback and may omit that argument.
"""
    head = generator.get_submodule("model.final.linear")
    _frozen_head(head)
    if type(head) not in (nn.Linear, BF16StudentFinalLinear, FP32StudentFinalLinear):
        raise TypeError("unsupported existing student final class")
    if teacher_head is not None:
        _frozen_head(teacher_head)
        if type(teacher_head) is not nn.Linear:
            raise ValueError("live teacher arithmetic must remain original Linear")
        _disjoint(head, teacher_head)
    identity = [(n, id(p), p.data_ptr(), p.dtype, p.requires_grad) for n, p in generator.named_parameters()]
    keys = tuple(generator.state_dict())
    head.__class__ = _head_class(precision)
    if identity != [(n, id(p), p.data_ptr(), p.dtype, p.requires_grad) for n, p in generator.named_parameters()] or keys != tuple(generator.state_dict()):
        raise AssertionError("final arithmetic installation changed Parameter ownership/keys")
    return precision_spec(precision)


def assert_policy_precision(policy, precision):
    expected = _head_class(precision)
    fast, average = (g.get_submodule("model.final.linear") for g in (policy.G, policy.ema_G))
    for head in (fast, average):
        _frozen_head(head)
        if type(head) is not expected:
            raise ValueError("FAST/EMA student final precision class mismatch")
    _disjoint(fast, average)
    for encoder in (policy.encoder, policy.ema_encoder):
        teacher = encoder.get_submodule("teacher.final.linear")
        _frozen_head(teacher)
        if type(teacher) is not nn.Linear:
            raise ValueError("teacher final arithmetic changed")
        _disjoint(fast, teacher); _disjoint(average, teacher)
    return precision_spec(precision)


@contextmanager
def serving_precision_scope(loop, precision, *, source="fast"):
    """Temporary named arithmetic for offline cross-evaluation only.

The checkpoint's training tag and EMA arithmetic remain unchanged. The caller
must not update or checkpoint while the selected head has temporary arithmetic.
Whole instance attributes/class are restored, including on evaluator errors.
"""
    validate_precision(precision)
    if source not in ("fast", "averaged"):
        raise ValueError("serving source must be fast or averaged")
    policy = loop.policy
    trained_precision = validate_precision(loop.config[PRECISION_KEY])
    assert_policy_precision(policy, trained_precision)
    generator = policy.G if source == "fast" else policy.ema_G
    encoder = policy.encoder if source == "fast" else policy.ema_encoder
    head = generator.model.final.linear
    teacher = encoder.teacher.final.linear
    original_class, original_attrs = type(head), dict(head.__dict__)
    config = deepcopy(loop.config)
    watched = (policy.G.model.final.linear, policy.ema_G.model.final.linear,
               policy.encoder.teacher.final.linear, policy.ema_encoder.teacher.final.linear)
    identities = [(type(owner), dict(owner.__dict__),
                   [(id(p), p.data_ptr(), p.dtype, p.requires_grad, p.detach().clone())
                    for p in owner.parameters()]) for owner in watched]
    try:
        install_student_final_(generator, precision, teacher_head=teacher)
        yield precision_spec(precision)
    finally:
        head.__class__ = original_class
        head.__dict__.clear(); head.__dict__.update(original_attrs)
        if loop.config != config:
            raise AssertionError("offline arithmetic scope changed the training tag/config")
        for owner, (cls, attrs, values) in zip(watched, identities):
            if type(owner) is not cls or set(owner.__dict__) != set(attrs) or any(
                    owner.__dict__[key] is not value for key, value in attrs.items()):
                raise AssertionError("offline arithmetic scope changed head class/attributes")
            actual = list(owner.parameters())
            if len(actual) != len(values) or any(
                    (id(p), p.data_ptr(), p.dtype, p.requires_grad) != saved[:4]
                    or not torch.equal(p, saved[4]) for p, saved in zip(actual, values)):
                raise AssertionError("offline arithmetic scope changed head ownership/storage/values")
        assert_policy_precision(policy, trained_precision)


def capture_globals(loop):
    """Caller-owned per-arm global streams; native private streams stay native."""
    p = loop.policy
    loop.globals = dict(cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=torch.cuda.get_rng_state(p.device).clone() if p.device.type == "cuda" else None)
    loop.globals_step = p.completed_steps


@contextmanager
def precision_rng_scope(loop):
    """Activate one arm for native updates; capture it and restore the caller.

Use this scope around the unchanged particle_game.update. No native private
generator is reset. A step performed outside this scope cannot be checkpointed
silently with stale global streams.
"""
    p = loop.policy
    if loop.globals_step != p.completed_steps:
        raise ValueError("native updates require precision_rng_scope or explicit capture_globals")
    devices = [p.device.index if p.device.index is not None else torch.cuda.current_device()] if p.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(loop.globals["cpu_rng"].cpu())
        if devices:
            torch.cuda.set_rng_state(loop.globals["cuda_rng"].cpu(), p.device)
        try:
            yield
        finally:
            capture_globals(loop)


def make_precision_training_loop(base_model, data, *, precision, device="cuda:0", probe_interval=100,
                                 branch_lr=5e-5):
    """Direct public factory; final class precedes optimizers and E22 EMA.

The source law is the existing full shared-Up, sampled-H/b-neutral factory.
Only student final arithmetic differs. This function never calls a trainer or
copies its update implementation; the caller uses particle_game.update.
"""
    validate_precision(precision); device = torch.device(device); sites = adapter_sites(base_model)
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=4,
                        output_noise_std=.125, birth_death_backend="auto", reopen_guard="settled")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(7)
        generator = SupraParticleHost(base_model, sites=sites, architecture=GATED_PARTICLE_V3).to(device)
        router = SupraParticleRouter(site_input_dims=adapter_input_dims(base_model, sites)).to(device)
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base_model).to(device)
        critic = ConditionalTokenCritic(data["coordinate_scale"].to(device)).to(device)
        seeds = {name: initialize_sampled(owner, role) for name, owner, role in (
            ("generator", generator, 0), ("critic", critic, 1), ("encoder", encoder, 2), ("router", router, 3))}
        generator.zero_particle_outputs(); generator.neutralize_hidden_initialization()
        prior = recipe.make_prior()
        bank_seed = int.from_bytes(hashlib.sha256(
            f"{SAMPLED_SEED_NAMESPACE}:4:z".encode()).digest()[:8], "little") % (2**63 - 1)
        init.initialize_(prior, method="sample_distributions_v1",
                         distributions={"z": init.Normal(0., prior.init_std)},
                         parameter_generators={"z": torch.Generator(device="cpu").manual_seed(bank_seed)})
        table = prior.to(device).z
        table.requires_grad_(True)
        install_student_final_(generator, precision, teacher_head=encoder.teacher.final.linear)
    opt_g = recipe.make_generator_optimizer([
        dict(params=[p for p in generator.parameters() if p.requires_grad], lr=branch_lr),
        dict(params=list(router.parameters()), lr=branch_lr),
        dict(params=[table], lr=recipe.lr * recipe.prior_lr_mult)], latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    rows = RoutedRows(model_forward=residual_model_forward, features=features_for_rows, sites=sites,
                      probe_interval=probe_interval, max_context_harm=0., output_error_guard=False)
    policy = E22Policy(recipe, generator, critic, table=table, encoder=encoder, router=router,
                      generator_optimizer=opt_g, critic_optimizer=opt_d,
                      roles=[["generator", "router", "table"], ["critic"]], routed_rows=rows, seed=21)
    policy.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    count = lambda owner: sum(p.numel() for p in owner.parameters() if p.requires_grad)
    config = dict(task="full final-boss slider", sites=list(sites), bank=[128,4], branch_rank=16,
        frozen_published_loras=0, batch_size=4, branch_lr=branch_lr, recipe=recipe.to_dict(),
        initialization="particlegan.init.initialize_", role_keys=[0,1,2,3],
        sampling="CPU generator7; fit only from update1", training_schedule="fresh_editing_only_v1",
        preservation_game_weight=0., penalty_weight_unchanged=True, probe_interval=probe_interval,
        output_error_guard=False, max_feature_context_harm=0., generator_parameters=count(generator),
        router_parameters=count(router), bank_parameters=table.numel(), critic_parameters=count(critic),
        dataset_digest=state_digest(data), checkpoint_selection="final fixed update horizon; evaluation only",
        teacher_pairing="frozen positive-prompt base at same live batch shape",
        structural_criterion="clean learned critic-feature proxy; game repair unproven",
        architecture=GATED_PARTICLE_V3, particle_profile=SHARED_ROUTED_PROFILE,
        particle_init=NEUTRAL_PARTICLE_INIT, initialization_method="sample_distributions_v1",
        parameter_seed_namespace=SAMPLED_SEED_NAMESPACE, parameter_seeds=seeds,
        student_final_precision=precision)
    loop = PilotLoop(policy, data["fit"]["context"].to(device),
        torch.zeros_like(data["fit"]["targets"], device=device), data["guard"]["context"].to(device),
        torch.zeros(len(data["guard"]["context"]),4,32,32,device=device),
        torch.Generator().manual_seed(7), torch.Generator(device=device).manual_seed(43), config)
    loop.hold_context = data["holds"]["context"].to(device)
    # This disposable fresh table is replaced by exact authenticated trained
    # FAST/EMA tables on bootstrap. Historical config stays byte-compatible.
    loop.fresh_bank_initialization = dict(api="particlegan.init.initialize_", role=4,
        parameter="z", seed=bank_seed, method="sample_distributions_v1",
        distribution=dict(kind="Normal", mean=0., std=prior.init_std),
        scope="fresh scratch owner before optimizer/EMA; trained bootstrap overwrites it")
    assert_policy_precision(policy, precision); capture_globals(loop)
    return loop


def bootstrap_precision(loop, legacy_state, *, expected_native_digest):
    """Authenticate exact original trained state BEFORE introducing its tag."""
    precision = validate_precision(loop.config[PRECISION_KEY]); assert_policy_precision(loop.policy, precision)
    if PRECISION_KEY in legacy_state["config"] or state_digest(legacy_state) != expected_native_digest:
        raise ValueError("original checkpoint precision/digest differs")
    config = deepcopy(loop.config); legacy_config = {k:v for k,v in config.items() if k != PRECISION_KEY}
    if legacy_config != legacy_state["config"]:
        raise ValueError("original editing-only factory/config differs")
    p = loop.policy
    devices = [p.device.index if p.device.index is not None else torch.cuda.current_device()] if p.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        loop.config = legacy_config
        try:
            restore(loop, legacy_state)
            if state_digest(checkpoint(loop)) != expected_native_digest:
                raise AssertionError("original native/caller roundtrip is not exact before precision tag")
            capture_globals(loop)
        finally:
            loop.config = config
    assert_policy_precision(loop.policy, precision)
    return dict(original_native_digest=expected_native_digest, legacy_roundtrip_exact=True,
                precision=precision_spec(precision), completed_steps=loop.policy.completed_steps,
                initialized_after_restore=False,
                fresh_bank_initialization=deepcopy(loop.fresh_bank_initialization))


def _owners(policy):
    return {name: owner for name, owner in (
        ("generator",policy.G), ("critic",policy.D), ("encoder",policy.encoder), ("router",policy.router),
        ("ema_generator",policy.ema_G), ("ema_critic",getattr(policy.opt_d,"ema_critic",None)),
        ("ema_encoder",policy.ema_encoder), ("ema_router",policy.ema_router)) if owner is not None}


def checkpoint_precision(loop):
    precision = validate_precision(loop.config[PRECISION_KEY]); assert_policy_precision(loop.policy, precision)
    with precision_rng_scope(loop):
        p = loop.policy; owners = _owners(p)
        caller = dict(
            modes={role:{n:m.training for n,m in owner.named_modules()} for role,owner in owners.items()},
            flags={role:{n:v.requires_grad for n,v in owner.named_parameters()} for role,owner in owners.items()},
            gradients={role:{n:None if v.grad is None else v.grad.detach().clone() for n,v in owner.named_parameters()} for role,owner in owners.items()},
            table_grad=None if p.table.grad is None else p.table.grad.detach().clone(),
            output_sigma_grad=None if p.log_output_sigma.grad is None else p.log_output_sigma.grad.detach().clone(),
            penalty_last_stats=deepcopy(p.penalty.last_stats), globals=deepcopy(loop.globals),
            globals_step=loop.globals_step)
        return dict(schema=CHECKPOINT_SCHEMA, precision=precision, precision_spec=precision_spec(precision),
                    pilot=checkpoint(loop), caller=caller)


def _restore_precision(loop, state):
    """Fresh matching classes must already exist; no initialization or class fix."""
    precision = validate_precision(loop.config[PRECISION_KEY])
    if (state.get("schema") != CHECKPOINT_SCHEMA or state.get("precision") != precision
            or state.get("precision_spec") != precision_spec(precision)
            or state["pilot"]["config"] != loop.config):
        raise ValueError("checkpoint precision/schema/config mismatch")
    assert_policy_precision(loop.policy, precision)
    p = loop.policy; owners = _owners(p); caller = state["caller"]
    if set(caller["modes"]) != set(owners) or set(caller["flags"]) != set(owners) or set(caller["gradients"]) != set(owners):
        raise ValueError("precision checkpoint caller owner mismatch")
    native = state["pilot"]["policy"]
    if (caller["globals_step"] != native["completed_steps"]
            or not torch.equal(caller["globals"]["cpu_rng"], native["cpu_rng"])
            or ((caller["globals"]["cuda_rng"] is None) != (native["cuda_rng"] is None))
            or (native["cuda_rng"] is not None and not torch.equal(caller["globals"]["cuda_rng"], native["cuda_rng"]))):
        raise ValueError("precision checkpoint global streams/clock mismatch")
    for role,owner in owners.items():
        if (set(caller["modes"][role]) != {n for n,_ in owner.named_modules()}
                or set(caller["flags"][role]) != {n for n,_ in owner.named_parameters()}
                or set(caller["gradients"][role]) != set(caller["flags"][role])):
            raise ValueError("precision checkpoint caller names mismatch")
    p.abort_step()
    for role,owner in owners.items():
        for n,v in owner.named_parameters(): v.requires_grad_(caller["flags"][role][n])
    restore(loop, state["pilot"])
    for role,owner in owners.items():
        for n,m in owner.named_modules(): m.training=caller["modes"][role][n]
        for n,v in owner.named_parameters():
            g=caller["gradients"][role][n]
            if g is not None and (g.shape != v.shape or g.dtype != v.dtype):
                raise ValueError("precision checkpoint gradient shape/dtype mismatch")
            v.grad=None if g is None else g.detach().clone().to(v.device)
    for parameter,name in ((p.table,"table_grad"),(p.log_output_sigma,"output_sigma_grad")):
        g=caller[name]
        if g is not None and (g.shape != parameter.shape or g.dtype != parameter.dtype):
            raise ValueError("precision checkpoint standalone gradient mismatch")
        parameter.grad=None if g is None else g.detach().clone().to(parameter.device)
    p.penalty.last_stats=deepcopy(caller["penalty_last_stats"])
    loop.globals=deepcopy(caller["globals"]);loop.globals_step=caller["globals_step"]
    assert_policy_precision(p, precision)


def restore_precision(loop, state):
    """Restore an authenticated owner envelope without advancing caller RNG."""
    p = loop.policy
    devices = [p.device.index if p.device.index is not None else torch.cuda.current_device()] if p.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        _restore_precision(loop, state)
