"""Bounded, matched E22 verification on the native Supra image generator."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn

from particlegan import E22Policy, RoutedRows, get_recipe, init
from .particle_adapter import SITES, SupraParticleHost, SupraParticleRouter
from .particle_game import ConditionalTokenCritic, features_for_rows, patchify


class FrozenTextContexts(nn.Module):
    """Checkpoint-owned text contexts; source/time conditioning is explicit."""

    def __init__(self, contexts, masks, pretrained_model=None, cfg=3.):
        super().__init__()
        self.register_buffer("contexts", contexts.detach().clone())
        self.register_buffer("masks", masks.detach().clone())
        self.register_buffer("teacher_site_scale", torch.tensor(1., dtype=torch.float32))
        self.teacher = None if pretrained_model is None else deepcopy(pretrained_model).eval().requires_grad_(False)
        self.cfg = float(cfg)

    def unpack(self, context):
        if context.ndim != 2 or context.shape[1] != 4099:
            raise ValueError("packed Supra context must have shape [B,4099]")
        ids = context[:, 4097].long()
        if not torch.equal(ids.float(), context[:, 4097]) or bool((ids < 0).any()) or bool((ids >= len(self.contexts)).any()):
            raise ValueError("invalid caption IDs")
        if not bool((context[:, 4098] == context[0, 4098]).all()):
            raise ValueError("a routed batch must have one slider strength")
        return (context[:, :4096].reshape(-1, 4, 32, 32), context[:, 4096],
                self.contexts[ids], self.masks[ids], self.contexts[:1], self.masks[:1],
                float(context[0, 4098]))

    def condition(self, context):
        ids = context[:, 4097].long()
        text, mask = self.contexts[ids], self.masks[ids]
        pooled = (text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((pooled, context[:, 4096:4097]), dim=-1)

    @torch.no_grad()
    def teacher_velocity(self, context):
        """Pair with the exact current batch shape, including structural reruns.

        BF16 GEMM kernels can change rounding with batch size. A single cached
        velocity must not be treated as an exact target for a larger probe.
        """
        if self.teacher is None:
            raise RuntimeError("this context owner has no frozen teacher")
        self.teacher.eval()
        z, t, ctx, mask, uctx, umask, strength = self.unpack(context)
        if self.cfg > 1:
            batch = len(z)
            z, t = torch.cat((z, z)), torch.cat((t, t))
            ctx = torch.cat((ctx, uctx.expand(batch, -1, -1)))
            mask = torch.cat((mask, umask.expand(batch, -1)))
        ordinary = [(name, module) for name, module in self.teacher.named_modules()
                    if hasattr(module, "multiplier")]
        previous = [module.multiplier for _, module in ordinary]
        try:
            site_scale = float(self.teacher_site_scale)
            for name, module in ordinary:
                module.multiplier = strength * (site_scale if name in SITES else 1.)
            with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
                velocity = self.teacher(z, t, ctx, mask).float()
            if self.cfg > 1:
                conditional, unconditional = velocity.chunk(2)
                velocity = unconditional + self.cfg * (conditional - unconditional)
            return velocity
        finally:
            for (_, module), value in zip(ordinary, previous):
                module.multiplier = value


def model_forward(models, context, candidate, routing):
    z, t, ctx, mask, uctx, umask, strength = models["encoder"].unpack(context)
    prediction = models["generator"].forward_routed(z, t, ctx, mask, uctx, umask,
                                                  candidate, routing, models["router"], strength=strength)
    return prediction - models["encoder"].teacher_velocity(context)


@dataclass
class PilotLoop:
    policy: E22Policy
    fit_context: torch.Tensor
    fit_targets: torch.Tensor
    guard_context: torch.Tensor
    guard_targets: torch.Tensor
    data_rng: torch.Generator
    paired_noise_rng: torch.Generator
    config: dict


def make_loop(pretrained_model, data, *, mode, device="cuda:0", batch_size=1, probe_interval=20,
              branch_lr=1e-4, recipe_overrides=None):
    if mode not in ("fixed", "movable", "full"):
        raise ValueError("mode must be fixed, movable or full")
    overrides = {} if recipe_overrides is None else dict(recipe_overrides)
    unknown = set(overrides) - {"reopen_signal", "reopen_anchor"}
    if unknown:
        raise ValueError(f"pilot recipe overrides are limited to R1 controls: {sorted(unknown)}")
    device = torch.device(device)
    fit, guard = data["fit"], data["guard"]
    scale = patchify(fit["targets"] - fit["partial"]).std(dim=(0, 1)).clamp_min(.04).to(device)
    controls = mode == "full"
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=batch_size,
                        output_noise_std=.125, row_evidence_gate=controls, particle_birth_death=controls,
                        **overrides)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        generator = SupraParticleHost(pretrained_model).to(device)
        encoder = FrozenTextContexts(data["text_contexts"], data["text_masks"], pretrained_model).to(device)
        router = SupraParticleRouter().to(device)
        critic = ConditionalTokenCritic(scale).to(device)
        for module, role in ((generator, 0), (critic, 1), (encoder, 2), (router, 3)):
            init.deterministic_orthogonal_(module, seed=role)
        generator.zero_particle_outputs()
        table = init.deterministic_orthogonal_(recipe.make_prior()).to(device).z
        table.requires_grad_(mode != "fixed")
    groups = [dict(params=[p for p in generator.parameters() if p.requires_grad], lr=branch_lr),
              dict(params=list(router.parameters()), lr=branch_lr)]
    roles = ["generator", "router"]
    if table.requires_grad:
        groups.append(dict(params=[table], lr=recipe.lr * recipe.prior_lr_mult))
        roles.append("table")
    opt_g = recipe.make_generator_optimizer(groups, latent_table=table if table.requires_grad else None,
                                             foreach=False)
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    rows = RoutedRows(model_forward=model_forward, features=features_for_rows, sites=SITES,
                      probe_interval=probe_interval, max_context_harm=0., output_error_guard=False)
    policy = E22Policy(recipe, generator, critic, table=table, encoder=encoder, router=router,
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       roles=[roles, ["critic"]], routed_rows=rows, seed=21)
    policy.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    config = dict(mode=mode, bank=[128, 4], sites=list(SITES), native_tokens=256, cfg=3,
                  branch_rank=16, frozen_published_loras=69, batch_size=batch_size,
                  initialization="particlegan.init.deterministic_orthogonal_", role_keys=[0, 1, 2, 3],
                  generator_output="paired native velocity residual", teacher_pairing="same live batch shape and CFG",
                  branch_lr=branch_lr, recipe=recipe.to_dict(), penalty_units="token",
                  dataset_digest=state_digest({name: data[name] for name in
                                              ("fit", "guard", "test", "text_contexts", "text_masks")}),
                  probe_interval=probe_interval, max_feature_context_harm=0., output_error_guard=False,
                  structural_criterion="clean learned critic-feature proxy; game repair unproven")
    return PilotLoop(policy, fit["context"].to(device), torch.zeros_like(fit["targets"], device=device),
                     guard["context"].to(device), torch.zeros_like(guard["targets"], device=device),
                     torch.Generator(device=device).manual_seed(42),
                     torch.Generator(device=device).manual_seed(43), config)


def checkpoint(loop):
    return dict(policy=loop.policy.state_dict(), data_rng=loop.data_rng.get_state(),
                paired_noise_rng=loop.paired_noise_rng.get_state(), config=deepcopy(loop.config))


def restore(loop, state):
    if loop.config != state["config"]:
        raise ValueError("pilot checkpoint configuration mismatch")
    policy_state = state["policy"]
    families = ("models", "averages")
    if (isinstance(loop.policy.encoder, FrozenTextContexts)
            and "teacher_schedule" not in state["config"]
            and all("teacher_site_scale" not in policy_state[family]["encoder"] for family in families)):
        # Static pilots predating moving teachers used the published +1
        # branches. Copy mappings only; native checkpoint tensors can be large.
        policy_state = dict(policy_state)
        for family, owner in zip(families, (loop.policy.encoder, loop.policy.ema_encoder)):
            encoder_state = dict(policy_state[family]["encoder"])
            encoder_state["teacher_site_scale"] = torch.ones_like(owner.teacher_site_scale)
            policy_state[family] = {**policy_state[family], "encoder": encoder_state}
    loop.policy.load_state_dict(policy_state)
    loop.data_rng.set_state(state["data_rng"].cpu())
    loop.paired_noise_rng.set_state(state["paired_noise_rng"].cpu())


def switch_teacher(loop, site_scale):
    """Change only the two teacher branches at a completed update boundary.

    Contexts and zero residual targets remain valid: every paired forward
    computes the current teacher. The public checkpoint path also releases
    any temporary served-weight swap before updating both encoder owners.
    Optimizer, controller, model parameters and random streams are preserved.
    """
    site_scale = float(site_scale)
    if not math.isfinite(site_scale):
        raise ValueError("teacher site scale must be finite")
    state = checkpoint(loop)
    for family in ("models", "averages"):
        state["policy"][family]["encoder"]["teacher_site_scale"].fill_(site_scale)
    restore(loop, state)


def state_digest(value):
    """Digest every tensor and scalar in a nested state without sampling."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def initial_digest(loop):
    return state_digest(dict(models={name: module.state_dict() for name, module in loop.policy._training_modules().items()},
                             table=loop.policy.table))


def frozen_digest(loop):
    return state_digest({role: {name: parameter for name, parameter in module.named_parameters()
                               if not parameter.requires_grad} for role, module in
                         (("generator", loop.policy.G), ("encoder", loop.policy.encoder))})


@torch.no_grad()
def evaluate(loop, data, *, batch_size=1):
    """Reporting only; these values never change training or checkpoint choice."""
    served = loop.policy.served_model()
    records = {}
    for name in ("fit", "guard", "test"):
        split = data[name]
        outputs = torch.cat([(served.routed_forward(context.to(loop.policy.device)) +
                              served.encoder.teacher_velocity(context.to(loop.policy.device))).cpu()
                             for context in split["context"].split(batch_size)])
        targets = split["targets"]
        mse = float((outputs - targets).square().mean())
        missing_mse = float((split["partial"] - targets).square().mean())
        base_mse = float((split["base"] - targets).square().mean())
        features = []
        for context, output, target in zip(split["context"].split(batch_size), outputs.split(batch_size), targets.split(batch_size)):
            context, output, target = (item.to(loop.policy.device) for item in (context, output, target))
            condition = served.encoder.condition(context)
            residual = patchify(output - target) / served.critic.scale
            feature_error = served.critic.features(residual, condition) - served.critic.features(torch.zeros_like(residual), condition)
            features.append(feature_error.square().flatten(1).mean(1).cpu())
        records[name] = dict(velocity_rmse=mse**.5, relative_missing_branch_mse=mse/max(missing_mse, 1e-30),
                             initial_missing_branch_rmse=missing_mse**.5, full_teacher_base_rmse=base_mse**.5,
                             clean_critic_feature_error=float(torch.cat(features).mean()), contexts=len(targets))
    zero = data["test"]["context"][:1].clone().to(loop.policy.device)
    zero[:, 4098] = 0
    baseline = data["test"]["base"][:1].to(loop.policy.device)
    actual = served.routed_forward(zero) + served.encoder.teacher_velocity(zero)
    records["zero_strength"] = dict(exact=torch.equal(actual, baseline), max_error=float((actual-baseline).abs().max()))
    records["served_source"] = served.source
    return records
