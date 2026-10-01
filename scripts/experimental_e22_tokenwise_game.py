"""Opt-in paired-token RpGAN payoff with unchanged pooled critic/KA2 units.

This changes only how D/G payoffs aggregate learned local critic scores.
ConditionalTokenCritic.forward remains pooled, and the native penalty still
receives [contexts*256,16] coordinates through ConditionalTokenPenaltyView.
Use this module's configure/checkpoint/restore/update entry points together.
The versioned wrapper refuses ordinary native checkpoints; migrating a native
boundary requires explicitly restoring it first, then calling configure().
No output metric, alternative initialization, controller override or guard is
introduced. This prototype has no GPU training result or production status.
"""
import torch

from particlegan import RoutedBatch
from supra.particle_game import ConditionalTokenCritic, _rng_digest, apply_critic_penalty, patchify
from supra.particle_pilot import checkpoint as native_checkpoint, restore as native_restore

FORMULATION = "paired_tokenwise_v1"
CHECKPOINT_SCHEMA = "supra_paired_game_aggregation_experimental_v1"
CONFIG_KEY = "game_aggregation"


def configure(loop):
    """Explicitly opt the current native boundary into this experimental law."""
    declared = loop.config.get(CONFIG_KEY)
    if declared not in (None, FORMULATION):
        raise ValueError("another game aggregation is already declared")
    if not isinstance(loop.policy.D, ConditionalTokenCritic):
        raise TypeError("paired-token game requires the conditional token critic")
    loop.config[CONFIG_KEY] = FORMULATION
    return loop


def _require_law(loop):
    if loop.config.get(CONFIG_KEY) != FORMULATION:
        raise ValueError("explicit paired_tokenwise_v1 configuration is required")
    if not isinstance(loop.policy.D, ConditionalTokenCritic):
        raise TypeError("paired-token game requires the unchanged conditional token critic")


def checkpoint(loop):
    """Save the complete native boundary under an explicit payoff-law schema."""
    _require_law(loop)
    return dict(schema=CHECKPOINT_SCHEMA, formulation=FORMULATION, native=native_checkpoint(loop))


def restore(loop, saved):
    """Strict restore: a legacy or different-law state cannot be reinterpreted."""
    _require_law(loop)
    if (saved.get("schema") != CHECKPOINT_SCHEMA or saved.get("formulation") != FORMULATION
            or saved.get("native", {}).get("config", {}).get(CONFIG_KEY) != FORMULATION):
        raise ValueError("paired-token restore requires its versioned checkpoint law")
    native_restore(loop, saved["native"])


def token_logits(critic, error, condition):
    """One learned score for each paired context/patch, in unchanged order."""
    features = critic.features(error, condition)
    logits = critic.score(features)
    if logits.shape != (*error.shape[:2], 1):
        raise ValueError("token game requires one score per context/patch")
    return logits.flatten(0, 1)


def update(loop, *, context=None, target=None, batch_indices=None, game_weight=1.):
    """Native E22 lifecycle; only the two paired payoff logit calls differ."""
    _require_law(loop)
    policy = loop.policy
    if context is None:
        indices = torch.randint(len(loop.fit_context), (policy.recipe.batch_size,),
                                generator=loop.data_rng, device=policy.device)
        context, target = loop.fit_context[indices], loop.fit_targets[indices]
    else:
        if target is None or batch_indices is None:
            raise ValueError("explicit batches require targets and source indices")
        indices = torch.as_tensor(batch_indices)
    if not 0 < game_weight <= 1:
        raise ValueError("game weight must be in (0,1]")
    condition = loop.condition(context) if hasattr(loop, "condition") else policy.encoder.condition(context)
    noise_shape = (len(target), 256, 16)
    critic_base = torch.randn(noise_shape, device=policy.device, dtype=torch.float32,
                              generator=loop.paired_noise_rng)
    generator_base = torch.randn(noise_shape, device=policy.device, dtype=torch.float32,
                                 generator=loop.paired_noise_rng)
    batch = RoutedBatch(context, target, loop.guard_context, loop.guard_targets)
    noise = policy.begin_step(target, routed=batch)
    loss = policy.recipe.make_loss()
    policy.G.eval()
    policy.encoder.eval()
    policy.router.eval()
    policy.D.train()
    with torch.no_grad():
        prediction = policy.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * critic_base
        fake = real + patchify(prediction - target).float() / policy.D.scale
    # Native continuous observation and KA2 receive the same contextual patch
    # tensors and the same POOLED forward. No token-count penalty multiplier.
    policy.observe_critic_pair(real, fake)
    penalty = apply_critic_penalty(policy.penalty, policy.D, real, fake, condition)
    loss_d_game = game_weight * loss.d_loss(token_logits(policy.D, real, condition),
                                         token_logits(policy.D, fake, condition))
    loss_d = loss_d_game + penalty
    policy.opt_d.zero_grad(set_to_none=True)
    policy.before_critic_backward()
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    policy.D.eval()
    policy.G.train()
    policy.encoder.train()
    policy.router.train()
    flags = [parameter.requires_grad for parameter in policy.D.parameters()]
    try:
        policy.D.requires_grad_(False)
        prediction = policy.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * generator_base
        with torch.no_grad():
            real_logits = token_logits(policy.D, real.detach(), condition)
        fake = real + patchify(prediction - target).float() / policy.D.scale
        loss_g = game_weight * loss.g_loss(token_logits(policy.D, fake, condition), real_logits)
        policy.opt_g.zero_grad(set_to_none=True)
        policy.before_generator_backward()
        loss_g.backward()
        gradient = policy.table.grad
        dense_rows = 0 if gradient is None else int(gradient.norm(dim=-1).gt(0).sum())
        bank_grad_norm = 0. if gradient is None else float(gradient.detach().double().norm())
        policy.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=loss_d_game.detach())
        policy.opt_g.step()
        policy.after_generator_step()
    finally:
        for parameter, flag in zip(policy.D.parameters(), flags):
            parameter.requires_grad_(flag)
    move = policy.finish_step()
    return dict(step=policy.completed_steps, loss_d=float(loss_d.detach()),
                loss_d_game=float(loss_d_game.detach()), loss_g=float(loss_g.detach()),
                penalty=float(penalty.detach()), penalty_phase=policy.penalty.last_stats.get("phase", "lazy_skip"),
                penalty_calls=policy.opt_d.record.calls, output_sigma=policy.output_sigma(), move=move,
                bank_grad_norm=bank_grad_norm, dense_gradient_rows=dense_rows,
                optimizer_surprise_fires=(0 if policy.surprise is None else policy.surprise.fires),
                optimizer_surprise_ratio=(None if policy.surprise is None else policy.surprise.last_ratio),
                anchor_release_events=(0 if policy.surprise is None else policy.surprise.anchor_events),
                batch_indices=indices.tolist(), base_noise_sums=[float(critic_base.sum()), float(generator_base.sum())],
                paired_rng_digest=_rng_digest(loop.paired_noise_rng), dv12_rng_digest=_rng_digest(policy.noise_generator),
                game_aggregation=FORMULATION)


def training_update(loop):
    """Keep the full Supra four-edit/one-preservation sampling and units."""
    _require_law(loop)
    hold = (loop.policy.completed_steps + 1) % 5 == 0
    pool = loop.hold_context if hold else loop.fit_context
    indices = torch.randint(len(pool), (4,), generator=loop.data_rng)
    context = pool[indices.to(pool.device)]
    with torch.autograd.set_multithreading_enabled(False):
        row = update(loop, context=context, target=torch.zeros(4, 4, 32, 32, device=pool.device),
                     batch_indices=indices, game_weight=.1 if hold else 1.)
    row.update(hold=hold, game_weight=.1 if hold else 1.)
    for name in ("loss_d", "loss_g", "penalty", "bank_grad_norm", "output_sigma"):
        if not torch.isfinite(torch.tensor(row[name])):
            raise RuntimeError(f"nonfinite {name}: {row}")
    return row
