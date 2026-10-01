"""Conditional paired-error RpGAN/KA2 for the complete frozen Supra host.

Teacher velocities define the paired residual, without a reconstruction loss.
The critic emits one score per source/time context. Only its penalty views the
input gradients in 16-coordinate velocity-patch units. The current ParticleGAN
row selector observes clean learned critic features; output error is evaluation
only and must not be installed as a row guard or checkpoint selector.
"""

import hashlib

import torch
from torch import nn

from particlegan import RoutedBatch


def patchify(velocity):
    """Map native [B,4,32,32] velocities to the DiT's [B,256,16] patches."""
    if not isinstance(velocity, torch.Tensor) or velocity.ndim != 4 or tuple(velocity.shape[1:]) != (4, 32, 32):
        raise ValueError("Supra velocities must have native shape [contexts,4,32,32]")
    if not velocity.is_floating_point() or not len(velocity):
        raise ValueError("Supra velocities must be floating and nonempty")
    batch = len(velocity)
    return velocity.reshape(batch, 4, 16, 2, 16, 2).permute(0, 2, 4, 1, 3, 5).reshape(batch, 256, 16)


class ConditionalTokenCritic(nn.Module):
    """Token features conditioned on frozen source embeddings and flow time.

    Condition inputs are supplied explicitly and are not penalty coordinates.
    The pooled context score preserves RpGAN's source-wise pairing contract.
    """

    def __init__(self, scale, *, condition_dim=769, width=48, feature_dim=16):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (condition_dim, width, feature_dim)):
            raise ValueError("critic dimensions must be positive integers")
        scale = torch.as_tensor(scale).detach().clone()
        if scale.shape != (16,) or not scale.is_floating_point() or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("critic scale must contain 16 finite positive patch-coordinate scales")
        self.condition_dim = condition_dim
        self.error_input = nn.Linear(16, width)
        self.condition_input = nn.Linear(condition_dim, width, bias=False)
        self.feature_output = nn.Linear(width, feature_dim)
        self.score = nn.Linear(feature_dim, 1)
        self.register_buffer("scale", scale)

    def features(self, error, condition):
        if (not isinstance(error, torch.Tensor) or error.ndim != 3 or not len(error)
                or not error.shape[1] or error.shape[2] != 16 or not error.is_floating_point()):
            raise ValueError("critic errors must have floating shape [contexts,tokens,16]")
        if (not isinstance(condition, torch.Tensor) or condition.shape != (len(error), self.condition_dim)
                or condition.device != error.device or not condition.is_floating_point()):
            raise ValueError("critic conditioning must match contexts and its declared source/time width")
        dtype = self.error_input.weight.dtype
        local = self.error_input(error.to(dtype=dtype))
        contextual = self.condition_input(condition.to(dtype=dtype)).unsqueeze(1)
        return self.feature_output((local + contextual).tanh()).tanh()

    def forward(self, error, condition):
        return self.score(self.features(error, condition).mean(dim=1))


class ConditionalTokenPenaltyView(nn.Module):
    """Use the same contextual critic in per-token input-gradient units.

    The direct critic child allows the native paired penalty to construct the
    corresponding EMA view. Repeated context scores cancel token pooling in
    the input gradient; flattening makes native KA2's dimension exactly 16.
    """

    def __init__(self, critic, tokens):
        super().__init__()
        if not isinstance(critic, nn.Module) or type(tokens) is not int or tokens < 1:
            raise ValueError("token penalty needs a critic module and a positive token count")
        self.critic, self.tokens = critic, tokens

    def forward(self, error, condition):
        if (not isinstance(error, torch.Tensor) or error.ndim != 2 or not len(error)
                or error.shape[0] % self.tokens or error.shape[1] != 16):
            raise ValueError("token penalty errors must have shape [contexts*tokens,16]")
        contextual = error.reshape(-1, self.tokens, 16)
        return self.critic(contextual, condition).repeat_interleave(self.tokens, dim=0)


def apply_critic_penalty(penalty, critic, real, fake, condition):
    """Forward fixed conditioning to native KA2 and its paired EMA critic."""
    if real.ndim != 3 or fake.shape != real.shape or real.shape[-1] != 16:
        raise ValueError("token penalty needs matching [contexts,tokens,16] errors")
    view = ConditionalTokenPenaltyView(critic, real.shape[1])
    return penalty(view, real.flatten(0, 1), fake.flatten(0, 1), condition)


def features_for_rows(models, context, samples, targets):
    """Current #155 clean learned-feature proxy, without a raw output guard."""
    critic = models["critic"]
    condition = models["encoder"].condition(context)
    residual = patchify(samples - targets).float() / critic.scale
    return critic.features(residual, condition).flatten(1)


def _rng_digest(stream):
    return hashlib.sha256(bytes(stream.get_state().cpu().tolist())).hexdigest()


def update(loop, *, context=None, target=None, batch_indices=None, game_weight=1.):
    """One native E22 D/G update through the complete routed Supra model."""
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
    policy.observe_critic_pair(real, fake)
    penalty = apply_critic_penalty(policy.penalty, policy.D, real, fake, condition)
    loss_d_game = game_weight * loss.d_loss(policy.D(real, condition), policy.D(fake, condition))
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
            real_logits = policy.D(real.detach(), condition)
        fake = real + patchify(prediction - target).float() / policy.D.scale
        loss_g = game_weight * loss.g_loss(policy.D(fake, condition), real_logits)
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
                paired_rng_digest=_rng_digest(loop.paired_noise_rng), dv12_rng_digest=_rng_digest(policy.noise_generator))
