"""Full native Supra slider trained using the public ParticleGAN game API."""
from copy import deepcopy
import torch

from particlegan import E22Policy, RoutedRows, get_recipe, init
from .particle_adapter import (
    LINEAR_MODULATED_V2, NONLINEAR_V1, SupraParticleHost, SupraParticleRouter,
    adapter_sites, adapter_input_dims, validate_particle_architecture,
)
from .particle_game import ConditionalTokenCritic, features_for_rows, update
from .particle_pilot import PilotLoop, state_digest
from .particle_training_data import FrozenSliderContexts

LEGACY_ROUTED_PROFILE = "pr155_routed"
SHARED_ROUTED_PROFILE = "pr223_shared_routed_v1"
TRAINING_PROFILES = (LEGACY_ROUTED_PROFILE, SHARED_ROUTED_PROFILE)


def resolve_training_profile(config=None, requested=None):
    """Fresh CLI training uses shared fixes; historical resumes retain their law."""
    if requested is not None and requested not in TRAINING_PROFILES:
        raise ValueError(f"unsupported particle training profile: {requested}")
    if config is None:
        return SHARED_ROUTED_PROFILE if requested is None else requested
    saved = config.get("particle_profile", LEGACY_ROUTED_PROFILE)
    if saved not in TRAINING_PROFILES:
        raise ValueError(f"unsupported saved particle training profile: {saved}")
    if requested is not None and requested != saved:
        raise ValueError("resume particle profile differs from the saved configuration")
    return saved


def raw_model_forward(models, context, candidate, routing):
    z, t, ctx, mask, uctx, umask, strength = models["encoder"].unpack(context)
    return models["generator"].forward_routed(
        z, t, ctx, mask, uctx, umask, candidate, routing, models["router"], strength=strength)


def residual_model_forward(models, context, candidate, routing):
    return raw_model_forward(models, context, candidate, routing) - models["encoder"].teacher_velocity(context)


def make_training_loop(base_model, data, *, device="cuda:0", probe_interval=100, branch_lr=5e-5,
                       architecture=LINEAR_MODULATED_V2, profile=LEGACY_ROUTED_PROFILE):
    # Keep existing programmatic research callers and native checkpoints exact.
    # The full training CLI resolves the shared profile explicitly for new runs.
    profile = resolve_training_profile(requested=profile)
    architecture = validate_particle_architecture(architecture)
    device = torch.device(device)
    sites = adapter_sites(base_model)
    shared = (dict(birth_death_backend="auto", reopen_guard="settled")
              if profile == SHARED_ROUTED_PROFILE else {})
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=4,
                        output_noise_std=.125, **shared)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        generator = SupraParticleHost(base_model, sites=sites, architecture=architecture).to(device)
        router = SupraParticleRouter(site_input_dims=adapter_input_dims(base_model, sites)).to(device)
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base_model).to(device)
        critic = ConditionalTokenCritic(data["coordinate_scale"].to(device)).to(device)
        for module, role in ((generator, 0), (critic, 1), (encoder, 2), (router, 3)):
            init.deterministic_orthogonal_(module, seed=role)
        generator.zero_particle_outputs()
        table = init.deterministic_orthogonal_(recipe.make_prior()).to(device).z
        table.requires_grad_(True)
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
    count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
    config = dict(task="full final-boss slider", sites=list(sites), bank=[128,4], branch_rank=16,
                  frozen_published_loras=0, batch_size=4, branch_lr=branch_lr, recipe=recipe.to_dict(),
                  initialization="particlegan.init.deterministic_orthogonal_", role_keys=[0,1,2,3],
                  sampling="original CPU generator 7; preservation every fifth update",
                  preservation_game_weight=.1, penalty_weight_unchanged=True,
                  probe_interval=probe_interval, output_error_guard=False, max_feature_context_harm=0.,
                  generator_parameters=count(generator), router_parameters=count(router), bank_parameters=table.numel(),
                  critic_parameters=count(critic), dataset_digest=state_digest(data),
                  checkpoint_selection="final fixed update horizon; evaluation only",
                  teacher_pairing="frozen positive-prompt base at same live batch shape",
                  structural_criterion="clean learned critic-feature proxy; game repair unproven")
    # Legacy runs lack this key. Keep their config identical for strict native
    # checkpoint replay rather than silently interpreting their weights as V2.
    if architecture != NONLINEAR_V1:
        config["architecture"] = architecture
    if profile != LEGACY_ROUTED_PROFILE:
        config["particle_profile"] = profile
    loop = PilotLoop(policy, data["fit"]["context"].to(device),
                     torch.zeros_like(data["fit"]["targets"], device=device),
                     data["guard"]["context"].to(device),
                     torch.zeros(len(data["guard"]["context"]),4,32,32,device=device),
                     torch.Generator().manual_seed(7), torch.Generator(device=device).manual_seed(43), config)
    loop.hold_context = data["holds"]["context"].to(device)
    return loop


def training_update(loop):
    hold = (loop.policy.completed_steps + 1) % 5 == 0
    pool = loop.hold_context if hold else loop.fit_context
    indices = torch.randint(len(pool), (4,), generator=loop.data_rng)
    context = pool[indices.to(pool.device)]
    with torch.autograd.set_multithreading_enabled(False):
        row = update(loop, context=context, target=torch.zeros(4,4,32,32,device=pool.device),
                     batch_indices=indices, game_weight=.1 if hold else 1.)
    row.update(hold=hold, game_weight=.1 if hold else 1.)
    for name in ("loss_d", "loss_g", "penalty", "bank_grad_norm", "output_sigma"):
        if not torch.isfinite(torch.tensor(row[name])):
            raise RuntimeError(f"nonfinite {name}: {row}")
    return row


@torch.no_grad()
def raw_velocity(served, context):
    """Serve the native velocity directly, avoiding residual cancellation."""
    spec = RoutedRows(model_forward=raw_model_forward, features=features_for_rows,
                      sites=served.generator.sites)
    candidate = served.routing.candidate_for(served.models, served.table, averaged=served.source=="averaged")
    return spec.forward(served.models, context, candidate)
