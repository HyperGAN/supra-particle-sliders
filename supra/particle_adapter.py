"""Native Supra projection sites driven by one E22 routed particle bank.

The default preserves the original two-site hybrid verification model. An
explicit site list can replace every ordinary LoRA branch with a fresh particle
branch. Routing resources exist only during one complete supplied forward.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn


SITES = ("blocks.12.cross_attn.proj", "blocks.13.cross_attn.proj")
NONLINEAR_V1 = "nonlinear_v1"
LINEAR_MODULATED_V2 = "linear_modulated_v2"
PARTICLE_ARCHITECTURES = (NONLINEAR_V1, LINEAR_MODULATED_V2)


def validate_particle_architecture(architecture):
    """Validate the host formula independently of unchanged tensor shapes."""
    if not isinstance(architecture, str) or architecture not in PARTICLE_ARCHITECTURES:
        raise ValueError(f"unsupported particle architecture {architecture!r}; "
                         f"expected one of {PARTICLE_ARCHITECTURES}")
    return architecture


def adapter_sites(model):
    """Find ordinary LoRA sites in Supra's native forward declaration order.

    The attachment helper's target sweep has a different order. Supra declares
    its context projection first, then each block's self/cross projections in
    their execution order, which is also the required routed-site order.
    """
    return tuple(name for name, module in model.named_modules()
                 if name and isinstance(getattr(module, "base", None), nn.Linear)
                 and isinstance(getattr(module, "down", None), nn.Linear)
                 and isinstance(getattr(module, "up", None), nn.Linear)
                 and hasattr(module, "multiplier"))


def _validated_sites(sites):
    if isinstance(sites, (str, set, frozenset, Mapping)):
        raise ValueError("particle sites must be an ordered collection of distinct paths")
    sites = tuple(sites)
    if not sites or any(not isinstance(site, str) or not site for site in sites) or len(set(sites)) != len(sites):
        raise ValueError("particle sites must be nonempty distinct module paths")
    return sites


def adapter_input_dims(model, sites=None):
    """Return ordered query input widths from each frozen base projection."""
    sites = _validated_sites(adapter_sites(model) if sites is None else sites)
    dimensions = {}
    for site in sites:
        original = model.get_submodule(site)
        if not isinstance(getattr(original, "base", None), nn.Linear):
            raise ValueError(f"{site} must be an existing ordinary LoRA linear")
        dimensions[site] = original.base.in_features
    return dimensions


class SupraParticleRouter(nn.Module):
    """Shared site queries plus the bank's represented row masses."""

    def __init__(self, num_particles=128, z_dim=4, input_dim=576, *, site_input_dims=None):
        super().__init__()
        if min(num_particles, z_dim, input_dim) < 1:
            raise ValueError("router dimensions must be positive")
        if site_input_dims is None:
            # These names and tensor shapes preserve existing checkpoints.
            self.sites = SITES
            self.site_input_dims = dict.fromkeys(SITES, input_dim)
            self.first_query = nn.Linear(input_dim, z_dim)
            self.second_query = nn.Linear(input_dim, z_dim)
            self._query_names = dict(zip(SITES, ("first_query", "second_query")))
            self.site_queries = None
        else:
            if not isinstance(site_input_dims, Mapping):
                raise ValueError("site_input_dims must map ordered module paths to positive input widths")
            self.sites = _validated_sites(tuple(site_input_dims))
            self.site_input_dims = dict(site_input_dims)
            if any(type(width) is not int or width < 1 for width in self.site_input_dims.values()):
                raise ValueError("site query input widths must be positive integers")
            self._query_names = {site: f"query_{index:03d}" for index, site in enumerate(self.sites)}
            self.site_queries = nn.ModuleDict({self._query_names[site]: nn.Linear(width, z_dim)
                                               for site, width in self.site_input_dims.items()})
        self.register_buffer("log_mass", torch.zeros(num_particles))

    def query_for_site(self, site):
        try:
            name = self._query_names[site]
        except KeyError as error:
            raise ValueError(f"router has no query for particle site {site!r}") from error
        return getattr(self, name) if self.site_queries is None else self.site_queries[name]


Router = SupraParticleRouter


@dataclass(frozen=True)
class _ForwardFrame:
    candidate: object
    routing: object
    router: nn.Module
    batch_size: int
    guided: bool
    strength: float


class _ParticleProjection(nn.Module):
    def __init__(self, original, site, query_name, rank, z_dim, architecture=NONLINEAR_V1):
        super().__init__()
        self.architecture = validate_particle_architecture(architecture)
        if not isinstance(getattr(original, "base", None), nn.Linear):
            raise ValueError(f"{site} must be an existing ordinary LoRA linear")
        self.base = original.base.requires_grad_(False)
        self.down = nn.Linear(self.base.in_features, rank, bias=False)
        self.bridge = nn.Linear(rank + z_dim, rank)
        self.up = nn.Linear(rank, self.base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.site, self.query_name = site, query_name
        self.frame = None

    def forward(self, x):
        frame = self.frame
        if frame is None:
            raise RuntimeError("particle projections require forward_routed()")
        expected_batch = frame.batch_size * (2 if frame.guided else 1)
        if x.ndim != 3 or x.shape[0] != expected_batch:
            raise ValueError("Supra projection activations must be [batch, tokens, channels]")
        base_output = self.base(x)
        # These queries use the current host activation. In particular, the
        # second site is recomputed after the first site's bank-dependent edit.
        with torch.autocast(device_type=x.device.type, enabled=False):
            projected_input = x.float()
            query = frame.router.query_for_site(self.site)(projected_input)
            table = frame.candidate.table
            logits = query @ table.T / math.sqrt(table.shape[-1])
            if frame.guided:
                conditional, unconditional = logits.chunk(2, dim=0)
                logits = torch.stack((conditional, unconditional), dim=1)
            codes = frame.routing.mix(self.site, logits)
            if frame.guided:
                codes = torch.cat((codes[:, 0], codes[:, 1]), dim=0)
            if frame.strength == 0:
                return base_output
            hidden = self.down(projected_input)
            modulated = self.bridge(torch.cat((hidden, codes.float()), dim=-1)).tanh()
            if self.architecture == LINEAR_MODULATED_V2:
                # The same particle-conditioned branch retains a linear input
                # path when the bounded modulation saturates. Bank codes still
                # determine the modulation and every native routed-site mix.
                modulated = hidden + modulated
            delta = self.up(modulated)
            # Preserve the original frozen host's precision at its boundary.
            return (base_output.float() + frame.strength * delta).to(base_output.dtype)


class SupraParticleHost(nn.Module):
    """Frozen native Supra host with fresh versioned particle branches.

    ``pretrained_model`` is the loaded DiT with the ordinary published LoRA
    already attached. The caller retains ownership of text encoding and VAE
    decoding. All contexts and every candidate are explicit forward inputs.
    ``nonlinear_v1`` preserves the original formula for legacy checkpoints.
    ``linear_modulated_v2`` adds the unsquashed hidden features to the existing
    bounded particle modulation without changing parameter keys or shapes.
    """

    def __init__(self, pretrained_model, rank=16, z_dim=4, cfg=3, *, sites=None,
                 architecture=NONLINEAR_V1):
        super().__init__()
        if rank < 1 or z_dim < 1 or not math.isfinite(cfg):
            raise ValueError("rank/z_dim must be positive and cfg finite")
        self.architecture = validate_particle_architecture(architecture)
        self.model = deepcopy(pretrained_model).requires_grad_(False).eval()
        self.sites = _validated_sites(SITES if sites is None else sites)
        self.rank, self.z_dim, self.cfg = rank, z_dim, float(cfg)
        self._in_forward = False
        query_names = dict(zip(SITES, ("first_query", "second_query")))
        for site in self.sites:
            original = self.model.get_submodule(site)
            replacement = _ParticleProjection(original, site, query_names.get(site), rank, z_dim,
                                               self.architecture)
            reference = original.base.weight
            replacement.to(device=reference.device)
            parent_path, _, attribute = site.rpartition(".")
            parent = self.model.get_submodule(parent_path) if parent_path else self.model
            setattr(parent, attribute, replacement)

    def particle_branches(self):
        return tuple(self.model.get_submodule(site) for site in self.sites)

    @torch.no_grad()
    def zero_particle_outputs(self):
        """Restore the base start after an initializer touches trainable weights."""
        for branch in self.particle_branches():
            branch.up.weight.zero_()

    def forward_routed(self, z, t, ctx, mask, uncond_ctx, uncond_mask,
                       candidate, routing, router, strength=1, cfg=None):
        """Run complete native Supra/CFG with one ordered mix per real site.

        A CFG pair is two correlated uses of one original context, represented
        as the mixer's token axes [B, 2, tokens, rows]. It does not double ESS.
        Native ordering remains [conditional batch, unconditional batch].
        """
        if self._in_forward:
            raise RuntimeError("SupraParticleHost does not support overlapping routed forwards")
        if tuple(router.sites) != self.sites:
            raise ValueError("host and router must declare the same ordered particle sites")
        strength = float(strength)
        guidance = self.cfg if cfg is None else float(cfg)
        if not math.isfinite(strength) or not math.isfinite(guidance):
            raise ValueError("strength and cfg must be finite")
        if z.ndim != 4 or not len(z) or ctx.ndim != 3 or len(ctx) != len(z):
            raise ValueError("z and encoded context must share a nonempty batch")
        batch_size, guided = len(z), guidance > 1
        t = torch.as_tensor(t, device=z.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch_size)
        if t.numel() != batch_size:
            raise ValueError("time must be scalar or contain one value per context")
        if guided:
            if uncond_ctx is None or uncond_mask is None or mask is None:
                raise ValueError("CFG requires explicit conditional/unconditional masks and context")
            if uncond_ctx.ndim != 3 or len(uncond_ctx) not in (1, batch_size):
                raise ValueError("unconditional context must have batch one or the source batch")
            ctx = torch.cat((ctx, uncond_ctx.expand(batch_size, -1, -1)))
            mask = torch.cat((mask, uncond_mask.expand(batch_size, -1)))
            z, t = torch.cat((z, z)), torch.cat((t, t))
        branches = self.particle_branches()
        ordinary = [module for module in self.model.modules()
                    if hasattr(module, "multiplier") and hasattr(module, "down") and hasattr(module, "up")]
        previous = [(module, module.multiplier) for module in ordinary]
        frame = _ForwardFrame(candidate, routing, router, batch_size, guided, strength)
        self._in_forward = True
        try:
            for module, _ in previous:
                module.multiplier = strength
            for branch in branches:
                branch.frame = frame
            with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16,
                                enabled=z.device.type == "cuda"):
                velocity = self.model(z, t, ctx, mask).float()
            if guided:
                conditional, unconditional = velocity.chunk(2, dim=0)
                velocity = unconditional + guidance * (conditional - unconditional)
            return velocity
        finally:
            for branch in branches:
                branch.frame = None
            for module, multiplier in previous:
                module.multiplier = multiplier
            self._in_forward = False
