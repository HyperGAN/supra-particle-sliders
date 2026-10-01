"""Explicitly tagged research routing; never an ordinary V2 clean export.

Native RoutedExecution owns the original candidate table and row masses. The
projection receives a temporary derived-key table only for its logits; native
mixing, decoded values, usage attribution, DV12 and candidate replay still use
the unchanged original candidate. There are no additional trainable tensors.
"""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import torch

from supra.particle_adapter import LINEAR_MODULATED_V2, _ParticleProjection

FORMAT = "e22_supra_experimental_fixed_rms_keys_v1"
ROUTING = dict(name="fixed_rms_particle_keys_v1", version=1,
               key_formula="sqrt(z_dim)*table/table.norm(dim=-1).clamp_min(1e-12)",
               values="original candidate.table", mass="native additive candidate.log_mass",
               epsilon=1e-12, base_architecture=LINEAR_MODULATED_V2)


def fixed_rms_keys(table):
    return table * (math.sqrt(table.shape[-1]) / table.norm(dim=-1, keepdim=True).clamp_min(ROUTING["epsilon"]))


class FixedRMSKeyProjection(_ParticleProjection):
    """Reuse the frozen V2 projection, changing only the keys in its logits."""

    def forward(self, x):
        frame = self.frame
        if frame is None:
            raise RuntimeError("experimental particle projections require forward_routed()")
        # The routing execution retains the real candidate. Only the parent's
        # logits calculation reads this proxy; mix uses its own native values.
        key_candidate = SimpleNamespace(table=fixed_rms_keys(frame.candidate.table))
        self.frame = replace(frame, candidate=key_candidate)
        try:
            return super().forward(x)
        finally:
            self.frame = frame


def install_experimental_routing(loop):
    for host in (loop.policy.G, loop.policy.ema_G):
        if host.architecture != LINEAR_MODULATED_V2:
            raise ValueError("fixed-RMS keys require the declared V2 host")
        for branch in host.particle_branches():
            if type(branch) not in (_ParticleProjection, FixedRMSKeyProjection):
                raise ValueError("an unrecognized projection intervention is already installed")
            branch.__class__ = FixedRMSKeyProjection
        host.experimental_routing = deepcopy(ROUTING)
    loop.config["experimental_routing"] = deepcopy(ROUTING)


def experimental_checkpoint(loop):
    from supra.particle_pilot import checkpoint
    if loop.config.get("experimental_routing") != ROUTING:
        raise ValueError("experimental checkpoint requires its explicit routing configuration")
    if any(type(branch) is not FixedRMSKeyProjection
           for host in (loop.policy.G, loop.policy.ema_G) for branch in host.particle_branches()):
        raise ValueError("experimental routing must be installed on FAST and averaged hosts")
    return dict(format=FORMAT, experimental_routing=deepcopy(ROUTING), native=checkpoint(loop),
                ordinary_clean_export_supported=False, dedicated_restore="experimental_e22_fixed_rms_keys.restore_experimental")


def restore_experimental(loop, state):
    from supra.particle_pilot import restore
    if state.get("format") != FORMAT or state.get("experimental_routing") != ROUTING:
        raise ValueError("not a supported explicitly tagged fixed-RMS-key checkpoint")
    if state["native"]["config"].get("experimental_routing") != ROUTING:
        raise ValueError("native configuration does not describe the experimental forward")
    install_experimental_routing(loop)
    restore(loop, state["native"])
