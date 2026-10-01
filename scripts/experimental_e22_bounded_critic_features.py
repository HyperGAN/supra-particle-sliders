"""Isolated critic-capacity intervention, never an ordinary native checkpoint.

features = tanh(a) + tanh(gain) * tanh(a / 4)

The new gain is zero on extension. Features remain strictly within [-2, 2],
and native input-gradient KA2/caps observe the entire changed function.
"""
from copy import deepcopy

import torch
from torch import nn

from supra.particle_game import ConditionalTokenCritic

FORMULATION = "bounded_second_feature_bypass_v1"


class BoundedFeatureCritic(ConditionalTokenCritic):
    def __init__(self, scale, *, condition_dim=769, width=48, feature_dim=16):
        super().__init__(scale, condition_dim=condition_dim, width=width, feature_dim=feature_dim)
        # Child registration after the existing modules keeps old parameter
        # positions intact. A root Parameter would silently reorder them.
        self.bypass = nn.ParameterList([nn.Parameter(torch.zeros(feature_dim))])

    def features(self, error, condition):
        native = super().features(error, condition)
        dtype = self.error_input.weight.dtype
        preactivation = self.feature_output((self.error_input(error.to(dtype=dtype))
            + self.condition_input(condition.to(dtype=dtype)).unsqueeze(1)).tanh())
        return native + self.bypass[0].tanh() * (preactivation / 4.).tanh()


def extend_native_state(critic_state, optimizer_state, critic):
    """Return explicit extended states; every old tensor/moment is unchanged."""
    if not isinstance(critic, BoundedFeatureCritic):
        raise TypeError("extension requires the tagged bounded critic class")
    state = deepcopy(critic_state)
    if "bypass.0" in state or "bypass.0" in optimizer_state["regularizer"]["ema"]:
        raise ValueError("state was already extended")
    names = list(dict(critic.named_parameters()))
    old_names = [name for name in names if name != "bypass.0"]
    if names != old_names + ["bypass.0"] or set(state) != set(critic.state_dict()) - {"bypass.0"}:
        raise ValueError("critic extension did not preserve the original state/parameter layout")
    extended_optimizer = deepcopy(optimizer_state)
    groups = extended_optimizer["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(old_names))):
        raise ValueError("extension requires the declared single native critic parameter group")
    fresh_gain = torch.zeros_like(critic.bypass[0])
    state["bypass.0"] = fresh_gain.clone()
    groups[0]["params"].append(len(old_names))
    # The new parameter alone starts with no Adam moments. All old moments,
    # counters, EMA parameters, cap/guard records and LR remain untouched.
    extended_optimizer["regularizer"]["ema"]["bypass.0"] = fresh_gain.clone()
    return state, extended_optimizer


def checkpoint(critic, optimizer, *, formulation):
    if formulation not in ("native_critic_v1", FORMULATION):
        raise ValueError("unknown isolated critic formulation")
    if (formulation == FORMULATION) != isinstance(critic, BoundedFeatureCritic):
        raise ValueError("critic class and checkpoint tag disagree")
    return dict(format="isolated_e22_critic_diagnostic_v1", formulation=formulation,
                critic=deepcopy(critic.state_dict()), optimizer=deepcopy(optimizer.state_dict()))


def restore(critic, optimizer, saved, *, formulation):
    if saved.get("format") != "isolated_e22_critic_diagnostic_v1" or saved.get("formulation") != formulation:
        raise ValueError("isolated critic state requires its exact formulation tag")
    if (formulation == FORMULATION) != isinstance(critic, BoundedFeatureCritic):
        raise ValueError("critic class does not match the requested formulation")
    deepcopy(critic).load_state_dict(saved["critic"], strict=True)
    critic.load_state_dict(saved["critic"], strict=True)
    optimizer.load_state_dict(deepcopy(saved["optimizer"]))
