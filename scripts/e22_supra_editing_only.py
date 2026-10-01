"""Explicit opt-in fit-only schedule after the qualified V3 update1600 boundary.

The native game/update implementation is unchanged. This module only versions
the application sampling schedule and chooses fit batches with task weight1.
Existing checkpoints/default training code keep their original configuration.
"""
from copy import deepcopy

EDITING_ONLY = "editing_only_v1"
START_STEP = 1600
BASE_SAMPLING = "original CPU generator 7; preservation every fifth update"
EDITING_SAMPLING = "original CPU generator 7; preservation every fifth update through1600; editing only after1600"
SCHEDULE_KEYS = ("training_schedule", "schedule_start_step", "pre_switch_edit_updates", "pre_switch_preservation_updates")


def editing_config(base_config):
    if any(key in base_config for key in SCHEDULE_KEYS):
        raise ValueError("editing schedule requires an unmodified base configuration")
    if base_config.get("sampling") != BASE_SAMPLING or base_config.get("batch_size") != 4:
        raise ValueError("editing schedule requires the qualified original sampling contract")
    if base_config.get("preservation_game_weight") != .1:
        raise ValueError("historical preservation weight differs")
    config = deepcopy(base_config)
    config.update(training_schedule=EDITING_ONLY, schedule_start_step=START_STEP,
        pre_switch_edit_updates=1280, pre_switch_preservation_updates=320, sampling=EDITING_SAMPLING)
    return config


def validate_editing_config(config):
    if config.get("training_schedule") != EDITING_ONLY:
        raise ValueError("unsupported saved editing schedule")
    if (config.get("schedule_start_step") != START_STEP or config.get("pre_switch_edit_updates") != 1280
            or config.get("pre_switch_preservation_updates") != 320 or config.get("sampling") != EDITING_SAMPLING):
        raise ValueError("editing schedule boundary/prefix differs")
    original = {key: value for key, value in config.items() if key not in SCHEDULE_KEYS}
    original["sampling"] = BASE_SAMPLING
    if editing_config(original) != config:
        raise ValueError("editing schedule configuration differs")
    return original


def opt_in(loop):
    if loop.policy.completed_steps != START_STEP:
        raise ValueError("editing schedule can begin only at the qualified update1600 boundary")
    loop.config = editing_config(loop.config)


def make_from_saved(base, data, saved, *, device):
    from supra.particle_training import LEGACY_ROUTED_PROFILE, make_training_loop
    config = saved["config"]
    edited = "training_schedule" in config
    if edited:
        validate_editing_config(config)
    elif any(key in config for key in SCHEDULE_KEYS):
        raise ValueError("incomplete saved editing schedule")
    loop = make_training_loop(base, data, device=device, architecture=config["architecture"],
        branch_lr=config["branch_lr"], probe_interval=config["probe_interval"],
        profile=config.get("particle_profile", LEGACY_ROUTED_PROFILE))
    if edited:
        loop.config = editing_config(loop.config)
    if loop.config != config:
        raise ValueError("saved native configuration differs from explicit factory configuration")
    return loop


def training_update(loop):
    import torch
    from supra.particle_game import update
    validate_editing_config(loop.config)
    if loop.policy.completed_steps < START_STEP:
        raise ValueError("editing schedule cannot reinterpret the inherited training prefix")
    pool = loop.fit_context
    indices = torch.randint(len(pool), (4,), generator=loop.data_rng)
    context = pool[indices.to(pool.device)]
    with torch.autograd.set_multithreading_enabled(False):
        row = update(loop, context=context, target=torch.zeros(4, 4, 32, 32, device=pool.device),
                     batch_indices=indices, game_weight=1.)
    row.update(hold=False, game_weight=1.)
    for name in ("loss_d", "loss_g", "penalty", "bank_grad_norm", "output_sigma"):
        if not torch.isfinite(torch.tensor(row[name])):
            raise RuntimeError(f"nonfinite {name}: {row}")
    return row
