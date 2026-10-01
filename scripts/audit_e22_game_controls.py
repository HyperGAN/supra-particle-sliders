"""Read saved native controls and demonstrate their separate dependencies on CPU.

This is an audit, not a training run. It never evaluates output accuracy and
does not modify the checkpoints or the installed ParticleGAN package.
"""
from argparse import ArgumentParser
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn


def tensor_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_observation(path):
    policy = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["policy"]
    critic_opt = policy["optimizers"][1]
    record = critic_opt["regularizer"]["record"]
    tester = policy["lr_settle"][1][0]
    table_tester = policy["lr_settle"][0][2]
    base_lr = policy["initial_lrs"][1][0]
    actual_lr = critic_opt["param_groups"][0]["lr"]
    return policy, dict(
        checkpoint=str(path), step=policy["completed_steps"],
        critic_tester_scale=tester["s"], table_tester_scale=table_tester["s"],
        proposed_critic_lr=base_lr * tester["s"], applied_critic_lr=actual_lr,
        applied_base_lr_ratio=actual_lr / base_lr,
        stationary_decisions=tester["counts"]["stationary"],
        ka2_alpha=record["alpha"], ka2_weight=record["w"],
        ka2_ratio=record["last_ratio"], ka2_ema_updates=record["ema_updates"],
        ka2_ema_skips=record["ema_skips"],
        critic_digest=tensor_digest(policy["models"]["critic"]),
        critic_ema_digest=tensor_digest(critic_opt["regularizer"]["ema"]),
        controller_game_trust=policy["controller"]["game_trust"],
        controller_data_drive=policy["controller"]["data_drive"],
        optimizer_surprise_fires=policy["surprise"]["fires"],
        optimizer_surprise_ratio=policy["surprise"]["last_ratio"],
        optimizer_surprise_since_calm=policy["surprise"]["since_calm"],
        optimizer_surprise_streak=policy["surprise"]["streak"],
    )


def floor_witness(policy_state):
    """Exercise the native begin-step LR assignment, without an optimizer step."""
    from particlegan import GANTrainer, get_recipe, init

    result = []
    tiny = policy_state["lr_settle"][1][0]["s"]
    for critic_scale, table_scale in ((.25, 1.), (tiny, 1.), (tiny, .25)):
        recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=2)
        generator, critic = nn.Linear(2, 2), nn.Linear(2, 1)
        init.deterministic_orthogonal_(generator, seed=0)
        init.deterministic_orthogonal_(critic, seed=1)
        trainer = GANTrainer(recipe, generator, critic, seed=21,
                             optimizer_options={"foreach": False})
        policy = trainer.policy
        policy.lr_settle.testers[1][0].s = critic_scale
        policy._table_tester().s = table_scale
        policy.controller.payoff_error = policy_state["controller"]["payoff_error"]
        policy.begin_step(torch.zeros(2, 2))
        result.append(dict(critic_tester_scale=critic_scale, table_tester_scale=table_scale,
                           proposed_critic_lr=policy.initial_lrs[1][0] * critic_scale,
                           applied_critic_lr=policy.opt_d.param_groups[0]["lr"],
                           payoff_damping=policy.controller.critic_scale(),
                           native_optimizer_step_performed=False))
        policy.abort_step()
    assert result[0]["applied_critic_lr"] == result[1]["applied_critic_lr"]
    assert result[2]["applied_critic_lr"] == result[1]["applied_critic_lr"] * .25
    return result


def alpha_witness(policy_state):
    """Change only recorded LR fields before the native KA2 blend decision.

    The additional nonzero-alpha case rules out a vacuous equality caused by
    the actual checkpoint's zero alpha. The substituted alpha is explicitly
    hypothetical; this is not a game continuation or optimizer update.
    """
    from particlegan.ka2 import KA2StepRecord

    saved = policy_state["optimizers"][1]["regularizer"]["record"]
    proposed = (policy_state["initial_lrs"][1][0]
                * policy_state["lr_settle"][1][0]["s"])
    actual = policy_state["optimizers"][1]["param_groups"][0]["lr"]
    rows = []
    for starting_alpha in (saved["alpha"], .4):
        outputs = []
        for recorded_lr in (proposed, actual):
            state = deepcopy(saved)
            state.update(lr_last=recorded_lr, lr_max=recorded_lr, alpha=starting_alpha)
            record = KA2StepRecord()
            record.load_state_dict(state)
            weight = record.advance_blend()
            alpha = record.alpha * policy_state["controller"]["game_trust"]
            outputs.append(dict(recorded_lr=recorded_lr, alpha_after_blend=alpha,
                                ema_decay=1. - alpha * (1. - record.anchor_min_decay),
                                native_blend_weight=weight, surprise_ratio=record.last_ratio))
        assert all(outputs[0][key] == outputs[1][key] for key in outputs[0] if key != "recorded_lr")
        rows.append(dict(starting_alpha=starting_alpha, hypothetical_alpha=starting_alpha != saved["alpha"],
                         outputs=outputs, alpha_and_decay_invariant_to_recorded_lr=True))
    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("--particlegan-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    torch.set_num_threads(1)
    import particlegan
    assert Path(particlegan.__file__).resolve().is_relative_to(args.particlegan_root.resolve())
    snapshots = []
    final_state = None
    for path in args.checkpoint:
        final_state, snapshot = checkpoint_observation(path)
        snapshots.append(snapshot)
    report = dict(
        scope="CPU mmap of control state and small critic/EMA tensors; native control-only witnesses; no training",
        particlegan_source=str(args.particlegan_root.resolve()),
        source_sha256={name: hashlib.sha256((args.particlegan_root / name).read_bytes()).hexdigest()
                       for name in ("particlegan/policy.py", "particlegan/ka2.py", "particlegan/continuous.py")},
        snapshots=snapshots,
        floor_witness=floor_witness(final_state),
        alpha_witness=alpha_witness(final_state),
        conclusions=[
            "Native critic stationarity proposals are masked by the table-linked floor before payoff damping.",
            "KA2 alpha and EMA decay do not consume the proposed or recorded critic LR.",
            "A frozen critic EMA can coexist with changing critic weights and a substantial applied critic LR.",
            "This establishes native control behavior, not its causal effect on the learned adversarial game.",
        ],
        proposed_game_diagnostic={
            "pairing": "same checkpoint, context, paired noise, DV12 stream and optimizer state",
            "intervention": "native KA2 anchor_weight=0 versus native anchor_weight=1; retain A and B caps",
            "observations": ["D adversarial gradient norm", "A and B-cap gradient norms",
                             "anchor-proximal gradient norm", "cosine of anchor and adversarial D gradients",
                             "native D update norm", "G payoff and G game-gradient norm"],
            "limits": "No output metric affects optimization or control; gradient conflict alone does not prove persistent repair.",
        },
        gpu_used=False, output_metrics_used=False, checkpoints_modified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), snapshots=len(snapshots),
                         floor_masks_critic_stationarity=True, ka2_alpha_independent_of_lr=True)), flush=True)


if __name__ == "__main__":
    main()
