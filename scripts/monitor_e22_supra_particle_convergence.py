"""Read-only fixed-game probes and rolling training losses during Supra runs."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@contextmanager
def evaluation_modes(policy):
    previous = [(module, module.training) for root in policy._training_modules().values()
                for module in root.modules()]
    try:
        for module, _ in previous:
            module.training = False
        yield
    finally:
        for module, training in previous:
            module.training = training


def rolling_losses(rows, *, window=100):
    """Keep preservation's .1 task weight out of displayed game units."""
    report = {}
    for name, hold in (("edit", False), ("preservation", True)):
        chosen = [row for row in rows if row["hold"] is hold][-window:]
        if not chosen:
            continue
        mean = lambda key: sum(row[key] for row in chosen) / len(chosen)
        report[name] = dict(updates=len(chosen), first_step=chosen[0]["step"], last_step=chosen[-1]["step"],
                           g_game=sum(row["loss_g"] / row["game_weight"] for row in chosen) / len(chosen),
                           d_game=sum(row["loss_d_game"] / row["game_weight"] for row in chosen) / len(chosen),
                           penalty=mean("penalty"), bank_grad_norm=mean("bank_grad_norm"))
    return report


class GameProgressMonitor:
    """Fixed training probes and Gaussian panels; never selects a checkpoint.

    Each probe compares the current FAST G with a frozen starting D and the
    live D, using identical panels. Clean and privately replayed DV12 forwards
    are reported separately. Complete native checkpoints must remain equal.
    """

    def __init__(self, loop, data, fixed_critic, output):
        self.fixed_critic = fixed_critic
        self.output = Path(output)
        self.history = []
        self.pools = {}
        stream = torch.Generator().manual_seed(72)
        for name, pool_name, per_subject in (("edit", "fit", 2), ("preservation", "holds", 4)):
            pool = data[pool_name]["context"]
            indices = []
            for subject in pool[:, 4097].unique(sorted=True):
                available = (pool[:, 4097] == subject).nonzero().flatten()
                offsets = torch.linspace(0, len(available) - 1, min(per_subject, len(available))).round().long()
                indices.extend(available[offsets].tolist())
            context = pool[indices].to(loop.policy.device)
            noise = torch.randn((4, len(indices), 256, 16), generator=stream).to(loop.policy.device)
            self.pools[name] = (context, noise)
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / f"progress-{name}-indices.json").write_text(json.dumps(indices) + "\n")
        self.dv12_state = loop.policy.noise_generator.get_state().clone()

    @torch.no_grad()
    def evaluate(self, loop, rows, *, sigma=.125):
        from supra.particle_game import patchify
        from supra.particle_pilot import checkpoint, state_digest
        policy = loop.policy
        before = state_digest(checkpoint(loop))
        candidate = policy.routed_control.candidate()
        spec = policy.routed_control.spec
        models = policy._training_modules()
        result = dict(step=policy.completed_steps, source="FAST current particle G", output_sigma=sigma,
                      training_output_sigma=policy.output_sigma(), rolling=rolling_losses(rows), probes={},
                      output_metrics_used=False, evaluation_only=True)
        with evaluation_modes(policy):
            for name, (contexts, noise) in self.pools.items():
                record = {mode: {judge: [] for judge in ("frozen_start_D", "live_D")}
                          for mode in ("clean", "dv12")}
                stream = torch.Generator(device=policy.device)
                stream.set_state(self.dv12_state)
                prior = policy.controller.routed_prior(candidate.table, candidate.log_mass)
                for start in range(0, len(contexts), 4):
                    context = contexts[start:start + 4]
                    condition = policy.encoder.condition(context).repeat(4, 1)
                    real = (sigma * noise[:, start:start + len(context)]).flatten(0, 1)
                    clean = spec.forward(models, context, candidate)
                    noisy = spec.forward(models, context, candidate, perturb_fn=lambda codes:
                                         policy.controller.perturb_latent(codes, stream, prior, record=False))
                    for mode, prediction in (("clean", clean), ("dv12", noisy)):
                        for judge, critic in (("frozen_start_D", self.fixed_critic), ("live_D", policy.D)):
                            fake = real + (patchify(prediction).float() / critic.scale).repeat(4, 1, 1)
                            gap = critic(real, condition) - critic(fake, condition)
                            record[mode][judge].extend(F.softplus(gap).reshape(4, len(context)).mean(0).cpu().tolist())
                result["probes"][name] = dict(contexts=len(contexts), **{
                    mode: {judge: sum(values) / len(values) for judge, values in scores.items()}
                    for mode, scores in record.items()})
        if before != state_digest(checkpoint(loop)):
            raise RuntimeError("progress probe changed native training state or RNGs")
        result["native_state_unchanged"] = True
        self.history.append(result)
        if len(self.history) >= 4:
            window = self.history[-4:]
            plateau = {}
            for name in self.pools:
                values = [row["probes"][name]["clean"]["frozen_start_D"] for row in window]
                improvement = (values[0] - values[-1]) / max(abs(values[0]), 1e-12)
                plateau[name] = dict(steps=window[-1]["step"] - window[0]["step"], relative_improvement=improvement,
                                    possible_plateau=abs(improvement) < .002,
                                    regressed=improvement < -.002)
            result["plateau_diagnostic"] = plateau
        with (self.output / "progress.jsonl").open("a") as trace:
            trace.write(json.dumps(result) + "\n")
        temporary = self.output / "progress-latest.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.output / "progress-latest.json")
        self.plot()
        return result

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        steps = [row["step"] for row in self.history]
        for axis, name in zip(axes, ("edit", "preservation")):
            for mode, style in (("clean", "-"), ("dv12", "--")):
                values = [row["probes"][name][mode]["frozen_start_D"] for row in self.history]
                axis.plot(steps, values, style, marker=".", label=f"Frozen critic, {mode}")
            axis.set(title=f"{name.capitalize()} training probes", xlabel="Completed updates", ylabel="G game loss (lower)")
            axis.grid(alpha=.25)
            axis.legend()
        temporary = self.output / "progress.tmp.png"
        figure.savefig(temporary, dpi=140)
        temporary.replace(self.output / "progress.png")
        plt.close(figure)
