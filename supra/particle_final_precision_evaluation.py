"""Fixed offline Supra TEST evaluation; never used by native training controls.

The numerical law is the retained B4/CFG3, four-panel CPU72 comparison. The
240 FIT panel draws precede TEST draws, even though FIT is not evaluated here.
"""
from contextlib import contextmanager
import hashlib

import torch
import torch.nn.functional as F

from .particle_game import patchify
from .particle_final_precision import serving_precision_scope

JUDGES = ("D1856", "D6400")
SOURCE_ORDER = (18, 4, 17, 5, 16, 15)


def summarize_records(records):
    def reduce(rows):
        return dict(count=len(rows),
                    rmse_diagnostic=(sum(r["mse_diagnostic"] for r in rows) / len(rows)) ** .5,
                    **{name: sum(r[name] for r in rows) / len(rows) for name in JUDGES})
    return dict(reduce(records), subjects={
        str(s): reduce([r for r in records if r["source_caption_id"] == s])
        for s in sorted({r["source_caption_id"] for r in records})})


@contextmanager
def evaluation_modes(policy):
    modules = {id(m): m for owner in policy._training_modules().values()
               for m in owner.modules()}
    flags = [(m, m.training) for m in modules.values()]
    try:
        for m in modules.values():
            m.training = False
        yield
    finally:
        for m, flag in flags:
            m.training = flag


@torch.no_grad()
def evaluate_test(loop, data, judges, *, precision, panel_sha, budget, ablate=False):
    """Return actual arrays and fixed metrics for one FAST trained checkpoint.

    Both checkpoint arms must be evaluated under common arithmetic before
    attributing a difference to training. Teacher arithmetic remains unchanged.
    """
    p = loop.policy
    context_cpu = data["test"]["context"]
    if context_cpu.shape != (240, 4100) or context_cpu.device.type != "cpu":
        raise ValueError("fixed CPU TEST240x4100 required")
    if context_cpu[:, 4097].long().tolist() != [s for s in SOURCE_ORDER for _ in range(40)]:
        raise ValueError("retained TEST source order differs")
    if set(judges) != set(JUDGES):
        raise ValueError("both fixed judges required")
    stream = torch.Generator().manual_seed(72)
    for _ in range(60):
        torch.randn(4, 4, 256, 16, generator=stream)
    panels, residuals, mses, records = [], [], [], []
    panel_scores = {name: [] for name in JUDGES}
    context_scores = {name: [] for name in JUDGES}
    panel_hash = hashlib.sha256()
    with evaluation_modes(p), serving_precision_scope(loop, precision):
        for start in range(0, 240, 4):
            context = context_cpu[start:start + 4].to(p.device)
            if ablate:
                residual = p.routed_control.spec.forward(
                    p._training_modules(), context, p.routed_control.candidate(),
                    perturb_fn=lambda codes: torch.zeros_like(codes))
            else:
                residual = p.routed_generate(context, sigma=0, perturb=False, averaged=False)
            if residual.shape != (4, 4, 32, 32) or not bool(torch.isfinite(residual).all()):
                raise FloatingPointError("physical B4 residual must be finite")
            condition = p.encoder.condition(context).repeat(4, 1)
            raw_panel = torch.randn(4, 4, 256, 16, generator=stream)
            panel_hash.update(raw_panel.contiguous().numpy().tobytes(order="C"))
            bases = .125 * raw_panel.to(p.device)
            values = {}
            for name in JUDGES:
                judge = judges[name]
                fake = (bases + (patchify(residual).float() / judge.scale).unsqueeze(0)).flatten(0, 1)
                scores = F.softplus(judge(bases.flatten(0, 1), condition)
                                    - judge(fake, condition)).reshape(4, 4)
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("paired judge scores must be finite")
                values[name] = scores.mean(0)
                panel_scores[name].append(scores.cpu())
                context_scores[name].append(values[name].cpu())
            mse = residual.float().square().flatten(1).mean(1)
            for i in range(4):
                records.append(dict(index=start + i, source_caption_id=int(context[i, 4097]),
                                    time=float(context[i, 4096]), mse_diagnostic=float(mse[i]),
                                    **{name: float(values[name][i]) for name in JUDGES}))
            residuals.append(residual.cpu())
            mses.append(mse.cpu())
            panels.append(raw_panel)
            budget()
    if panel_hash.hexdigest() != panel_sha:
        raise ValueError("retained CPU72 TEST panels differ")
    return dict(physical_residual=torch.cat(residuals), mse_by_context=torch.cat(mses),
                paired_game_by_panel={k: torch.cat(v, 1) for k, v in panel_scores.items()},
                paired_game_by_context={k: torch.cat(v) for k, v in context_scores.items()},
                records=records, summary=summarize_records(records),
                gaussian_panels=torch.cat(panels, 1), TEST_panel_sha256=panel_hash.hexdigest(),
                precision=precision, particle_codes_disabled=ablate,
                model_forwards=60, teacher_forwards=60, native_updates=0)
