#!/usr/bin/env python3
"""Read-only parity of two qualified, source/time-conditioned Supra critics.

Inspect the zero-residual paired game, including detached real references.
Supra's actual G estimator uses one Gaussian; antithetic/evenized scores here
are observational decompositions, not training proposals or quality winners.
No generator, optimizer, controller or KA2 update is constructed or executed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path("/ml2/hypergan/supra-e22-verification/outputs")
SIGMA, PANELS, TOKENS, BATCH = .125, 4, 256, 4


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HISTORY / "e22-particle-v2-6400")
    parser.add_argument("--earlier-run", type=Path, default=HISTORY / "e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path,
                        default=HISTORY / "e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-frozen-critic-parity")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence; choose a fresh output directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from particlegan.gan_loss import GANLoss
    from supra.particle_game import ConditionalTokenCritic

    torch.set_num_threads(1)
    start = time.monotonic()
    cpu_rng = torch.get_rng_state().clone()

    def digest(value):
        result = hashlib.sha256()

        def visit(item):
            if isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                result.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    result.update(str(key).encode())
                    visit(item[key])
            elif isinstance(item, (list, tuple)):
                result.update(type(item).__name__.encode())
                for child in item:
                    visit(child)
            else:
                result.update(repr(item).encode())
        visit(value)
        return result.hexdigest()

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    pg_dir = Path(particlegan.__file__).resolve().parent
    if pg_dir.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from the declared historical source")
    pg_hashes = {path.name: sha(path) for path in sorted(pg_dir.glob("*.py"))}
    declared = json.loads((args.run / "run.json").read_text())
    if digest(pg_hashes) != declared["particlegan_source_digest"]:
        raise RuntimeError("native game source differs from the qualified critic cohort")
    inputs = {"D6400": args.run / "final.pt", "D1856": args.earlier_run / "final.pt",
              "data": args.run / "data.pt", "qualification": args.run / "qualification-review.json",
              "plan": args.run / "plan.json", "run": args.run / "run.json",
              "earlier_receipt": args.earlier_run / "receipt.json"}
    hashes = {name: sha(path) for name, path in inputs.items()}
    qualification = json.loads(inputs["qualification"].read_text())
    plan = json.loads(inputs["plan"].read_text())
    earlier_receipt = json.loads(inputs["earlier_receipt"].read_text())
    if (not qualification["qualified"]
            or hashes["D6400"] != qualification["artifact_sha256"]["final.pt"]
            or hashes["data"] != qualification["artifact_sha256"]["data.pt"]
            or hashes["D1856"] != plan["input_sha256"]["final.pt"]
            or hashes["D1856"] != earlier_receipt["final_checkpoint_sha256"]):
        raise RuntimeError("inputs do not match the qualified two-critic chain")
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    saved = {name: torch.load(inputs[name], map_location="cpu", weights_only=False, mmap=True)
             for name in ("D1856", "D6400")}
    dataset_digest = digest(data)
    native_streams = {}
    for name, state in saved.items():
        if (state["policy"]["completed_steps"] != int(name[1:])
                or state["config"]["dataset_digest"] != dataset_digest
                or state["config"]["architecture"] != "linear_modulated_v2"
                or float(state["policy"]["last_output_sigma"]) != SIGMA):
            raise RuntimeError("saved critic belongs to a different native law/cohort")
        encoder = state["policy"]["models"]["encoder"]
        if (not torch.equal(encoder["contexts"], data["text_contexts"])
                or not torch.equal(encoder["masks"], data["text_masks"])):
            raise RuntimeError("critic source conditioning differs from checkpoint-owned text")
        native_streams[name] = digest({"native": state["policy"]["streams"],
                                       "data": state["data_rng"], "paired": state["paired_noise_rng"]})
    critic_before = {name: digest(state["policy"]["models"]["critic"]) for name, state in saved.items()}
    if (critic_before["D1856"] != qualification["common_start_critic_digest"]
            or critic_before["D6400"] != qualification["final_critic_digest"]):
        raise RuntimeError("actual common-judge tensor digests differ from qualification")
    data_before = digest(data)

    # All actual cached fitting source/time pairs; no latent or score selection.
    pairs = sorted({(int(row[4097]), float(row[4096])) for row in data["fit"]["context"]})
    sources = sorted({source for source, _ in pairs})
    times = sorted({t for _, t in pairs})
    if len(sources) != 6 or len(times) != 10 or len(pairs) != 60:
        raise RuntimeError("expected the complete six-source, ten-time native conditioning grid")
    ids = torch.tensor([source for source, _ in pairs])
    text, mask = data["text_contexts"][ids].float(), data["text_masks"][ids].float()
    pooled = (text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    cond = torch.cat((pooled, torch.tensor([t for _, t in pairs]).unsqueeze(1)), dim=-1)
    stream = torch.Generator().manual_seed(72)
    panels = SIGMA * torch.randn((PANELS, len(pairs), TOKENS, 16), generator=stream)
    repeated = cond.repeat(PANELS, 1)
    source_files = (Path(__file__), ROOT / "supra/particle_game.py", pg_dir / "gan_loss.py")
    source_hashes = {str(path): sha(path) for path in source_files}
    args.output.mkdir(parents=True)
    declaration = dict(schema="supra_frozen_critic_parity_v1", critics=list(saved),
        conditioning="All six source captions and all ten cached fit times; fixed mask-pooled text and time.",
        condition_pairs=pairs, private_gaussian="CPU72, four panels shared by both critics", sigma=SIGMA,
        normalized_error_shape=[60, TOKENS, 16], native_generator_estimator="single Gaussian per G pass",
        units="Per-context game gradient before batch averaging; physical velocity derivative divides by coordinate_scale and native B4.",
        native_KA2_units="The unchanged token penalty view repeats contextual scores, yielding 16-coordinate units; it is not called here.",
        game="Native RpGAN with detached real reference; zero residual always gives log2, not necessarily zero G gradient.",
        interventions="Evenized and antithetic games are read-only counterfactuals, never optimizer/guard/selection changes.",
        updates=0, generator_instantiated=False, output_metrics=False, input_sha256=hashes,
        source_sha256=source_hashes, critic_tensor_digests=critic_before,
        dataset_digest=dataset_digest, condition_digest=digest(cond), private_panel_digest=digest(panels),
        scope="Zero residual at two frozen historical judges. Nonzero force is possible critic lag, not evidence of the current G residual, Hessian, quality improvement or broken equilibrium.")
    write(args.output / "plan.json", declaration)
    game = GANLoss()
    results, raw = {}, {}

    def rms(tensor):
        return tensor.detach().double().square().flatten(1).mean(1).sqrt()

    def summary(tensor):
        values = tensor.detach().double().flatten()
        return dict(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()))

    for name, state in saved.items():
        with torch.random.fork_rng(devices=[]):
            critic = ConditionalTokenCritic(data["coordinate_scale"]).float().eval().requires_grad_(False)
        critic.load_state_dict(state["policy"]["models"]["critic"], strict=True)
        if digest(critic.state_dict()) != critic_before[name]:
            raise RuntimeError("copied critic differs from the actual frozen judge")
        before = digest(critic.state_dict())
        positive = panels.flatten(0, 1).detach().requires_grad_(True)
        negative = (-panels).flatten(0, 1).detach().requires_grad_(True)
        d_plus, d_minus = critic(positive, repeated), critic(negative, repeated)
        grad_plus = torch.autograd.grad(d_plus.sum(), positive)[0].reshape_as(panels)
        grad_minus = torch.autograd.grad(d_minus.sum(), negative)[0].reshape_as(panels)
        odd = .5 * (grad_plus + grad_minus)
        even = .5 * (grad_plus - grad_minus)
        expected = {"native_single": -.5 * grad_plus.mean(0),
                    "native_antithetic": -.5 * odd.mean(0),
                    "evenized_single": -.5 * even.mean(0),
                    "evenized_antithetic": torch.zeros_like(grad_plus[0])}

        def loss_gradient(evenized=False, antithetic=False):
            residual = torch.zeros_like(panels[0], requires_grad=True)
            halves = (panels, -panels) if antithetic else (panels,)
            values = []
            for noise in halves:
                real, fake = noise.flatten(0, 1), (noise + residual.unsqueeze(0)).flatten(0, 1)
                if evenized:
                    ref = .5 * (critic(real, repeated) + critic(-real, repeated))
                    score = .5 * (critic(fake, repeated) + critic(-fake, repeated))
                else:
                    ref, score = critic(real, repeated), critic(fake, repeated)
                values.append(game.g_loss(score, ref.detach()))
            value = torch.stack(values).mean()
            # Undo only context batch averaging; retain four-panel mean.
            gradient = torch.autograd.grad(value, residual)[0] * len(pairs)
            torch.testing.assert_close(value, value.new_tensor(math.log(2)), rtol=0, atol=1e-7)
            return float(value.detach()), gradient

        metrics = {}
        for label, flags in {"native_single": (False, False), "native_antithetic": (False, True),
                             "evenized_single": (True, False), "evenized_antithetic": (True, True)}.items():
            value, gradient = loss_gradient(*flags)
            torch.testing.assert_close(gradient, expected[label], rtol=2e-5, atol=2e-8)
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("nonfinite diagnostic game gradient")
            metrics[label] = dict(zero_residual_game=value, normalized_patch_gradient_rms=summary(rms(gradient)),
                physical_velocity_gradient_rms_native_B4=summary(rms(gradient / data["coordinate_scale"] / BATCH)),
                per_condition_normalized_patch_gradient_rms=rms(gradient).tolist())
        torch.testing.assert_close(expected["native_single"],
                                   expected["native_antithetic"] + expected["evenized_single"], rtol=2e-5, atol=2e-8)
        zero = torch.zeros_like(panels[0], requires_grad=True)
        zero_score = critic(zero, cond)
        zero_force = -.5 * torch.autograd.grad(zero_score.sum(), zero)[0]
        metrics["zero_gaussian_native_force_rms"] = summary(rms(zero_force))
        metrics["odd_gradient_panel_rms"] = summary(rms(odd.flatten(0, 1)))
        metrics["even_gradient_panel_rms"] = summary(rms(even.flatten(0, 1)))
        if (digest(critic.state_dict()) != before or any(parameter.grad is not None or parameter.requires_grad
                                                       for parameter in critic.parameters())):
            raise RuntimeError("read-only differentiation changed critic state or gradient ownership")
        results[name] = metrics
        raw[name] = {"gradient_plus": grad_plus, "gradient_minus": grad_minus, "zero_residual_forces": expected,
                     "zero_gaussian_force": zero_force.detach(), "score_plus": d_plus.detach(), "score_minus": d_minus.detach()}
        print(json.dumps({"critic": name, "normalized_force_rms":
                         {key: metrics[key]["normalized_patch_gradient_rms"]["mean"] for key in expected}}, allow_nan=False), flush=True)

    if not torch.equal(cpu_rng, torch.get_rng_state()) or digest(data) != data_before:
        raise RuntimeError("diagnostic changed global RNG or cached data")
    for name, state in saved.items():
        after_streams = digest({"native": state["policy"]["streams"], "data": state["data_rng"],
                                "paired": state["paired_noise_rng"]})
        if after_streams != native_streams[name] or digest(state["policy"]["models"]["critic"]) != critic_before[name]:
            raise RuntimeError("diagnostic changed a checkpoint-owned critic or RNG stream")
    if {name: sha(path) for name, path in inputs.items()} != hashes or {str(path): sha(path) for path in source_files} != source_hashes:
        raise RuntimeError("read-only diagnostic source or original input artifact changed")
    torch.save(dict(condition=cond, condition_pairs=pairs, panels=panels, critics=raw), args.output / "gradients.pt")
    write(args.output / "report.json", dict(plan=declaration, results=results, qualified=True,
        global_cpu_rng_unchanged=True, native_streams_unchanged=True, critic_states_flags_gradients_unchanged=True,
        source_and_inputs_unchanged=True, gradient_artifact_sha256=sha(args.output / "gradients.pt"),
        wall_seconds=time.monotonic() - start, training_updates=0, qualification_credit="none"))


if __name__ == "__main__":
    main()
