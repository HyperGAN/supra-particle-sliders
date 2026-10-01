#!/usr/bin/env python3
"""Observe token geometry of two qualified historical Supra critics.

No Supra generator, optimizer, policy, or training update is constructed.
The fixed noise law is exactly the earlier qualified CPU72 parity panel.
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
PARITY = ROOT / "outputs/e22-frozen-critic-parity"
SIGMA, DRAWS, CONTEXTS, TOKENS, WIDTH, NATIVE_BATCH = .125, 4, 60, 256, 16, 4


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-frozen-critic-token-force-v1")
    parser.add_argument("--compact-report", type=Path, default=ROOT / "docs/e22_supra_frozen_critic_token_force_results.json")
    args = parser.parse_args()
    if args.output.exists() or args.compact_report.exists():
        parser.error("preserve existing evidence; choose fresh output and compact-report paths")
    native_root = HISTORY / "e22-convergence-gap/particlegan-cabe2084-source"
    sys.path.insert(0, str(native_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from particlegan.gan_loss import GANLoss
    from supra.particle_game import ConditionalTokenCritic

    torch.set_num_threads(1)
    started = time.monotonic()
    global_rng = torch.get_rng_state().clone()

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

    def budget():
        if time.monotonic() - started > 120:
            raise TimeoutError("fixed 120-second observational budget exceeded")

    pg_dir = Path(particlegan.__file__).resolve().parent
    if pg_dir.parent != native_root:
        raise RuntimeError("native import differs from the qualified historical source")
    parity_report_path = PARITY / "report.json"
    parity = json.loads(parity_report_path.read_text())
    if not parity["qualified"] or parity["training_updates"] != 0:
        raise RuntimeError("parent parity evidence is not qualified and observational")
    parent = parity["plan"]
    inputs = {
        "D6400": HISTORY / "e22-particle-v2-6400/final.pt",
        "D1856": HISTORY / "e22-particle-noise-update-256/latest/final.pt",
        "data": HISTORY / "e22-particle-v2-6400/data.pt",
        "qualification": HISTORY / "e22-particle-v2-6400/qualification-review.json",
        "plan": HISTORY / "e22-particle-v2-6400/plan.json",
        "run": HISTORY / "e22-particle-v2-6400/run.json",
        "earlier_receipt": HISTORY / "e22-particle-noise-update-256/latest/receipt.json",
        "parity_report": parity_report_path,
        "parity_gradients": PARITY / "gradients.pt",
    }
    hashes = {name: sha(path) for name, path in inputs.items()}
    if any(hashes[name] != expected for name, expected in parent["input_sha256"].items()):
        raise RuntimeError("original inputs differ from the qualified parity chain")
    if hashes["parity_gradients"] != parity["gradient_artifact_sha256"]:
        raise RuntimeError("parent gradient artifact differs from its receipt")
    native_hashes = {path.name: sha(path) for path in sorted(pg_dir.glob("*.py"))}
    declared = json.loads(inputs["run"].read_text())
    if digest(native_hashes) != declared["particlegan_source_digest"]:
        raise RuntimeError("native source differs from the qualified critic cohort")
    source_files = [Path(__file__), ROOT / "supra/particle_game.py",
                    ROOT / "scripts/diagnose_e22_supra_critic_parity.py", pg_dir / "gan_loss.py"]
    source_hashes = {str(path): sha(path) for path in source_files}
    if any(source_hashes[path] != expected for path, expected in parent["source_sha256"].items()):
        raise RuntimeError("held critic, game, or parent probe source changed")
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    saved = {name: torch.load(inputs[name], map_location="cpu", weights_only=False, mmap=True)
             for name in ("D1856", "D6400")}
    artifact = torch.load(inputs["parity_gradients"], map_location="cpu", weights_only=False)
    before_data, before_artifact = digest(data), digest(artifact)
    if before_data != parent["dataset_digest"]:
        raise RuntimeError("dataset tensor identity differs from the qualified cohort")
    pairs = sorted({(int(row[4097]), float(row[4096])) for row in data["fit"]["context"]})
    if len(pairs) != CONTEXTS or pairs != [tuple(pair) for pair in parent["condition_pairs"]]:
        raise RuntimeError("expected all original six-source, ten-time fitting pairs")
    ids = torch.tensor([source for source, _ in pairs])
    text, mask = data["text_contexts"][ids].float(), data["text_masks"][ids].float()
    pooled = (text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    condition = torch.cat((pooled, torch.tensor([t for _, t in pairs]).unsqueeze(1)), dim=-1)
    stream = torch.Generator(device="cpu").manual_seed(72)
    panels = SIGMA * torch.randn((DRAWS, CONTEXTS, TOKENS, WIDTH), generator=stream)
    if (digest(condition) != parent["condition_digest"] or digest(panels) != parent["private_panel_digest"]
            or not torch.equal(condition, artifact["condition"]) or not torch.equal(panels, artifact["panels"])):
        raise RuntimeError("private conditioning/noise law differs from the qualified parity panel")
    original_native = {}
    original_critics = {}
    for name, state in saved.items():
        if (state["policy"]["completed_steps"] != int(name[1:])
                or state["config"]["dataset_digest"] != before_data
                or state["config"]["architecture"] != "linear_modulated_v2"
                or float(state["policy"]["last_output_sigma"]) != SIGMA):
            raise RuntimeError("critic belongs to a different law/cohort")
        encoder = state["policy"]["models"]["encoder"]
        if not torch.equal(encoder["contexts"], data["text_contexts"]) or not torch.equal(encoder["masks"], data["text_masks"]):
            raise RuntimeError("checkpoint-owned source conditioning changed")
        original_native[name] = digest({"native": state["policy"]["streams"], "data": state["data_rng"],
                                        "paired": state["paired_noise_rng"]})
        original_critics[name] = digest(state["policy"]["models"]["critic"])
    if original_critics != parent["critic_tensor_digests"]:
        raise RuntimeError("actual judge tensors differ from qualification")

    declaration = {
        "schema": "supra_frozen_critic_token_force_v1", "time_budget_seconds": 120,
        "critics": ["D1856", "D6400"], "input_sha256": hashes, "source_sha256": source_hashes,
        "critic_tensor_digests": original_critics, "dataset_digest": before_data,
        "condition_digest": digest(condition), "private_panel_digest": digest(panels),
        "condition_pairs": pairs, "condition_pool": "All six source captions and ten cached fitting times; no context selection.",
        "private_noise_law": "Exact parent CPU72, four FP32 panels, sigma.125, shape[4,60,256,16]. No new draws, seed/noise/amplitude scan.",
        "gradient_law": "Native paired RpGAN at zero residual with detached real scores. Undo context averaging; report each single draw and their fixed four-draw mean.",
        "projection": "Per context and coordinate: mean force over256 tokens, broadcast back; remainder has zero token mean. Report FP64 squared norm fractions and orthogonality.",
        "units": "Primary normalized critic input; secondary physical velocity derivative divides coordinate_scale and native B4, without changing token geometry.",
        "permutation_check": "One fixed reverse-token permutation jointly reindexes the identical real/fake Gaussian input. Check feature/force equivariance and pooled-score invariance.",
        "zero_noise_check": "At zero Gaussian and zero residual, every token force must be exactly identical, and its global squared-norm fraction must be one.",
        "limits": [
            "The critic has shared per-token features and global source/time conditioning, with no spatial positions. Pooled scores are mathematically invariant to joint token permutations.",
            "Permuting residual alone while holding a finite Gaussian panel fixed need not preserve its score/gradient; exchangeability holds in expectation under the IID noise law.",
            "Zero-noise identical-token force and finite-panel global fractions do not establish a wrong sign, convergence cause, current generator residual, or a quality improvement.",
            "Nonlinear token features can respond to nonzero token variance; permutation invariance does not make all zero-mean residuals invisible.",
            "The structural row guard flattens per-token features and is a separate mechanism from this pooled scalar generator game.",
        ],
        "training_updates": 0, "generator_instantiated": False, "full_supra_forward_calls": 0,
        "optimizer_policy_KA2_calls": 0, "observational_critic_forwards": True, "output_metrics": False,
    }
    args.output.mkdir(parents=True)
    write(args.output / "plan.json", declaration)  # Freeze before any critic forward.
    budget()
    game = GANLoss()
    results = {}

    def stats(tensor):
        values = tensor.detach().double().flatten()
        return {"count": values.numel(), "mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max())}

    def decomposition(force):
        value = force.detach().double()
        mean = value.mean(dim=-2, keepdim=True)
        global_component = mean.expand_as(value)
        centered = value - global_component
        total = value.square().sum(dim=(-2, -1))
        global_energy = global_component.square().sum(dim=(-2, -1))
        centered_energy = centered.square().sum(dim=(-2, -1))
        cross = (global_component * centered).sum(dim=(-2, -1))
        if not bool((total > 0).all()) or not torch.isfinite(value).all():
            raise RuntimeError("zero or nonfinite native force makes the declared fractions undefined")
        torch.testing.assert_close(global_energy + centered_energy, total, rtol=2e-12, atol=1e-20)
        return {"global_squared_force_fraction": stats(global_energy / total),
                "energy_weighted_global_fraction": float(global_energy.sum() / total.sum()),
                "force_rms": stats((total / (TOKENS * WIDTH)).sqrt()),
                "global_force_rms": stats((global_energy / (TOKENS * WIDTH)).sqrt()),
                "centered_force_rms": stats((centered_energy / (TOKENS * WIDTH)).sqrt()),
                "maximum_absolute_orthogonal_dot": float(cross.abs().max()),
                "maximum_absolute_centered_token_mean": float(centered.mean(dim=-2).abs().max())}

    for name, state in saved.items():
        budget()
        with torch.random.fork_rng(devices=[]):
            critic = ConditionalTokenCritic(data["coordinate_scale"]).float().eval().requires_grad_(False)
        critic.load_state_dict(state["policy"]["models"]["critic"], strict=True)
        if set(critic._modules) != {"error_input", "condition_input", "feature_output", "score"} or set(critic._buffers) != {"scale"}:
            raise RuntimeError("critic topology differs from the shared-token pooled architecture")
        before = digest(critic.state_dict())
        if before != original_critics[name]:
            raise RuntimeError("copied judge is not bitwise identical")
        flags = [(module.training, tuple(p.requires_grad for p in module.parameters(recurse=False))) for module in critic.modules()]
        forward_calls = 0

        def scores(error, cond):
            nonlocal forward_calls
            forward_calls += 1
            return critic(error, cond)

        def force(noise, cond):
            residual = torch.zeros_like(noise, requires_grad=True)
            with torch.no_grad():
                real = scores(noise, cond)
            fake = scores(noise + residual, cond)
            loss = game.g_loss(fake, real.detach())
            torch.testing.assert_close(loss, loss.new_tensor(math.log(2)), rtol=0, atol=1e-7)
            return torch.autograd.grad(loss, residual)[0] * len(noise)

        noise, repeated = panels.flatten(0, 1), condition.repeat(DRAWS, 1)
        finite_force = force(noise, repeated).reshape_as(panels)
        torch.testing.assert_close(finite_force, -.5 * artifact["critics"][name]["gradient_plus"], rtol=2e-5, atol=2e-8)
        averaged_force = finite_force.mean(0)
        torch.testing.assert_close(averaged_force, artifact["critics"][name]["zero_residual_forces"]["native_single"], rtol=2e-5, atol=2e-8)
        zero_force = force(torch.zeros_like(panels[0]), condition)
        torch.testing.assert_close(zero_force, artifact["critics"][name]["zero_gaussian_force"], rtol=2e-5, atol=2e-8)
        if not torch.equal(zero_force, zero_force[:, :1].expand_as(zero_force)):
            raise AssertionError("zero-noise force is not identical at every token")
        permuted_force = force(noise.flip(1), repeated)
        torch.testing.assert_close(permuted_force, finite_force.flatten(0, 1).flip(1), rtol=2e-5, atol=2e-8)
        with torch.no_grad():
            original_score, permuted_score = scores(noise, repeated), scores(noise.flip(1), repeated)
            original_features, permuted_features = critic.features(noise, repeated), critic.features(noise.flip(1), repeated)
        torch.testing.assert_close(permuted_score, original_score, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(permuted_features, original_features.flip(1), rtol=0, atol=0)
        metrics = {"normalized_single_draw_native_force": decomposition(finite_force),
                   "normalized_four_draw_mean_native_force": decomposition(averaged_force),
                   "normalized_zero_gaussian_native_force": decomposition(zero_force),
                   "physical_single_draw_native_B4": decomposition(finite_force / critic.scale / NATIVE_BATCH),
                   "physical_four_draw_mean_native_B4": decomposition(averaged_force / critic.scale / NATIVE_BATCH),
                   "zero_noise_all_tokens_bitwise_equal": True,
                   "reverse_token_feature_equivariance_bitwise": True,
                   "reverse_token_score_max_absolute_difference": float((permuted_score - original_score).abs().max()),
                   "reverse_token_force_max_absolute_difference": float((permuted_force - finite_force.flatten(0, 1).flip(1)).abs().max()),
                   "observational_critic_forward_calls": forward_calls,
                   "observational_critic_feature_calls": 2}
        if metrics["normalized_zero_gaussian_native_force"]["energy_weighted_global_fraction"] != 1.:
            raise AssertionError("zero-noise global force fraction differs from one")
        if (digest(critic.state_dict()) != before
                or flags != [(m.training, tuple(p.requires_grad for p in m.parameters(recurse=False))) for m in critic.modules()]
                or any(p.grad is not None for p in critic.parameters())):
            raise RuntimeError("observational differentiation changed critic state, modes, flags, or gradients")
        results[name] = metrics
        print(json.dumps({"critic": name, "single_draw_global_fraction": metrics["normalized_single_draw_native_force"]["energy_weighted_global_fraction"],
                          "four_draw_mean_global_fraction": metrics["normalized_four_draw_mean_native_force"]["energy_weighted_global_fraction"],
                          "zero_noise_global_fraction": 1.}, allow_nan=False), flush=True)

    if (not torch.equal(global_rng, torch.get_rng_state()) or torch.cuda.is_initialized()
            or digest(data) != before_data or digest(artifact) != before_artifact):
        raise RuntimeError("probe changed global RNG, data/parent tensors, or initialized CUDA")
    for name, state in saved.items():
        if (digest(state["policy"]["models"]["critic"]) != original_critics[name]
                or original_native[name] != digest({"native": state["policy"]["streams"], "data": state["data_rng"], "paired": state["paired_noise_rng"]})):
            raise RuntimeError("probe changed original checkpoint-owned critics or streams")
    if hashes != {name: sha(path) for name, path in inputs.items()} or source_hashes != {str(path): sha(path) for path in source_files}:
        raise RuntimeError("probe changed protected inputs or sources")
    if native_hashes != {path.name: sha(path) for path in sorted(pg_dir.glob("*.py"))}:
        raise RuntimeError("native package source changed during the probe")
    budget()
    report = {"plan": declaration, "results": results, "qualified": True, "qualification_credit": "none",
              "immutability": {"global_cpu_rng_unchanged": True, "CUDA_not_initialized": True,
                               "original_critic_states_and_streams_unchanged": True, "copied_critic_state_modes_flags_grads_unchanged": True,
                               "dataset_and_parent_gradient_tensors_unchanged": True, "original_inputs_and_sources_unchanged": True},
              "training_updates": 0, "full_supra_forward_calls": 0, "wall_seconds": time.monotonic() - started}
    write(args.output / "report.json", report)
    compact = {"schema": declaration["schema"], "scope": "Qualified observational saved-critic token-force geometry; no training or convergence-cause claim.",
               "report_path": str(args.output / "report.json"), "report_sha256": sha(args.output / "report.json"),
               "plan_sha256": sha(args.output / "plan.json"), "bindings": {k: declaration[k] for k in
                 ("input_sha256", "source_sha256", "critic_tensor_digests", "dataset_digest", "condition_digest", "private_panel_digest")},
               "results": results, "immutability": report["immutability"], "wall_seconds": report["wall_seconds"],
               "training_updates": 0, "full_supra_forward_calls": 0, "limits": declaration["limits"]}
    write(args.compact_report, compact)


if __name__ == "__main__":
    main()
