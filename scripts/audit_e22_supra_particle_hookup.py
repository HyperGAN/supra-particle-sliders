#!/usr/bin/env python3
"""Independent particle wiring and contribution audit, with no training update.

The gradients below are native G-game gradients. Output differences are only
intervention measurements: they never control optimization or row decisions.
``audit_particle_hookup`` can also run on a live loop at a completed boundary.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from particlegan.routing import RoutedExecution
from particlegan.policy import output_noise_std
from supra.particle_adapter import NONLINEAR_V1, LINEAR_MODULATED_V2
from supra.particle_game import patchify
from supra.particle_pilot import checkpoint, restore, state_digest
from supra.particle_training import make_training_loop, raw_model_forward
from supra.runtime import model_module, TARGETS


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


class AuditExecution(RoutedExecution):
    """Use the native mixer, retaining only detached site summaries."""

    def __init__(self, sites, candidate, batch_size, perturb_fn=None):
        super().__init__(sites, candidate, batch_size, perturb_fn)
        self.calls = []

    def mix(self, site, logits):
        codes = super().mix(site, logits)
        with torch.no_grad():
            weights = (logits.to(self._candidate.table.dtype) + self._candidate.log_mass).softmax(-1)
            mean_mass = weights.reshape(-1, weights.shape[-1]).mean(0)
            self.calls.append(dict(site=site, logits_shape=list(logits.shape),
                codes_require_grad=codes.requires_grad,
                mean_max_weight=float(weights.max(-1).values.mean()),
                mean_entropy=float(-(weights * weights.clamp_min(1e-30).log()).sum(-1).mean()),
                most_used_row=int(mean_mass.argmax()),
                most_used_row_mass=float(mean_mass.max())))
        return codes


@contextmanager
def evaluation_modes(policy):
    modules = [module for root in policy._training_modules().values() for module in root.modules()]
    previous = [(module, module.training) for module in modules]
    try:
        for module, _ in previous:
            module.training = False
        yield
    finally:
        for module, flag in previous:
            module.training = flag


def optimizer_ownership(policy):
    expected = {
        "generator": [p for p in policy.G.parameters() if p.requires_grad],
        "router": [p for p in policy.router.parameters() if p.requires_grad],
        "table": [policy.table],
        "critic": [p for p in policy.D.parameters() if p.requires_grad],
    }
    if policy.log_output_sigma is not None:
        expected["noise"] = [policy.log_output_sigma]
    actual = {}
    all_ids = []
    for optimizer, roles in zip(policy.optimizers, policy.roles):
        if len(optimizer.param_groups) != len(roles):
            raise RuntimeError("optimizer groups and policy roles differ")
        for group, role in zip(optimizer.param_groups, roles):
            ids = [id(parameter) for parameter in group["params"]]
            actual.setdefault(role, []).extend(ids)
            all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("a trainable parameter has duplicate optimizer owners")
    if set(actual) != set(expected) or any(
        set(actual[role]) != {id(p) for p in parameters}
        for role, parameters in expected.items()
    ):
        raise RuntimeError("optimizer owners do not exactly match particle game roles")
    return {role: dict(tensors=len(parameters), parameters=sum(p.numel() for p in parameters))
            for role, parameters in expected.items()}


def audit_particle_hookup(loop, context, *, all_site_interventions=False):
    """Audit FAST training weights without touching RNGs, gradients or state.

    The caller chooses the checkpoint and contexts before seeing this receipt.
    A useful particle contribution is measured separately from connectivity.
    A zero-output fresh adapter intentionally has zero bank/router gradients;
    inspect it after two native updates to qualify those backward paths.
    """
    policy = loop.policy
    if policy._phase != "ready":
        raise ValueError("audit requires a completed native update boundary")
    context = context.to(policy.device)
    before = state_digest(checkpoint(loop))
    ownership = optimizer_ownership(policy)
    architecture = policy.G.architecture
    if policy.ema_G.architecture != architecture or any(
        branch.architecture != architecture for branch in policy.G.particle_branches()
    ):
        raise RuntimeError("live/averaged particle architecture attributes differ")
    if tuple(policy.G.sites) != tuple(policy.router.sites) or tuple(policy.G.sites) != tuple(policy.routed_control.spec.sites):
        raise RuntimeError("host, router and structural replay have different site order")
    spec = policy.routed_control.spec
    if spec.output_error_guard or spec.max_context_harm != 0:
        raise RuntimeError("audit requires zero-harm native feature guards without an output-error guard")
    models = policy._training_modules()
    candidate = policy.routed_control.candidate()
    parameters, labels = [], []
    for name, parameter in policy.G.named_parameters():
        if parameter.requires_grad:
            parameters.append(parameter)
            labels.append("generator." + name)
    for name, parameter in policy.router.named_parameters():
        if parameter.requires_grad:
            parameters.append(parameter)
            labels.append("router." + name)
    parameters.append(policy.table)
    labels.append("bank.table")
    if policy.log_output_sigma is not None:
        parameters.append(policy.log_output_sigma)
        labels.append("noise.log_output_sigma")
    saved_gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    pair_stream = torch.Generator(device=policy.device)
    pair_stream.set_state(loop.paired_noise_rng.get_state())
    torch.randn(len(context), 256, 16, generator=pair_stream, device=policy.device)
    base_noise = torch.randn(len(context), 256, 16, generator=pair_stream, device=policy.device)
    private_dv12 = torch.Generator(device=policy.device)
    private_dv12.set_state(policy.noise_generator.get_state())
    prior = policy.controller.routed_prior(candidate.table, candidate.log_mass)
    condition = policy.encoder.condition(context)
    native_loss = policy.recipe.make_loss()
    task_weight = .1 if bool((context[:, 4097] == context[:, 4098]).all()) else 1.

    def forward(current_candidate=candidate, perturb=None, *, raw=False):
        execution = AuditExecution(spec.sites, current_candidate, len(context), perturb)
        try:
            result = (raw_model_forward if raw else spec.model_forward)(models, context, current_candidate, execution)
            usage = execution.finish()
            calls = execution.calls
        finally:
            execution.close()
        if [row["site"] for row in calls] != list(policy.G.sites):
            raise RuntimeError("whole-model candidate did not visit every ordered site exactly once")
        if any(branch.frame is not None for branch in policy.G.particle_branches()) or policy.G._in_forward:
            raise RuntimeError("native host retained a completed forward frame")
        return result, usage, calls

    def game(residual):
        sigma = policy._output_sigma(output_noise_std(policy.recipe, policy.completed_steps), detach=False)
        real = sigma * base_noise
        with torch.no_grad():
            real_logits = policy.D(real.detach(), condition)
        return native_loss.g_loss(policy.D(real + patchify(residual).float()/policy.D.scale, condition), real_logits)

    def gradient(residual):
        payoff = game(residual)
        gradients = torch.autograd.grad(task_weight * payoff, parameters, allow_unused=True)
        norms = {name: 0. if grad is None else float(grad.detach().double().norm())
                 for name, grad in zip(labels, gradients)}
        bank_gradient = gradients[labels.index("bank.table")]
        return dict(game_payoff=float(payoff.detach()), task_weight=task_weight,
            parameter_gradient_norms=norms,
            bank_rows_with_nonzero_gradient=0 if bank_gradient is None else int((bank_gradient.norm(dim=-1)>0).sum()),
            router_tensors_with_nonzero_gradient=sum(norms[name]>0 for name in norms if name.startswith("router.")))

    bridge_records = {}
    handles = []
    for branch in policy.G.particle_branches():
        def capture(module, inputs, value, branch=branch):
            with torch.no_grad(), torch.autocast(policy.device.type, enabled=False):
                joined = inputs[0].float()
                hidden, codes = joined[..., :policy.G.rank], joined[..., policy.G.rank:]
                without_codes = F.linear(hidden, module.weight[:, :policy.G.rank], module.bias)
                modulation = F.linear(value.tanh(), branch.up.weight)
                particle_effect = F.linear(value.tanh()-without_codes.tanh(), branch.up.weight)
                input_path = F.linear(hidden, branch.up.weight) if architecture == LINEAR_MODULATED_V2 else torch.zeros_like(modulation)
                delta = input_path + modulation
                bridge_records[branch.site] = dict(hidden_rms=rms(hidden), code_rms=rms(codes),
                    input_path_output_rms=rms(input_path), modulation_output_rms=rms(modulation),
                    code_effect_at_fixed_input_rms=rms(particle_effect), total_adapter_delta_rms=rms(delta),
                    code_effect_over_total_delta=rms(particle_effect)/max(rms(delta), 1e-30),
                    bridge_saturation_fraction=float((value.abs()>3).float().mean()),
                    tanh_derivative_mean=float((1-value.tanh().square()).mean()))
        handles.append(branch.bridge.register_forward_hook(capture))
    started = time.perf_counter()
    try:
        with evaluation_modes(policy), torch.autograd.set_multithreading_enabled(False):
            clean, usage, calls = forward()
            for handle in handles:
                handle.remove()
            handles.clear()
            clean_detached = clean.detach()
            clean_gradient = gradient(clean)
            with torch.no_grad():
                native = spec.forward(models, context, candidate)
            if not torch.equal(native, clean_detached):
                raise RuntimeError("audit execution differs from the native public clean routed forward")
            noisy, _, noisy_calls = forward(perturb=lambda codes: policy.controller.perturb_latent(
                codes, private_dv12, prior, record=False))
            noisy_gradient = gradient(noisy)
            interventions = {}
            def measure(name, new_candidate=candidate, perturb=None):
                with torch.no_grad():
                    value, _, intervention_calls = forward(new_candidate, perturb)
                    payoff = float(game(value))
                    interventions[name] = dict(output_change_rms=rms(value-clean_detached),
                        game_payoff=payoff,
                        game_payoff_change_from_clean=payoff-clean_gradient["game_payoff"],
                        ordered_site_calls=len(intervention_calls))
            # A code intervention leaves the tied bank keys and every router
            # evaluation live. The rerun propagates changed activations onward.
            measure("zero_particle_codes", perturb=lambda codes: torch.zeros_like(codes))
            measure("translate_entire_bank_by_0_1", replace(candidate, table=candidate.table+.1))
            deleted_row = int(usage.detach().mean(0).argmax())
            mass = candidate.log_mass.detach().clone()
            mass[deleted_row] = -torch.inf
            measure("delete_most_used_row", replace(candidate, log_mass=mass,
                    row_state={**candidate.row_state, "log_mass":mass}))
            interventions["delete_most_used_row"]["row"] = deleted_row
            modulation_handles = [branch.bridge.register_forward_hook(lambda _, __, value: torch.zeros_like(value))
                                  for branch in policy.G.particle_branches()]
            try:
                measure("zero_entire_bounded_modulation")
            finally:
                for handle in modulation_handles:
                    handle.remove()
            site_effects = {}
            indexes = range(len(policy.G.sites)) if all_site_interventions else sorted({0, len(policy.G.sites)//2, len(policy.G.sites)-1})
            for index in indexes:
                count = [0]
                def one_site(codes, index=index):
                    current = count[0]
                    count[0] += 1
                    return codes + .1 if current == index else codes
                name = policy.G.sites[index]
                measure("site:"+name, perturb=one_site)
                site_effects[name] = interventions.pop("site:"+name)
                if count[0] != len(policy.G.sites):
                    raise RuntimeError("per-site perturbation did not consume every site")
            # Raw strength-zero output must equal the source-conditioned pure
            # base, avoiding any cancellation in the residual callback.
            source_context = context.clone()
            source_context[:, 4098] = source_context[:, 4097]
            source_context[:, 4099] = 0
            with torch.no_grad():
                zero_execution = AuditExecution(spec.sites, candidate, len(source_context))
                try:
                    zero_raw = raw_model_forward(models, source_context, candidate, zero_execution)
                    zero_execution.finish()
                finally:
                    zero_execution.close()
                pure_base = policy.encoder.teacher_velocity(source_context)
            if not torch.equal(zero_raw, pure_base):
                raise RuntimeError("strength zero does not exactly preserve the native source base")
            report = dict(completed_steps=policy.completed_steps, architecture=architecture,
                native_game_only=True, optimizer_updates=0, optimizer_ownership=ownership,
                bank_shape=list(policy.table.shape), native_sites=len(policy.G.sites),
                contexts=len(context), cfg=policy.G.cfg, native_clean_forward_exact=True,
                zero_strength_pure_base_exact=True, all_frames_released=True,
                clean_routing_sites=calls, noisy_routing_sites=noisy_calls,
                clean_gradient=clean_gradient, dv12_gradient=noisy_gradient,
                clean_branch_contributions=bridge_records, interventions=interventions,
                per_site_code_interventions=site_effects,
                structural_criterion="clean paired learned critic features; output-error guard disabled; feature harm zero",
                limits="Connectivity and counterfactual dependence do not prove generalization or beneficial particle use. Intervention output RMS is reporting only.")
    finally:
        for handle in handles:
            handle.remove()
    after = state_digest(checkpoint(loop))
    if after != before:
        raise RuntimeError("read-only audit changed full native checkpoint state or RNG")
    if any((old is None) != (parameter.grad is None) or
           (old is not None and not torch.equal(old, parameter.grad))
           for old, parameter in zip(saved_gradients, parameters)):
        raise RuntimeError("read-only audit changed existing parameter gradients")
    report.update(full_checkpoint_and_RNG_unchanged=True, existing_gradients_unchanged=True,
                  seconds=time.perf_counter()-started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-site-interventions", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh audit output")
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint or args.run/"final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run/"data.pt", map_location="cpu", weights_only=False)
    module = model_module()
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict({key.removeprefix("teacher."):value for key,value in
        saved["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}, strict=True, assign=True)
    base.to(args.device).eval().requires_grad_(False)
    loop = make_training_loop(base, data, device=args.device,
        probe_interval=saved["config"]["probe_interval"], branch_lr=saved["config"]["branch_lr"],
        architecture=saved["config"].get("architecture", NONLINEAR_V1))
    restore(loop, saved)
    if state_digest(checkpoint(loop)) != state_digest(saved):
        raise RuntimeError("audit did not exactly restore the chosen checkpoint")
    stream = torch.Generator()
    stream.set_state(saved["data_rng"])
    ids = torch.randint(len(data["fit"]["context"]), (4,), generator=stream)
    report = audit_particle_hookup(loop, data["fit"]["context"][ids],
                                   all_site_interventions=args.all_site_interventions)
    report.update(checkpoint=str(args.checkpoint or args.run/"final.pt"), fit_indices=ids.tolist(),
                  particlegan_source=str(Path(sys.modules["particlegan"].__file__).resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(event="particle_hookup_audit_complete", output=str(args.output),
        completed_steps=report["completed_steps"], architecture=report["architecture"],
        clean_bank_rows=report["clean_gradient"]["bank_rows_with_nonzero_gradient"],
        noisy_bank_rows=report["dv12_gradient"]["bank_rows_with_nonzero_gradient"],
        particles_removed_output_change_rms=report["interventions"]["zero_particle_codes"]["output_change_rms"],
        seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
