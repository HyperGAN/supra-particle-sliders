#!/usr/bin/env python3
"""Fixed saved-FAST CFG game-gradient witness; no optimization or intervention.

Observe the unchanged public routed forward. Separate the conditional and
unconditional VJPs of its actual learned-game upstream gradient. All owners are
isolated copies of saved FAST tensors; native policy/optimizer/diagnostic objects
are neither constructed nor called. This is a local conditioning diagnostic.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_supra_cfg_gradient_diagnostic_v1.json"
NORM_FLOOR = 1e-12
DECOMPOSITION_RTOL = 2e-4
DECOMPOSITION_ATOL = 1e-12
REVIEWER_SOURCES = ("scripts/review_e22_supra_neutral_initialization.py",
                    "scripts/review_e22_supra_pr223.py")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_budget(started, *, limit=900, clock=time.monotonic):
    elapsed = clock() - started
    if elapsed > limit:
        raise TimeoutError("fixed posthoc 900-second budget exhausted; incomplete diagnostic")
    return elapsed


def final_report(output, report, started, *, clock=time.monotonic, writer=write):
    """Completion also requires the deadline after report/hash/receipt writes."""
    output = Path(output)
    report = {**report, "completion_receipt_required": True,
              "seconds_before_report_serialization": clock() - started}
    try:
        writer(output / "report.json", report)
        check_budget(started, clock=clock)
        digest = sha(output / "report.json")
        check_budget(started, clock=clock)
        writer(output / "completion.json", dict(complete=True, report_sha256=digest,
            wall_seconds=clock() - started, quality_qualification_credit="none",
            clock_scope="After report serialization/write/SHA; deadline checked again after this receipt write."))
        return check_budget(started, clock=clock)
    except TimeoutError:
        report.update(complete=False, budget_overrun=True, seconds_at_overrun=clock() - started)
        writer(output / "report.json", report)
        writer(output / "completion.json", dict(complete=False, budget_overrun=True,
            report_sha256=sha(output / "report.json"), wall_seconds=clock() - started,
            quality_qualification_credit="none"))
        raise


def raw_sha(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def qualified_review_identity(card, review_path, expected_sha256, *, root=ROOT):
    """Bind the frozen final review and its source before any model/CUDA load."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("--qualified-review-sha256 must freeze the final review SHA256")
    contents = Path(review_path).read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError("final independent review does not match frozen SHA256")
    review = json.loads(contents)
    if (review.get("schema") != "supra_neutral_initialization_independent_review_v1"
            or review.get("qualified") is not True or review.get("partial") is not False):
        raise ValueError("qualified nonpartial independent final review required")
    sources = card["reviewer_sources_sha256"]
    if set(sources) != set(REVIEWER_SOURCES):
        raise ValueError("expected fixed reviewer and dependency source identities")
    main, helper = REVIEWER_SOURCES
    expected_dependencies = {str((Path(root) / helper).resolve()): sources[helper]}
    if (review.get("reviewer_sha256") != sources[main]
            or review.get("reviewer_dependency_sha256") != expected_dependencies):
        raise ValueError("final review's reviewer/dependency identity changed")
    if any(sha(Path(root) / name) != digest for name, digest in sources.items()):
        raise ValueError("actual independent reviewer source identity changed")
    return review


def check_checkpoint_contract(state, declared_config, step):
    """Compare the unchanged saved config in the plan's JSON representation."""
    actual_config = json.loads(json.dumps(state["config"], allow_nan=False))
    if state["policy"]["completed_steps"] != step or actual_config != declared_config:
        raise ValueError("actual saved checkpoint clock/config mismatch")


def select_fit_indices(context):
    """Private CPU7, two contexts per fixed source; no data/metric search."""
    import torch
    stream = torch.Generator(device="cpu").manual_seed(7)
    indices = []
    for source in (4, 5, 15, 16, 17, 18):
        rows = (context[:, 4097].long() == source).nonzero().flatten()
        if len(rows) < 2:
            raise ValueError("missing fixed source contexts")
        indices.extend(rows[torch.randperm(len(rows), generator=stream)[:2]].sort().values.tolist())
    return indices, raw_sha(stream.get_state())


def private_panels():
    import torch
    return torch.randn((4, 12, 256, 16), generator=torch.Generator(device="cpu").manual_seed(72))


def geometry(conditional, unconditional, guided):
    """Aggregate named VJPs with double precision reductions, retaining signs."""
    import torch
    if not guided or set(conditional) != set(guided) or set(unconditional) != set(guided):
        raise ValueError("gradient owners must be nonempty and identical")
    c, u, g = (torch.cat([family[name].detach().reshape(-1).double() for name in guided])
               for family in (conditional, unconditional, guided))
    summed, difference = c + u, c + u - g
    # One scalar transfer per group, instead of synchronizing for every
    # parameter/reduction in the 71-site, per-context witness.
    cc, uu, cu, gg, ss, ee, largest, finite = torch.stack((c.square().sum(),
        u.square().sum(), (c*u).sum(), g.square().sum(), summed.square().sum(),
        difference.square().sum(), difference.abs().max(),
        (torch.isfinite(c).all() & torch.isfinite(u).all() & torch.isfinite(g).all()).double())).cpu().tolist()
    if not finite:
        raise FloatingPointError("nonfinite group gradient")
    coordinates = g.numel()
    cn, un, gn, sn = map(math.sqrt, (cc, uu, gg, ss))
    error_norm = math.sqrt(ee)
    tolerance = DECOMPOSITION_ATOL + DECOMPOSITION_RTOL * (cn + un)
    verified = error_norm <= tolerance
    return dict(coordinates=coordinates, conditional_gradient_norm=cn,
                unconditional_gradient_norm=un, guided_gradient_norm=gn,
                summed_half_gradient_norm=sn,
                ratio_norm_floor=NORM_FLOOR,
                half_cosine=None if min(cn, un) <= NORM_FLOOR else max(-1.0, min(1.0, cu / (cn * un))),
                cancellation_ratio=None if cn + un <= NORM_FLOOR else sn / (cn + un),
                half_sum_cancellation_ratio=None if cn + un <= NORM_FLOOR else sn / (cn + un),
                guided_to_half_norm_ratio=None if cn + un <= NORM_FLOOR else gn / (cn + un),
                half_sum_cancellation_faithful_to_guided=verified,
                guided_to_conditional_norm=None if cn <= NORM_FLOOR else gn / cn,
                decomposition_error_norm=error_norm,
                decomposition_relative_error=None if gn <= NORM_FLOOR else error_norm / gn,
                decomposition_half_norm_relative_error=None if cn + un <= NORM_FLOOR else error_norm / (cn + un),
                decomposition_absolute_tolerance=tolerance,
                decomposition_within_tolerance=verified,
                decomposition_verified=verified,
                decomposition_interpretation=("half-sum reconstruction matches guided VJP within fixed numerical bound"
                    if verified else "half-sum cancellation unverified; not a faithful decomposition of the actual guided VJP"),
                decomposition_max_absolute_error=largest)


def verify_decomposition(result):
    """Software algebra assertion; actual observations retain unverified cases."""
    tolerance = DECOMPOSITION_ATOL + DECOMPOSITION_RTOL * (
        result["conditional_gradient_norm"] + result["unconditional_gradient_norm"])
    if result["decomposition_error_norm"] > tolerance:
        raise AssertionError("half VJPs fail fixed half-norm-scaled numerical tolerance")


def half_vjps(conditional, unconditional, guided, upstream, parameters, *, cfg=3.0):
    """Actual graph VJPs; no .backward(), .grad writes, detach surrogate or step."""
    import torch
    if cfg != 3.0 or conditional.shape != unconditional.shape or guided.shape != upstream.shape:
        raise ValueError("this frozen diagnostic requires matching CFG3 halves")
    if not torch.equal(unconditional + cfg * (conditional - unconditional), guided):
        raise AssertionError("observed halves do not recreate the actual guided forward")
    names, owners = zip(*parameters.items())
    values = []
    for output, force in ((conditional, cfg * upstream),
                          (unconditional, (1.0 - cfg) * upstream), (guided, upstream)):
        grads = torch.autograd.grad(output, owners, grad_outputs=force, retain_graph=True,
                                    allow_unused=True, materialize_grads=False)
        values.append({name: torch.zeros_like(parameter) if gradient is None else gradient
                       for name, parameter, gradient in zip(names, owners, grads)})
    return tuple(values)


def tensor_stats(value):
    import torch
    x = value.detach().double()
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("nonfinite captured gate")
    return dict(minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()),
                rms=float(x.square().mean().sqrt()))


class RoutingObserver:
    """Delegate each actual mixer once, retaining only this ephemeral output."""
    def __init__(self, routing):
        self.routing, self.codes, self.sites = routing, {}, []

    def mix(self, site, logits):
        codes = self.routing.mix(site, logits)
        if site in self.codes:
            raise AssertionError("duplicate observed routing site")
        self.codes[site] = codes.detach()
        self.sites.append(site)
        return codes


@contextmanager
def observe_forward(generator, callback):
    """Observe output/down tensors; restore all hooks even after a failed call."""
    import torch
    import torch.nn.functional as F
    capture = {"model_outputs": [], "gates": {}, "calls": 0, "observer": None}
    hooks = [generator.model.register_forward_hook(
        lambda module, inputs, output: capture["model_outputs"].append(output))]

    def down_hook(branch):
        def record(module, inputs, hidden):
            observer = capture["observer"]
            if observer is None or branch.site not in observer.codes:
                raise AssertionError("hidden captured before its actual routed code")
            codes = observer.codes.pop(branch.site)
            if codes.ndim != 4 or codes.shape[1] != 2:
                raise AssertionError("expected CFG token axes [B,2,T,Z]")
            codes = torch.cat((codes[:, 0], codes[:, 1]), dim=0)
            if codes.shape[:-1] != hidden.shape[:-1]:
                raise AssertionError("CFG code/hidden ordering mismatch")
            rank = hidden.shape[-1]
            with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
                hpre = F.linear(hidden.detach().float(), branch.bridge.weight[:, :rank], branch.bridge.bias)
                cpre = F.linear(codes.float(), branch.bridge.weight[:, rank:])
                gain = 1.0 + cpre.tanh()
                halves = {}
                for name, h, c, g in zip(("conditional", "unconditional"),
                                         hpre.chunk(2), cpre.chunk(2), gain.chunk(2)):
                    halves[name] = dict(hidden_preactivation=tensor_stats(h),
                        code_preactivation=tensor_stats(c), input_gate_gain=tensor_stats(g),
                        gain_below_005_fraction=float((g < .05).float().mean()),
                        hidden_abs_above_3_fraction=float((h.abs() > 3).float().mean()))
                capture["gates"][branch.site] = halves
        return record

    hooks.extend(branch.down.register_forward_hook(down_hook(branch))
                 for branch in generator.particle_branches())

    def observed(models, context, candidate, routing):
        if capture["observer"] is not None:
            raise AssertionError("overlapping observed forward")
        observer = RoutingObserver(routing)
        capture["observer"] = observer
        try:
            output = callback(models, context, candidate, observer)
            if tuple(observer.sites) != tuple(generator.sites) or observer.codes:
                raise AssertionError("actual sites/code captures incomplete")
            capture["calls"] += 1
            return output
        finally:
            capture["observer"] = None

    try:
        yield observed, capture
    finally:
        for hook in hooks:
            hook.remove()


def runtime_identity(models, table):
    """Check values plus modes, flags, .grad, hook identities and transient state."""
    from supra.particle_pilot import state_digest
    values, runtime = {}, {}
    for role, owner in models.items():
        values[role] = owner.state_dict()
        for name, module in owner.named_modules():
            runtime[role + "/" + name] = dict(training=module.training,
                forward_hooks=list(module._forward_hooks),
                forward_pre_hooks=list(module._forward_pre_hooks),
                backward_hooks=list(module._backward_hooks))
            if hasattr(module, "multiplier") and hasattr(module, "down") and hasattr(module, "up"):
                multiplier = module.multiplier
                runtime[role + "/" + name]["ordinary_lora_multiplier"] = (
                    state_digest(multiplier) if hasattr(multiplier, "detach") else multiplier)
        for name, parameter in owner.named_parameters():
            runtime[role + "/parameter/" + name] = dict(requires_grad=parameter.requires_grad,
                gradient=None if parameter.grad is None else state_digest(parameter.grad))
    values["table"] = table
    runtime["table"] = dict(requires_grad=table.requires_grad,
        gradient=None if table.grad is None else state_digest(table.grad))
    g = models["generator"]
    runtime["host_in_forward"] = g._in_forward
    runtime["branch_frames_empty"] = all(branch.frame is None for branch in g.particle_branches())
    runtime["host_schema"] = {name: getattr(g, name, None)
                              for name in ("cfg", "rank", "z_dim", "architecture", "sites")}
    runtime["encoder_cfg"] = getattr(models["encoder"], "cfg", None)
    runtime["router_schema"] = {name: getattr(models["router"], name, None)
                                for name in ("sites", "site_input_dims", "_query_names")}
    return dict(tensors=state_digest(values), runtime=runtime)


def build_fast(saved, data, device):
    """Instantiate isolated host classes, then load saved FAST tensors strictly.

    No initializer is applied to trained tensors, and no policy, optimizer,
    averaged owner, native RNG or controller is constructed.
    """
    import torch
    from supra.runtime import TARGETS, model_module
    from supra.particle_adapter import GATED_PARTICLE_V3, SupraParticleHost, SupraParticleRouter, adapter_input_dims
    from supra.particle_training_data import FrozenSliderContexts
    from supra.particle_pilot import state_digest
    config, state = saved["config"], saved["policy"]
    if config["architecture"] != GATED_PARTICLE_V3 or config["output_error_guard"]:
        raise ValueError("unexpected saved architecture or output guard")
    teacher_state = {key.removeprefix("teacher."): value
                     for key, value in state["models"]["encoder"].items() if key.startswith("teacher.")}
    with torch.random.fork_rng(devices=[device.index or 0]), torch.device("meta"):
        module = model_module()
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict(teacher_state, strict=True, assign=True)
    base = base.to(device).requires_grad_(False).eval()
    sites = tuple(config["sites"])
    with torch.random.fork_rng(devices=[device.index or 0]):
        generator = SupraParticleHost(base, sites=sites, rank=16, z_dim=4, cfg=3,
                                      architecture=GATED_PARTICLE_V3).to(device)
        router = SupraParticleRouter(site_input_dims=adapter_input_dims(base, sites)).to(device)
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base, cfg=3).to(device)
    owners = dict(generator=generator, router=router, encoder=encoder)
    for role, owner in owners.items():
        owner.load_state_dict(state["models"][role], strict=True)
        flags = state["requires_grad"][role]
        if set(flags) != set(dict(owner.named_parameters())):
            raise AssertionError("isolated named ownership mismatch: " + role)
        for name, parameter in owner.named_parameters():
            parameter.requires_grad_(flags[name])
        owner.eval()
        if state_digest(owner.state_dict()) != state_digest(state["models"][role]):
            raise AssertionError("isolated FAST load mismatch: " + role)
    table = state["table"].detach().to(device).clone().requires_grad_(state["table_requires_grad"])
    if len(sites) != 71 or tuple(table.shape) != (128, 4):
        raise ValueError("unexpected saved owner dimensions")
    if sum(p.numel() for p in generator.parameters() if p.requires_grad) != config["generator_parameters"]:
        raise AssertionError("isolated trainable generator ownership mismatch")
    if sum(p.numel() for p in router.parameters() if p.requires_grad) != config["router_parameters"]:
        raise AssertionError("isolated trainable router ownership mismatch")
    del base
    return owners, table


def parameter_groups(models, table):
    parameters = {"generator." + name: p for name, p in models["generator"].named_parameters()
                  if p.requires_grad}
    parameters.update({"router." + name: p for name, p in models["router"].named_parameters()
                       if p.requires_grad})
    parameters["bank.table"] = table
    groups = {"all": list(parameters), "generator": [n for n in parameters if n.startswith("generator.")],
              "router": [n for n in parameters if n.startswith("router.")], "bank": ["bank.table"]}
    for site in models["generator"].sites:
        groups["branch/" + site] = [name for name in parameters if name.startswith("generator.model." + site + ".")]
        query = models["router"]._query_names[site]
        groups["query/" + site] = [name for name in parameters if name.startswith("router.site_queries." + query + ".")]
        if not groups["branch/" + site] or not groups["query/" + site]:
            raise AssertionError("missing saved site gradient owners")
    return parameters, groups


def diagnose_batch(models, table, judge, context, panels):
    import torch
    import torch.nn.functional as F
    from particlegan import RoutedRows
    from supra.particle_training import raw_model_forward
    from supra.particle_game import features_for_rows, patchify
    parameters, groups = parameter_groups(models, table)
    teacher = models["encoder"].teacher_velocity(context)
    condition = models["encoder"].condition(context)
    with observe_forward(models["generator"], raw_model_forward) as (callback, capture):
        # This temporary owner-free specification observes the same native mix,
        # mass and ordered-site semantics. It has no row controller/diagnostics.
        spec = RoutedRows(model_forward=callback, features=features_for_rows,
                          sites=models["generator"].sites, output_error_guard=False)
        candidate = spec.candidate_for(models, table)
        guided = spec.forward(models, context, candidate)
    if capture["calls"] != 1 or len(capture["model_outputs"]) != 1 or len(capture["gates"]) != 71:
        raise AssertionError("not exactly one complete actual routed model forward")
    conditional, unconditional = capture["model_outputs"][0].float().chunk(2, dim=0)
    bases = .125 * panels.to(device=guided.device)
    residual = patchify(guided - teacher).float() / judge.scale
    fake = (bases + residual.unsqueeze(0)).flatten(0, 1)
    repeated = condition.repeat(4, 1)
    with torch.no_grad():
        real_logits = judge(bases.flatten(0, 1), repeated)
    losses = F.softplus(real_logits - judge(fake, repeated)).reshape(4, len(context)).mean(0)
    loss = losses.mean()
    def decompose(objective):
        upstream = torch.autograd.grad(objective, guided, retain_graph=True)[0].detach()
        c, u, g = half_vjps(conditional, unconditional, guided, upstream, parameters)
        result = {group: geometry({n: c[n] for n in names}, {n: u[n] for n in names},
                                 {n: g[n] for n in names}) for group, names in groups.items()}
        # BF16 host reductions can legitimately round differently when each
        # half is reduced separately. Geometry marks fixed-bound parity and
        # retains the actual guided VJP even when the half sum is unverified.
        return result, tensor_stats(upstream)
    results, upstream_stats = decompose(loss)
    individual = []
    for row in range(len(context)):
        geometry_by_group, force = decompose(losses[row])
        individual.append(dict(batch_row=row, per_context_game=float(losses[row].detach()),
            geometry=geometry_by_group, upstream_velocity_gradient=force,
            gradient_objective="this context's four-draw mean, with other contexts' upstream forces zero"))
    return dict(context_indices=None, per_context_game=losses.detach().cpu().tolist(),
                mean_game=float(loss.detach()), upstream_velocity_gradient=upstream_stats,
                geometry=results, batch_gradient_objective="four-context mean of four-draw context games; includes cross-context summation",
                per_context_gradients=individual, gates=capture["gates"], actual_full_supra_forward_calls=2,
                observed_student_forward_calls=1, frozen_teacher_forward_calls=1,
                optimizer_updates=0, native_policy_or_diagnostic_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=CARD)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-supra-cfg-gradient-v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--qualified-review-sha256", help="Required for execution: root-frozen final independent-review SHA256.")
    args = parser.parse_args()
    card = json.loads(args.card.read_text())
    if not args.execute or not card["execution"]["authorized"]:
        parser.error("source-only diagnostic: coordinator must freeze/authorize this card after the full run qualifies")
    if not args.qualified_review_sha256:
        parser.error("--qualified-review-sha256 is required before loading parent artifacts or CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("launch with CUDA_VISIBLE_DEVICES=0 to bind the declared physical GPU0")
    started = time.monotonic()
    def budget():
        return check_budget(started)
    run = Path(card["parent_run"])
    status = json.loads((run / "status.json").read_text())
    review_path = run / "independent-review.json"
    if status.get("phase") != "complete" or status.get("step") != 6400:
        parser.error("full6400 and independent qualification required; no overlap with the active run")
    try:
        review = qualified_review_identity(card, review_path, args.qualified_review_sha256)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("preserve earlier diagnostics; choose a fresh output directory")
    pg = Path(card["particlegan"]["root"]).resolve()
    if subprocess.check_output(["git", "-C", str(pg), "rev-parse", "HEAD"], text=True).strip() != card["particlegan"]["commit"]:
        parser.error("wrong declared native source")
    sys.path.insert(0, str(pg))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.runtime import MODEL_SOURCE
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import state_digest
    if Path(particlegan.__file__).resolve().parent.parent != pg:
        raise RuntimeError("wrong imported native package")
    if not torch.cuda.is_available():
        parser.error("saved BF16 Supra diagnostic requires physical GPU0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    plan = json.loads((run / "plan.json").read_text())
    backend = Path(plan["protocol"]["inputs"]["backend_model_source"]["path"]).resolve()
    if Path(MODEL_SOURCE).resolve() != backend:
        raise ValueError("actual backend import path differs; use the parent's PARTICLE_SLIDERS_ROOT")
    inputs = {"data": Path(plan["input_paths"]["data"]),
              "D1856": Path(plan["input_paths"]["D1856"]),
              "D6400": Path(plan["input_paths"]["D6400"]),
              "parent_plan": run / "plan.json", "parent_review": review_path,
              "parent_protocol": ROOT / "docs/e22_supra_neutral_initialization_protocol.json"}
    declared = {"data": plan["input_sha256"]["data"], "D1856": plan["input_sha256"]["D1856"],
                "D6400": plan["input_sha256"]["D6400"], "parent_plan": review["plan_sha256"],
                "parent_review": args.qualified_review_sha256,
                "parent_protocol": review["protocol_sha256"]}
    for arm in card["arms"]:
        for step in card["checkpoints"]:
            name = arm + "/" + str(step)
            inputs[name] = run / arm / f"checkpoint-{step:05d}.pt"
            declared[name] = review["checkpoint_artifacts"][arm][str(step)]["file_sha256"]
    sources = {str(ROOT / name): value for name, value in plan["application_source_sha256"].items()}
    sources.update({str(pg / name): value for name, value in plan["particlegan_source_sha256"].items()})
    sources[str(Path(__file__).resolve())] = card["diagnostic_source_sha256"]
    sources[str(args.card.resolve())] = sha(args.card)
    sources.update({str((ROOT / name).resolve()): digest
                    for name, digest in card["reviewer_sources_sha256"].items()})
    sources[plan["protocol"]["inputs"]["backend_model_source"]["path"]] = plan["protocol"]["inputs"]["backend_model_source"]["sha256"]
    if any(sha(Path(name)) != value for name, value in sources.items()):
        raise ValueError("held source/card/native/backend identity changed")
    actual_hashes = {}
    for name, path in inputs.items():
        budget()
        actual_hashes[name] = sha(path)
        if name in declared and actual_hashes[name] != declared[name]:
            raise ValueError("qualified input identity changed: " + name)
    args.output.mkdir(parents=True)
    execution_plan = dict(card=card, card_sha256=sha(args.card), input_paths={n: str(p) for n, p in inputs.items()},
        input_sha256=actual_hashes, source_sha256=sources, parent_review_sha256=actual_hashes["parent_review"],
        qualified_review_sha256=args.qualified_review_sha256,
        device=str(device), gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__)
    write(args.output / "plan.json", execution_plan)
    for name, digest in sources.items():
        path = Path(name)
        prefix = "native" if path.is_relative_to(pg) else "application" if path.is_relative_to(ROOT) else "backend"
        relative = path.relative_to(pg if prefix == "native" else ROOT if prefix == "application" else path.parent)
        target = args.output / "source" / prefix / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(target) != digest:
            raise AssertionError("archived source identity changed")
    results = []
    try:
        data = torch.load(inputs["data"], map_location="cpu", weights_only=False, mmap=True)
        if state_digest(data) != plan["protocol"]["data"]["digest"]:
            raise ValueError("actual frozen data content changed")
        indices, stream_hash = select_fit_indices(data["fit"]["context"])
        context = data["fit"]["context"][indices].contiguous()
        panels = private_panels()
        if indices != card["contexts"]["fit_indices"] or stream_hash != card["contexts"]["selection_stream_sha256"]:
            raise ValueError("fixed selection changed")
        if raw_sha(context) != card["contexts"]["raw_sha256"] or raw_sha(panels) != card["panels"]["unscaled_raw_sha256"]:
            raise ValueError("fixed context or private Gaussian panel changed")
        judges = {}
        global_rng = dict(cpu=raw_sha(torch.get_rng_state()), cuda=raw_sha(torch.cuda.get_rng_state(device)))
        with torch.random.fork_rng(devices=[0]):
            for name in card["judges"]:
                state = torch.load(inputs[name], map_location="cpu", weights_only=False, mmap=True)
                judge = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
                judge.load_state_dict(state["policy"]["models"]["critic"], strict=True)
                judge.eval().requires_grad_(False)
                if state_digest(judge.state_dict()) != review["judges"][name]:
                    raise AssertionError("actual saved common judge tensor mismatch")
                judges[name] = judge
                del state
            for arm in card["arms"]:
                for step in card["checkpoints"]:
                    budget()
                    state = torch.load(inputs[arm + "/" + str(step)], map_location="cpu", weights_only=False, mmap=True)
                    check_checkpoint_contract(state, plan["configs"][arm], step)
                    models, table = build_fast(state, data, device)
                    before = runtime_identity({**models, **judges}, table)
                    # Constructors/load only isolated owners. The native saved
                    # state (including optimizer/controller/RNG/diagnostics) is
                    # never passed to a mutable policy or diagnostic method.
                    for name, judge in judges.items():
                        for start in range(0, 12, 4):
                            budget()
                            with torch.autograd.set_multithreading_enabled(False):
                                result = diagnose_batch(models, table, judge,
                                    context[start:start+4].to(device), panels[:, start:start+4])
                            result.update(arm=arm, checkpoint=step, judge=name,
                                context_indices=indices[start:start+4])
                            for row, item in enumerate(result["per_context_gradients"]):
                                item.update(fit_index=indices[start+row],
                                    source_caption_id=int(context[start+row, 4097]),
                                    flow_time=float(context[start+row, 4096]))
                            results.append(result)
                            print(json.dumps(dict(event="cfg_gradient_batch", arm=arm, checkpoint=step,
                                judge=name, batch=start//4, game=result["mean_game"],
                                geometry=result["geometry"]["all"])), flush=True)
                    if before != runtime_identity({**models, **judges}, table):
                        raise AssertionError("isolated owner values/modes/flags/grads/hooks/transient state changed")
                    del models, table, state, before
                    gc.collect()
                    torch.cuda.empty_cache()
        if global_rng != dict(cpu=raw_sha(torch.get_rng_state()), cuda=raw_sha(torch.cuda.get_rng_state(device))):
            raise AssertionError("global CPU/CUDA random state changed")
        if any(sha(path) != actual_hashes[name] for name, path in inputs.items()):
            raise AssertionError("native/input files changed, including full saved RNG/optimizer/diagnostics")
        if any(sha(Path(name)) != value for name, value in sources.items()):
            raise AssertionError("held sources changed during diagnostic")
        budget()
        if len(results) != 24:
            raise AssertionError("missing fixed arm/checkpoint/judge/batch result")
        final_seconds = final_report(args.output, dict(schema="supra_cfg_gradient_diagnostic_v1", complete=True,
            plan_sha256=sha(args.output / "plan.json"), results=results, seconds=time.monotonic()-started,
            input_files_unchanged=True, isolated_owners_unchanged=True, global_rng_unchanged=True,
            optimizer_updates=0, native_policy_or_diagnostic_calls=0,
            actual_full_supra_forward_calls=48, student_forward_calls=24, teacher_forward_calls=24,
            unverified_batch_decompositions=sum(not result["geometry"]["all"]["decomposition_verified"]
                for result in results),
            unverified_per_context_decompositions=sum(not item["geometry"]["all"]["decomposition_verified"]
                for result in results for item in result["per_context_gradients"]),
            output_objectives_or_metrics_used=False, quality_qualification_credit="none",
            limits=card["limits"]), started)
        print(json.dumps(dict(event="cfg_gradient_complete", complete=True,
            seconds_after_completion_receipt=final_seconds, quality_qualification_credit="none")), flush=True)
    except Exception as error:
        write(args.output / "incomplete.json", dict(complete=False, error=type(error).__name__+": "+str(error),
            completed_batches=len(results), seconds=time.monotonic()-started, results=results,
            quality_qualification_credit="none"))
        raise


if __name__ == "__main__":
    main()
