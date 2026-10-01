"""Experimental same-batch RpGAN trust projection of the native bank step.

The native optimizer updates G/router/bank and every moment normally. Before
native displacement observation, only the applied bank displacement is scaled
to the largest predeclared fraction with zero clean/noisy paired-game harm.
This is an explicit projected-Adam experiment, not a ParticleGAN API fix.
The selector sees no output-error metric, hold-out context, or critic feature
guard; native clean structural guards retain their original independent law.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch

from supra.particle_game import patchify
from supra.particle_pilot import state_digest
from scripts.monitor_e22_supra_particle_convergence import evaluation_modes


FRACTIONS = (1., .5, .25, .125, 0.)
SCHEMA = "e22_bank_samebatch_game_trust_v1"
FORMULATION = "bank_game_trust_v1"


def choose_fraction(baseline, candidates):
    """Largest tested native bank fraction passing both game-only guards."""
    if set(baseline) != {"clean", "noisy"} or not all(math.isfinite(value) for value in baseline.values()):
        raise ValueError("trust baseline must contain finite clean/noisy paired game")
    for fraction, scores in candidates:
        if fraction not in FRACTIONS[:-1] or set(scores) != set(baseline):
            raise ValueError("invalid predeclared bank trust candidate")
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("nonfinite bank candidate game")
        if all(scores[mode] <= baseline[mode] for mode in baseline):
            return fraction
    return 0.


def _mutable_digest(loop):
    """Fingerprint mutable owners; frozen parameters are runner-qualified.

    Including all frozen host weights here would transfer gigabytes to CPU
    on every candidate. Registered buffers, trainable weights, critic weights,
    gradients, moments, control/evidence and every stream remain covered.
    """
    policy = loop.policy
    modules = policy._training_modules()
    parameters = {role: {name: parameter for name, parameter in module.named_parameters()
                         if parameter.requires_grad or role == "critic"}
                  for role, module in modules.items()}
    averages = policy._average_modules()
    average_parameters = {role: {name: parameter for name, parameter in module.named_parameters()
                                if role == "critic" or (role in parameters and name in parameters[role])}
                          for role, module in averages.items()}
    return state_digest(dict(parameters=parameters,
        average_parameters=average_parameters,
        average_buffers={role: dict(module.named_buffers()) for role, module in averages.items()},
        buffers={role: dict(module.named_buffers()) for role, module in modules.items()},
        gradients={role: {name: parameter.grad for name, parameter in module.named_parameters()
                          if parameter.requires_grad or role == "critic"} for role, module in modules.items()},
        table_grad=policy.table.grad,
        modes={role: [module.training for module in owner.modules()] for role, owner in modules.items()},
        requires_grad={role: {name: parameter.requires_grad for name, parameter in module.named_parameters()}
                       for role, module in modules.items()},
        optimizers=[optimizer.state_dict() for optimizer in policy.optimizers],
        controller=policy.controller.state_dict(), routing=policy.routed_control.state_dict(),
        lr_settle=policy.lr_settle.state_dict(), surprise=None if policy.surprise is None else policy.surprise.state_dict(),
        penalty=policy.penalty.regularizer.state_dict(), penalty_stats=policy.penalty.last_stats,
        phase=policy._phase, completed_steps=policy.completed_steps,
        data_rng=loop.data_rng.get_state(), paired_rng=loop.paired_noise_rng.get_state(),
        streams={name: getattr(policy, name).get_state() for name in policy._STREAMS},
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(policy.device) if policy.device.type == "cuda" else None))


class BankGameTrust:
    """An isolated helper installed around explicitly versioned trial updates.

    ``begin_update`` must precede the native update's paired Gaussian draws.
    Counters/history are diagnostics only; no state changes the fixed trust law.
    Native resume must opt into this experimental law explicitly in its runner.
    """

    def __init__(self):
        self.records = []
        self._loop = None
        self._capture = None

    @property
    def last_record(self):
        if not self.records:
            raise RuntimeError("bank trust has not completed an update")
        return self.records[-1]

    def state_dict(self):
        if self._capture is not None or (self._loop is not None and self._loop.policy._phase != "ready"):
            raise RuntimeError("bank trust checkpoints require a completed native boundary")
        return dict(schema=SCHEMA, fractions=list(FRACTIONS), moment_law="native Adam moments retained; applied table step projected",
                    game_guard="zero harm on both clean and native noisy same-batch RpGAN with post-step G/router",
                    completed_trials=len(self.records), records=deepcopy(self.records))

    def begin_update(self):
        if self._loop is None or self._capture is not None:
            raise RuntimeError("install bank trust and finish prior native update before beginning")
        loop, policy = self._loop, self._loop.policy
        if policy._phase != "ready":
            raise RuntimeError("trust initialization requires native ready boundary")
        stream = torch.Generator(device=policy.device).set_state(loop.paired_noise_rng.get_state().cpu())
        shape = (policy.recipe.batch_size, 256, 16)
        torch.randn(shape, device=policy.device, dtype=torch.float32, generator=stream)
        generator_base = torch.randn(shape, device=policy.device, dtype=torch.float32, generator=stream)
        self._capture = dict(step=policy.completed_steps + 1, calls=0, generator_base=generator_base)

    @contextmanager
    def install(self, loop):
        if self._loop is not None:
            raise RuntimeError("bank trust helper is already installed")
        policy = loop.policy
        if policy.recipe.continuous_policy != "dv12" or policy.row_policy != "routed_paired":
            raise ValueError("bank trust requires native routed DV12")
        if policy.routed_control.spec.max_context_harm != 0. or policy.routed_control.spec.output_error_guard:
            raise ValueError("bank trust requires unchanged zero feature-harm native guard and no output guard")
        table_groups = [group for group, role in zip(policy.opt_g.param_groups, policy.roles[0]) if role == "table"]
        if len(table_groups) != 1 or len(table_groups[0]["params"]) != 1 or table_groups[0]["params"][0] is not policy.table:
            raise ValueError("bank trust requires one explicitly owned native table group")
        if policy._table_location is not None or not policy.table.requires_grad:
            raise ValueError("bank trust prototype requires an external learned particle bank")
        original_generate, original_step = policy.routed_generate, policy.opt_g.step
        self._loop = loop

        def generated(context, **kwargs):
            capture = self._capture
            if capture is None or capture["step"] != policy.completed_steps + 1:
                raise RuntimeError("call trust.begin_update before each native update")
            capture["calls"] += 1
            if capture["calls"] == 2:
                if len(context) != policy.recipe.batch_size or kwargs.get("sigma") != 0 or kwargs.get("perturb") is not True:
                    raise RuntimeError("unexpected native G-pass sampling contract")
                capture.update(context=context.detach().clone(), dv12_state=policy.noise_generator.get_state().clone(),
                               controller=deepcopy(policy.controller), sigma=policy._noise.output_sigma.detach().clone())
            value = original_generate(context, **kwargs)
            if capture["calls"] == 2:
                capture["prediction"] = value.detach().clone()
            return value

        @torch.no_grad()
        def step(*args, **kwargs):
            capture = self._capture
            if capture is None or capture["calls"] != 2 or policy._phase != "generator_step":
                raise RuntimeError("trust projection requires the actual completed native G backward")
            table_before = policy.table.detach().clone()
            before_proof = _mutable_digest(loop)
            with evaluation_modes(policy):
                proof = self._prediction(noisy=True)
            if not torch.equal(proof, capture["prediction"]):
                raise RuntimeError("private trust replay failed exact actual pre-step native G forward")
            if before_proof != _mutable_digest(loop):
                raise RuntimeError("private pre-step proof altered native mutable state or streams")
            if not torch.equal(table_before, policy.table):
                raise RuntimeError("private pre-step proof changed the actual particle bank")
            native_return = original_step(*args, **kwargs)
            table_proposal = policy.table.detach().clone()
            displacement = table_proposal - table_before
            native_moments = state_digest([optimizer.state_dict() for optimizer in policy.optimizers])
            # Table is the sole permitted owner change during trust evaluation.
            invariants = _mutable_digest(loop)
            condition = policy.encoder.condition(capture["context"])
            real = capture["sigma"] * capture["generator_base"]
            real_logits = policy.D(real, condition)
            candidates = []
            try:
                with evaluation_modes(policy):
                    policy.table.copy_(table_before)
                    baseline = self._games(real, real_logits, condition)
                    for fraction in FRACTIONS[:-1]:
                        policy.table.copy_(table_proposal if fraction == 1. else table_before + fraction * displacement)
                        scores = self._games(real, real_logits, condition)
                        candidates.append((fraction, scores))
                        if choose_fraction(baseline, [(fraction, scores)]) != 0.:
                            break
            finally:
                policy.table.copy_(table_proposal)
            if invariants != _mutable_digest(loop):
                raise RuntimeError("private bank trust candidates altered other mutable owners or streams")
            selected = choose_fraction(baseline, candidates)
            policy.table.copy_(table_before if selected == 0. else table_proposal if selected == 1.
                               else table_before + selected * displacement)
            if native_moments != state_digest([optimizer.state_dict() for optimizer in policy.optimizers]):
                raise RuntimeError("projected bank step changed native optimizer moments")
            chosen = baseline if selected == 0. else next(scores for fraction, scores in candidates if fraction == selected)
            record = dict(schema=SCHEMA, step=capture["step"], fractions=list(FRACTIONS), selected_fraction=selected,
                baseline=baseline, candidates=[dict(fraction=fraction, game=scores,
                    new_minus_baseline={mode: scores[mode] - baseline[mode] for mode in baseline}) for fraction, scores in candidates],
                accepted_game=chosen, accepted_new_minus_baseline={mode: chosen[mode] - baseline[mode] for mode in baseline},
                proposed_displacement_norm=float(displacement.double().norm()),
                applied_displacement_norm=float((policy.table - table_before).double().norm()),
                dense_native_gradient_rows=0 if policy.table.grad is None else int(policy.table.grad.norm(dim=-1).gt(0).sum()),
                original_native_Gpass_replayed_exact=True, private_evaluations_state_unchanged=True,
                native_optimizer_moments_unchanged=True, output_metrics_used=False,
                pre_step_output_sigma=float(capture["sigma"]), task_weight=.1 if capture["step"] % 5 == 0 else 1.,
                moment_law="native Adam moments retained; applied table step projected",
                criterion="same training batch only; zero clean and native noisy paired-game harm relative post-G/router fraction0")
            self.records.append(record)
            self._capture = None
            return native_return

        policy.routed_generate, policy.opt_g.step = generated, step
        try:
            yield self
            if self._capture is not None:
                raise RuntimeError("bank trust context ended before a complete projected update")
        finally:
            policy.routed_generate, policy.opt_g.step = original_generate, original_step
            self._loop = None
            self._capture = None

    @torch.no_grad()
    def _prediction(self, *, noisy):
        policy, capture = self._loop.policy, self._capture
        candidate = policy.routed_control.candidate()
        controller = capture["controller"]
        stream = torch.Generator(device=policy.device).set_state(capture["dv12_state"].cpu())
        prior = controller.routed_prior(candidate.table, candidate.log_mass) if noisy else None
        return policy.routed_control.spec.forward(policy._training_modules(), capture["context"], candidate,
            perturb_fn=(lambda codes: controller.perturb_latent(codes, stream, prior, record=False)) if noisy else None)

    @torch.no_grad()
    def _games(self, real, real_logits, condition):
        policy = self._loop.policy
        result = {}
        for mode, noisy in (("noisy", True), ("clean", False)):
            residual = self._prediction(noisy=noisy)
            fake = real + patchify(residual).float() / policy.D.scale
            score = float(policy.recipe.make_loss().g_loss(policy.D(fake, condition), real_logits))
            if not math.isfinite(score):
                raise RuntimeError("nonfinite same-batch bank trust game")
            result[mode] = score
        return result


@contextmanager
def bank_game_trust(loop):
    """Install the helper; call begin_update before every native update."""
    helper = BankGameTrust()
    with helper.install(loop):
        yield helper
