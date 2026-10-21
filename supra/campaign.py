"""Thin Supra native orchestration; shared core owns every GAN update/law.

No MOSS/FinalForm orchestration or generic lifecycle implementation is copied.
The only neural critic and GAN are imported from the reviewed core. See the
selected-definition provenance receipt and licenses/NOTICE.
"""
from dataclasses import dataclass
import json
import math

import torch
from particle_sliders.pg09_backends import FlowSchedule, ResponseRow, SupraRuntimeResponseAdapter
from particle_sliders.field_critic import FieldCritic as CampaignFieldCritic
from particle_sliders.field_critic import seed_initialization_rng as _seed_initialization_rng
from .cache import SupraCampaignRows, tensor_identity


@dataclass
class SupraCampaignRun:
    runtime: object
    rows: SupraCampaignRows
    backend: object
    game: object

    def step(self):
        if self.game.completed_steps >= self.game.recipe.total_steps:
            raise RuntimeError('fixed GAN campaign budget exhausted')
        return self.game.step()

    def render_cases(self):
        lookup = {r['neutral']: r['positive'] for r in self.rows.config['rows']}
        for row in self.rows.config['verification']:
            for seed in (42, 1234):
                for label, prompt, scale in (
                    ('off', row['prompt'], 0.), ('half', row['prompt'], .5),
                    ('plus', row['prompt'], 1.),
                    ('teacher', row.get('teacher', lookup.get(row['prompt'], row['prompt'])), 0.),
                ):
                    yield dict(row_id=row['name'], seed=seed, label=label, prompt=prompt, scale=scale,
                               steps=50, cfg=3., width=256, height=256,
                               original_cfg_adapter='both branches', corrected_cfg_adapter='conditional_only',
                               noise_generator='native CUDA SupraRuntime.noise', heldout_training=False)

    @torch.no_grad()
    def render(self, case):
        # Same native CUDA RNG and positive Euler convention as archived images.
        from PIL import Image
        noise = self.runtime.noise(case['seed'])
        row = ResponseRow(case['prompt'], case['prompt'], '', noise,
                          schedule=FlowSchedule.supra(case['steps']), guidance=case['cfg'])
        image = self.backend.render(row, scale=case['scale'])
        if not torch.isfinite(image).all():
            raise FloatingPointError('nonfinite Supra VAE float output before image conversion')
        pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
        return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())

def build_supra_campaign(runtime, rows, *, model_identity, total_steps=28800, batch_size=4,
                         seed=7, critic_width=32, response_scale=None, recipe_overrides=None):
    """Bind already-admitted native runtime; no model loads or implicit text encode."""
    from particle_sliders.pg09 import ConditionalPG09
    if rows.split != 'train':
        raise ValueError('training requires train split; validation is evaluation-only')
    if total_steps < 1 or total_steps > 28800 or batch_size != 4:
        raise ValueError('bounded Supra campaign requires batch4 and at most28800 steps')
    if not model_identity.get('frozen_parameter_sha256'):
        raise ValueError('resource owner must supply verified frozen base SHA')
    for prompt, expected in rows.contexts.items():
        if prompt not in runtime.cache:
            raise ValueError('admitted native runtime lacks frozen context: ' + prompt)
        if any(tensor_identity(a) != tensor_identity(b) for a, b in zip(runtime.cache[prompt], expected)):
            raise ValueError('admitted native text context differs from frozen export')
    if not runtime.loras or any(not torch.isfinite(p).all() for p in runtime.parameters()):
        raise ValueError('finite native LoRA parameters required')
    if any(torch.count_nonzero(module.up.weight) for module in runtime.loras):
        raise ValueError('fresh zero-up native LoRA required; restore only reviewed full GAN checkpoints after build')
    calibrated_scale = rows.response_scale()
    if response_scale is not None and not math.isclose(calibrated_scale, response_scale, rel_tol=0, abs_tol=1e-12):
        raise ValueError('reviewed training response scale differs')
    backend = SupraRuntimeResponseAdapter(runtime, model_identity=model_identity)
    native_decode = backend.decoder
    def checked_decode(latent):
        if not torch.isfinite(latent).all():
            raise FloatingPointError('nonfinite Supra latent before VAE decode')
        image = native_decode(latent)
        if not torch.isfinite(image).all():
            raise FloatingPointError('nonfinite Supra VAE float output')
        return image
    backend.decoder = checked_decode
    rows.device = torch.device(runtime.device)
    device = rows.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        _seed_initialization_rng(seed, devices)
        critic = CampaignFieldCritic(4, rows.condition_width, critic_width).to(device)
    game = ConditionalPG09(backend, rows, critic, conditions=None, particles=None,
                          response_domain='continuous_field', response_scale=calibrated_scale,
                          preset='bispf_optimized', batch_size=batch_size, seed=seed,
                          total_steps=total_steps, sampling='balanced', recipe_overrides=recipe_overrides)
    return SupraCampaignRun(runtime, rows, backend, game)

def teacher_parity(run, indices, *, relative_rms_limit=.001, absolute_rms_limit=1e-6):
    """Read-only teacher/cache gate, never a fitted loss or optimizer.

    Edited rows retain the frozen edit RMS as their comparison scale.
    Preservation rows have no edit; compare against their frozen neutral
    velocity RMS, with an absolute RMS floor for zero or tiny velocities.
    """
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) for value in (relative_rms_limit, absolute_rms_limit)) or
            relative_rms_limit <= 0 or absolute_rms_limit < 0):
        raise ValueError('finite positive relative and nonnegative absolute RMS limits required')
    results = []
    with torch.no_grad():
        for index in indices:
            cached = run.rows[index]
            actual = run.backend.prepare(cached.row)
            edit = float((cached.target.float() - cached.neutral.float()).square().mean().sqrt())
            preservation = edit == 0.
            reference = float(cached.neutral.float().square().mean().sqrt()) if preservation else edit
            tolerance = relative_rms_limit * reference
            if preservation:
                tolerance = max(tolerance, absolute_rms_limit)
                reference = max(reference, absolute_rms_limit / relative_rms_limit)
            reference = max(reference, 1e-20)
            target_error = float((actual.target.float() - cached.target.to(
                device=actual.target.device, dtype=torch.float32)).square().mean().sqrt())
            base_error = float((actual.neutral.float() - cached.neutral.to(
                device=actual.neutral.device, dtype=torch.float32)).square().mean().sqrt())
            results.append(dict(index=index,
                                target_relative_rms=target_error / reference,
                                base_relative_rms=base_error / reference,
                                normalization='frozen_neutral_velocity_rms' if preservation else 'frozen_edit_rms',
                                normalization_rms=reference, target_rms_error=target_error,
                                base_rms_error=base_error, rms_tolerance=tolerance,
                                relative_rms_limit=relative_rms_limit,
                                absolute_rms_limit=absolute_rms_limit if preservation else 0.))
    if any(not all(math.isfinite(r[key]) for key in (
                'target_rms_error', 'base_rms_error', 'normalization_rms', 'rms_tolerance')) or
           max(r['target_rms_error'], r['base_rms_error']) > r['rms_tolerance'] for r in results):
        error = ValueError('native teacher/cache parity failed: ' + json.dumps(results))
        error.diagnostics = results
        raise error
    return results
