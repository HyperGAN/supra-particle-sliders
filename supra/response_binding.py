"""Supra runtime adapters from the MIT-licensed core model bindings."""
import torch
from particle_sliders.backends import FrozenResponseAdapter, _adapter_parameters, _roots, bind_native_gradient_checkpointing

class AnimaResponseAdapter(FrozenResponseAdapter):
    family = "anima"
    trajectory_direction = "decreasing"
    time_coordinates = "scheduler training timestep; LiveAnimaBackend normalizes by num_train_timesteps"

    def __init__(self, backend, *, decoder=None, model_identity=None):
        self.backend = backend
        super().__init__(_adapter_parameters(backend), decoder=decoder, model_identity=model_identity,
                         trajectory_direction=self.trajectory_direction)
        roots = list(_roots(backend))
        pipe = getattr(backend, "pipe", None)
        if pipe is not None and hasattr(pipe, "transformer"):
            roots.append(("pipe.transformer", pipe.transformer))
        seen = set()
        for _, root in roots:
            if id(root) not in seen:
                bind_native_gradient_checkpointing(root, self.network)
                seen.add(id(root))

    def _velocity(self, condition, latent, timestep):
        maximum = getattr(self.backend, "num_train_timesteps", 1000)
        if torch.any((timestep < 0) | (timestep > maximum)):
            raise ValueError("Anima time must be a raw scheduler timestep")
        # No scale argument: the outer reversible context must survive backward.
        # Encoding is live, so conditioner LoRA is never replaced by frozen embeds.
        return self.backend.predict_v(condition, latent, timestep)


class SupraResponseAdapter(AnimaResponseAdapter):
    family = "supra"
    trajectory_direction = "increasing"
    time_coordinates = "Supra flow time t=0 noise -> t=1 data; dx/dt=velocity"

    def _velocity(self, condition, latent, timestep):
        if torch.any((timestep < 0) | (timestep > 1)):
            raise ValueError("Supra time must be flow sigma in [0,1]")
        # LiveSupraBackend.predict_v resets scale to 1 internally. Reuse its frozen
        # encoding/context and native transformer directly under our scale scope.
        if hasattr(self.backend, "_context"):
            embeds, mask = self.backend._context(condition)
            hidden = latent.to(self.backend.device)
            embeds = embeds.to(hidden)
            if embeds.ndim == 2:
                embeds = embeds.unsqueeze(0)
            time = timestep.reshape(-1).to(hidden.device, torch.float32).expand(hidden.shape[0])
            mask = None if mask is None else mask.to(hidden.device)
            return self.backend.transformer(hidden, time, embeds, mask).float()
        embeds, _ = self.backend.encode_text(condition)
        return self.backend._forward(latent, timestep, embeds)


class _SupraRuntimeBackend:
    """Inspectable runtime view; reuses the product's actual model and text cache."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.transformer, self.device = runtime.model, runtime.device

    def trainable_parameters(self):
        return self.runtime.parameters()

    def _context(self, condition):
        return self.runtime.encode(condition)


class SupraRuntimeResponseAdapter(SupraResponseAdapter):
    """Reuse supra.runtime.SupraRuntime's frozen text cache, DiT, LoRA and VAE."""
    family = "supra_runtime"

    def __init__(self, runtime, *, model_identity=None):
        self.runtime = runtime
        backend = _SupraRuntimeBackend(runtime)
        super().__init__(backend, decoder=self._decode, model_identity=model_identity)

    def _velocity(self, condition, latent, timestep):
        # Preserve product autocast while controlling conditional/negative scales
        # separately, unlike legacy runtime.velocity's concatenated CFG branches.
        from contextlib import nullcontext
        context = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if self.runtime.device.type == 'cuda' else nullcontext())
        with context:
            return super()._velocity(condition, latent, timestep)

    def _decode(self, latent):
        from contextlib import nullcontext
        context = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if self.runtime.device.type == 'cuda' else nullcontext())
        with context:
            return self.runtime.vae.decode(latent / .18215).sample
