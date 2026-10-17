"""Native Supra components with architecture/LoRA from shared particle-sliders."""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from supra.shared import shared_module

MODEL_ID = "SupraLabs/Supra2-IMG"
MODEL_REV = "10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b"
T5_REV = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
VAE_REV = "31f26fdeee1355a5c34592e401dd41e45d25a493"
TARGETS = ("ctx_proj", "cross_attn.q", "cross_attn.kv", "cross_attn.proj",
           "self_attn.qkv", "self_attn.proj")
BACKEND_ROOT = Path(os.environ.get("PARTICLE_SLIDERS_ROOT",
    str(Path(__file__).resolve().parents[1] / "external-particle-sliders")))
MODEL_SOURCE = BACKEND_ROOT / "conceptmod/textsliders/supra_model.py"


def model_module():
    return shared_module("conceptmod.textsliders.supra_model")


def adapter_state(model):
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
            if n.endswith((".down.weight", ".up.weight"))}


def load_adapter_state(model, state):
    expected = {n: p for n, p in model.named_parameters() if n.endswith((".down.weight", ".up.weight"))}
    if set(state) != set(expected):
        raise ValueError(f"Adapter keys differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    # Validate the complete payload before load_state_dict can copy any leaf.
    for name, value in state.items():
        parameter = expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
            raise ValueError("Adapter tensor schema/finiteness differs: " + name)
    model.load_state_dict(state, strict=False)


class SupraRuntime:
    MODEL_REV, T5_REV = MODEL_REV, T5_REV

    def __init__(self, device="cuda:0", rank=16, allow_hub=False):
        mod = model_module()
        from diffusers import AutoencoderKL
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, T5EncoderModel

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Live Supra training/sampling requires CUDA")
        torch.cuda.set_device(self.device)
        self.rank = rank
        self.cfg_adapter_branch = "both"
        local = not allow_hub
        self.checkpoint = hf_hub_download(MODEL_ID, "model_final_ema.pt", revision=MODEL_REV,
                                          local_files_only=local)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        cfg = state.get("config", {})
        self.ctx_len = int(cfg.get("ctx_len", 128))
        self.model = mod.SupraDiT()
        mod.load_supra_weights(self.model, state, strict=True)
        self.base_params = sum(p.numel() for p in self.model.parameters())
        self.loras = mod.attach_supra_lora(self.model, rank=rank, alpha=rank, targets=TARGETS)
        self.model.to(self.device).eval()
        self.cache = {}
        self.token_counts = {}
        if "uncond_text" in cfg:
            self.cache[""] = (cfg["uncond_text"].float().unsqueeze(0).to(self.device),
                               cfg["uncond_mask"].float().unsqueeze(0).to(self.device))
        self.stored_unconditional = "" in self.cache
        del state
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", revision=T5_REV,
                                                       local_files_only=local)
        self.text_encoder = T5EncoderModel.from_pretrained(
            "google/flan-t5-base", revision=T5_REV, local_files_only=local,
        ).to(self.device).eval().requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", revision=VAE_REV, local_files_only=local,
        ).to(self.device).eval().requires_grad_(False)
        print(f"Loaded Supra {self.base_params:,} parameters, "
              f"{sum(p.numel() for p in self.parameters()):,} trainable; "
              f"stored unconditional={self.stored_unconditional}", flush=True)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode(self, prompt):
        if prompt not in self.cache:
            count = len(self.tokenizer(prompt)["input_ids"])
            if count > self.ctx_len:
                raise ValueError(f"Prompt exceeds {self.ctx_len} tokens ({count}): {prompt}")
            self.token_counts[prompt] = count
            tokens = self.tokenizer(prompt, padding="max_length", max_length=self.ctx_len,
                                    truncation=False, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = self.text_encoder(**tokens).last_hidden_state.float()
            self.cache[prompt] = (ctx, tokens.attention_mask.float())
        return self.cache[prompt]

    @contextmanager
    def scale(self, value):
        prev = [m.multiplier for m in self.loras]
        for m in self.loras:
            m.multiplier = float(value)
        try:
            yield
        finally:
            for m, old in zip(self.loras, prev):
                m.multiplier = old

    def velocity(self, prompts, z, t, scale=0.0, cfg=3.0):
        if not math.isfinite(cfg) or cfg < 0 or not math.isfinite(scale):
            raise ValueError("Finite scale and nonnegative guidance required")
        times = torch.as_tensor(t, device=z.device).float().reshape(-1)
        if (times.numel() not in (1, z.shape[0]) or not torch.isfinite(times).all()
                or torch.any((times < 0) | (times > 1))):
            raise ValueError("One native timestep in [0,1] per latent required")
        if isinstance(prompts, str):
            prompts = [prompts] * z.shape[0]
        if len(prompts) != z.shape[0]:
            raise ValueError("One prompt per latent required")
        if self.cfg_adapter_branch == "conditional_only":
            mod = shared_module("conceptmod.textsliders.pg09_backends")
            backend = mod.SupraRuntimeResponseAdapter(self,
                model_identity=dict(model_revision=MODEL_REV))
            times = torch.as_tensor(t, device=z.device).float().reshape(-1)
            if times.numel() == 1:
                times = times.expand(z.shape[0])
            if times.numel() != z.shape[0]:
                raise ValueError("One native timestep per latent required")
            return torch.cat([backend.cfg(mod.ResponseRow(prompt, prompt, "", latent,
                timestep=time, guidance=cfg), latent, time, scale=scale)
                for prompt, latent, time in zip(prompts, z.split(1), times)])
        contexts, masks = zip(*(self.encode(p) for p in prompts))
        ctx, mask = torch.cat(contexts), torch.cat(masks)
        t = torch.as_tensor(t, device=self.device).float().reshape(-1)
        if t.numel() == 1:
            t = t.expand(z.shape[0])
        if cfg != 1:
            uctx, umask = self.encode("")
            ctx = torch.cat([ctx, uctx.expand(z.shape[0], -1, -1)])
            mask = torch.cat([mask, umask.expand(z.shape[0], -1)])
            z, t = torch.cat([z, z]), torch.cat([t, t])
        with self.scale(scale), torch.autocast("cuda", dtype=torch.bfloat16):
            v = self.model(z, t, ctx, mask).float()
        if cfg != 1:
            cond, uncond = v.chunk(2)
            v = uncond + cfg * (cond - uncond)
        return v

    def noise(self, seed, n=1):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(n, 4, 32, 32, generator=generator, device=self.device)

    @torch.no_grad()
    def trajectory(self, prompt, seed, steps=50, cfg=3.0, scale=0.0, keep=False):
        if type(steps) is not int or steps < 1:
            raise ValueError("Positive integer sampling steps required")
        z = self.noise(seed)
        states = []
        for i in range(steps):
            if keep:
                states.append(z.cpu())
            z = z + self.velocity(prompt, z, i / steps, scale=scale, cfg=cfg) / steps
        return (z, states) if keep else z

    @torch.no_grad()
    def render(self, prompt, seed, scale=0.0, steps=50, cfg=3.0):
        if self.cfg_adapter_branch == "conditional_only":
            from supra.pg09 import render
            return render(self, prompt, seed, scale, steps, cfg)
        z = self.trajectory(prompt, seed, steps, cfg, scale)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.vae.decode(z / 0.18215).sample
        pixels = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
        return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())

    def save(self, path, extra=None):
        metadata = {"format": "supra-native-lora-v1", "model_id": MODEL_ID,
                    "model_revision": MODEL_REV, "rank": self.rank, "alpha": self.rank,
                    "targets": list(TARGETS), "text_encoder_revision": T5_REV,
                    "vae_revision": VAE_REV, "cfg_adapter_branch": self.cfg_adapter_branch,
                    **(extra or {})}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(adapter_state(self.model), str(path),
                  metadata={k: json.dumps(v) for k, v in metadata.items()})
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    def load(self, path):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as handle:
            meta = {k: json.loads(v) for k, v in handle.metadata().items()}
        for key, value in {"format": "supra-native-lora-v1", "rank": self.rank,
                           "alpha": self.rank, "model_id": MODEL_ID,
                           "model_revision": MODEL_REV, "targets": list(TARGETS),
                           "text_encoder_revision": T5_REV, "vae_revision": VAE_REV}.items():
            if meta.get(key) != value:
                raise ValueError(f"Adapter {key} mismatch: {meta.get(key)} != {value}")
        branch = meta.get("cfg_adapter_branch", "both")
        if branch not in ("both", "conditional_only"):
            raise ValueError("Unknown adapter CFG branch policy")
        if branch == "conditional_only":
            from supra.shared import LOCK, file_sha
            if (meta.get("training_objective") != "pg09_gan_only"
                    or meta.get("shared_lock_sha256") != file_sha(LOCK)):
                raise ValueError("PG adapter objective/shared source binding differs")
        load_adapter_state(self.model, load_file(str(path)))
        self.cfg_adapter_branch = branch
