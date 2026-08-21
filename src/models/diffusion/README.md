# Diffusion models

This is the canonical directory for diffusion-related model implementations.

> Note: the historical `src/models/diffussion/` path is deprecated. New code should
> use `src/models/diffusion/`. A later cleanup should move all files from
> `diffussion/` here and remove the old directory.

## Backend ownership

Neural-network components (`DDPMUNet`, `ClsCondDDPMUNet`, and
`AutoEncoderKL`) own unified `InferenceSession` objects through
`BackendCvModel`. Samplers are scheduling algorithms rather than single-shot
inference models: they intentionally remain `BaseAiModel` composers and call
the network wrapper once per denoising timestep.

`LDMSampler` supports both DDPM and DDIM scheduling over one latent input. It
creates one shared latent `DDPMUNet`, initializes that session once, and
injects it into both schedulers. The VAE decoder remains a separate session.
