"""ContinualHyper: a UnHype-style LoRA-hypernetwork for continual learning in diffusion models.

The hypernetwork is conditioned on a prompt's CLIP `pooler_output` and the diffusion timestep;
per-layer heads emit a LoRA on the UNet cross-attention. Concepts are learned sequentially
(continual learning). This package is self-contained: it loads SD-1.5 via `diffusers`.

Step 1 (current): sequential CL with NO regularization -> demonstrates catastrophic forgetting.
Next: von-Oswald hypernet regularization (arxiv:1906.00695), then LoRA (arxiv:2508.08812).
"""
