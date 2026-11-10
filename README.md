# SLAVIK research code

This repository collects the SLAVIK activation research code, compact language
model pilot, and a small image-diffusion prototype.

## What is included

- `slavik_activations.py` — shared trainable SLA'vik-A and SLA'vik-R2 cells.
- `slavik_llm.py` — character-level decoder-only Transformer pilot.
- `slavik_diffusion.py` — class-conditional latent DDPM for MNIST, with
  SLA'vik-A activations.
- `slavik_cells.py` — modular classification and cell-audit experiments.
- `slavik_vae_*.py` — earlier VAE experiments.
- Russian-language reports document experimental settings, results, and limits.
- `tests/` — shape, gradient, and diffusion-schedule tests.

This is research code, not a 1B language model or a reproduction of Stable
Diffusion. The image model generates 28x28 MNIST digits conditioned on a digit
label; it does not support text prompts or photorealistic images. See
`SLAvik_диффузия_на_ноутбуке.md` for its setup and training instructions.

## Install and test

Python 3.12 is recommended. Install the dependencies with:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-slavik-diffusion.txt
.venv/bin/python -m unittest discover -s tests -v
```

MNIST is not bundled. Place the standard IDX files under `data/MNIST/raw/`, or
pass another data root with `--data`. The diffusion script can use Apple MPS,
CUDA, or CPU.

## Attribution and scope

The research and SLA'vik concept are presented by the project author. Code
preparation and integration for this publication involved AI assistance; that
contribution is identified in the corresponding commit. The destination
repository had no prior commits, so this publication starts its history rather
than rewriting existing history.

Model checkpoints, generated images, datasets, experiment logs, and plot assets
are intentionally not included.
