"""Small, class-conditional latent diffusion model using SLA'vik-A cells.

This is a local MNIST research prototype, not a reproduction of Stable
Diffusion. It uses no downloaded weights and needs no network access.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset

from slavik_activations import SlavikA

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "MNIST"
DEFAULT_MODELS = ROOT / "models"
NULL_CLASS = 10


class MNISTRaw(Dataset):
    """Read the already-downloaded MNIST IDX files without torchvision."""

    def __init__(self, root: Path, train: bool):
        prefix = "train" if train else "t10k"
        image_path = root / "raw" / f"{prefix}-images-idx3-ubyte"
        label_path = root / "raw" / f"{prefix}-labels-idx1-ubyte"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"MNIST raw files not found under {root / 'raw'}"
            )
        image_bytes = image_path.read_bytes()
        label_bytes = label_path.read_bytes()
        if len(image_bytes) < 16 or len(label_bytes) < 8:
            raise ValueError("MNIST IDX file is truncated")
        magic, count, rows, cols = struct.unpack(">IIII", image_bytes[:16])
        label_magic, label_count = struct.unpack(">II", label_bytes[:8])
        if magic != 2051 or label_magic != 2049 or count != label_count:
            raise ValueError("MNIST IDX headers are invalid or inconsistent")
        if len(image_bytes) != 16 + count * rows * cols:
            raise ValueError("MNIST image payload has an invalid length")
        if len(label_bytes) != 8 + label_count:
            raise ValueError("MNIST label payload has an invalid length")
        self.images = torch.frombuffer(
            bytearray(image_bytes[16:]), dtype=torch.uint8
        ).view(count, 1, rows, cols)
        self.labels = torch.frombuffer(
            bytearray(label_bytes[8:]), dtype=torch.uint8
        ).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].float().div_(127.5).sub_(1)
        return image, self.labels[index]


def choose_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable in this PyTorch build")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(root: Path, train: bool, limit: int, batch_size: int,
                shuffle: bool) -> DataLoader:
    dataset = MNISTRaw(root, train=train)
    if limit > 0:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def group_count(channels: int) -> int:
    return math.gcd(channels, 8)


class SlavikAutoencoder(torch.nn.Module):
    def __init__(self, latent_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.enc1 = torch.nn.Conv2d(1, base_channels, 4, stride=2, padding=1)
        self.enc1_norm = torch.nn.GroupNorm(group_count(base_channels), base_channels)
        self.enc1_act = SlavikA(base_channels, channel_dim=1)
        self.enc2 = torch.nn.Conv2d(base_channels, base_channels * 2, 4,
                                    stride=2, padding=1)
        self.enc2_norm = torch.nn.GroupNorm(
            group_count(base_channels * 2), base_channels * 2
        )
        self.enc2_act = SlavikA(base_channels * 2, channel_dim=1)
        self.to_mu = torch.nn.Conv2d(base_channels * 2, latent_channels, 3, padding=1)
        self.to_logvar = torch.nn.Conv2d(
            base_channels * 2, latent_channels, 3, padding=1
        )
        self.dec1 = torch.nn.ConvTranspose2d(
            latent_channels, base_channels * 2, 4, stride=2, padding=1
        )
        self.dec1_norm = torch.nn.GroupNorm(
            group_count(base_channels * 2), base_channels * 2
        )
        self.dec1_act = SlavikA(base_channels * 2, channel_dim=1)
        self.dec2 = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, 4, stride=2, padding=1
        )
        self.dec2_norm = torch.nn.GroupNorm(group_count(base_channels), base_channels)
        self.dec2_act = SlavikA(base_channels, channel_dim=1)
        self.to_image = torch.nn.Conv2d(base_channels, 1, 3, padding=1)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1_act(self.enc1_norm(self.enc1(image)))
        h = self.enc2_act(self.enc2_norm(self.enc2(h)))
        return self.to_mu(h), self.to_logvar(h).clamp(-8, 8)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.dec1_act(self.dec1_norm(self.dec1(latent)))
        h = self.dec2_act(self.dec2_norm(self.dec2(h)))
        return torch.tanh(self.to_image(h))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(image)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(latent), mu, logvar


def sinusoidal_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
    )
    phases = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((phases.sin(), phases.cos()), dim=-1)
    if width % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(group_count(in_channels), in_channels)
        self.act1 = SlavikA(in_channels, channel_dim=1)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = torch.nn.Linear(time_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(group_count(out_channels), out_channels)
        self.act2 = SlavikA(out_channels, channel_dim=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (torch.nn.Identity() if in_channels == out_channels
                     else torch.nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_embedding))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class SlavikUNet(torch.nn.Module):
    """Compact latent-space U-Net with timestep and digit-class conditioning."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 32,
                 num_classes: int = 10):
        super().__init__()
        time_channels = base_channels * 4
        self.num_classes = num_classes
        self.time_channels = time_channels
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(base_channels, time_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(time_channels, time_channels),
        )
        self.class_embedding = torch.nn.Embedding(num_classes + 1, time_channels)
        self.input = torch.nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        self.down_block = ResBlock(base_channels, base_channels, time_channels)
        self.downsample = torch.nn.Conv2d(
            base_channels, base_channels * 2, 3, stride=2, padding=1
        )
        self.middle1 = ResBlock(base_channels * 2, base_channels * 2, time_channels)
        self.middle2 = ResBlock(base_channels * 2, base_channels * 2, time_channels)
        self.up_project = torch.nn.Conv2d(base_channels * 2, base_channels, 1)
        self.up_block = ResBlock(base_channels * 2, base_channels, time_channels)
        self.output_norm = torch.nn.GroupNorm(group_count(base_channels), base_channels)
        self.output_act = SlavikA(base_channels, channel_dim=1)
        self.output = torch.nn.Conv2d(base_channels, latent_channels, 3, padding=1)

    def forward(self, latent: torch.Tensor, timesteps: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(sinusoidal_embedding(timesteps, self.time_channels // 4))
        time = time + self.class_embedding(labels)
        skip = self.down_block(self.input(latent), time)
        h = self.downsample(skip)
        h = self.middle2(self.middle1(h, time), time)
        h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = self.up_project(h)
        h = self.up_block(torch.cat((h, skip), dim=1), time)
        return self.output(self.output_act(self.output_norm(h)))


def diffusion_schedule(steps: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    if steps < 2:
        raise ValueError("Diffusion needs at least two timesteps")
    offset = 0.008
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar_curve = torch.cos(
        ((points / steps + offset) / (1 + offset)) * math.pi / 2
    ).square()
    alpha_bar_curve = alpha_bar_curve / alpha_bar_curve[0]
    betas = 1 - alpha_bar_curve[1:] / alpha_bar_curve[:-1]
    betas = betas.clamp(min=1e-5, max=0.999).to(device=device, dtype=torch.float32)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def checkpoint_load(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def train_autoencoder(args: argparse.Namespace, device: torch.device) -> None:
    model = SlavikAutoencoder(args.latent_channels, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume and args.checkpoint.exists():
        saved = checkpoint_load(args.checkpoint, device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"])
        print(f"Продолжаю с эпохи {start_epoch}", flush=True)

    train_loader = make_loader(args.data, True, args.limit, args.batch_size, True)
    validation_loader = make_loader(
        args.data, False, args.val_limit, args.batch_size, False
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for images, _ in train_loader:
            images = images.to(device)
            reconstruction, mu, logvar = model(images)
            reconstruction_loss = F.mse_loss(reconstruction, images)
            kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction_loss + args.kl_weight * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            total_items += len(images)

        model.eval()
        validation_loss = 0.0
        validation_items = 0
        with torch.no_grad():
            for images, _ in validation_loader:
                images = images.to(device)
                mu, _ = model.encode(images)
                loss = F.mse_loss(model.decode(mu), images)
                validation_loss += float(loss) * len(images)
                validation_items += len(images)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": {
                "latent_channels": args.latent_channels,
                "base_channels": args.base_channels,
            },
        }, args.checkpoint)
        print(
            f"AE epoch {epoch + 1}/{args.epochs} "
            f"train={total_loss / total_items:.5f} "
            f"val_mse={validation_loss / validation_items:.5f} "
            f"saved={args.checkpoint}",
            flush=True,
        )


@torch.no_grad()
def encode_training_latents(model: SlavikAutoencoder, loader: DataLoader,
                            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    latents, labels = [], []
    for images, batch_labels in loader:
        mu, _ = model.encode(images.to(device))
        latents.append(mu.cpu())
        labels.append(batch_labels)
    return torch.cat(latents), torch.cat(labels)


def train_diffusion(args: argparse.Namespace, device: torch.device) -> None:
    autoencoder_checkpoint = checkpoint_load(args.autoencoder, device)
    ae_config = autoencoder_checkpoint["config"]
    autoencoder = SlavikAutoencoder(**ae_config).to(device)
    autoencoder.load_state_dict(autoencoder_checkpoint["model"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    loader = make_loader(args.data, True, args.limit, args.batch_size, False)
    latents, labels = encode_training_latents(autoencoder, loader, device)
    model = SlavikUNet(
        ae_config["latent_channels"], args.base_channels
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    latent_mean = latents.mean().item()
    latent_std = latents.std().clamp_min(1e-5).item()
    if args.resume and args.checkpoint.exists():
        saved = checkpoint_load(args.checkpoint, device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"])
        latent_mean = float(saved["latent_mean"])
        latent_std = float(saved["latent_std"])
        print(f"Продолжаю с эпохи {start_epoch}", flush=True)
    latents = ((latents - latent_mean) / latent_std).to(device)
    labels = labels.to(device)
    betas, alphas, alpha_bars = diffusion_schedule(args.diffusion_steps, device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        permutation = torch.randperm(len(latents), device=device)
        total_loss = 0.0
        total_items = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            clean = latents[indices]
            condition = labels[indices].clone()
            condition[torch.rand(len(indices), device=device) < args.class_dropout] = NULL_CLASS
            timesteps = torch.randint(
                args.diffusion_steps, (len(indices),), device=device
            )
            noise = torch.randn_like(clean)
            alpha_bar = alpha_bars[timesteps, None, None, None]
            noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
            prediction = model(noisy, timesteps, condition)
            loss = F.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_items += len(indices)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "config": {
                "latent_channels": ae_config["latent_channels"],
                "base_channels": args.base_channels,
                "num_classes": 10,
                "diffusion_steps": args.diffusion_steps,
                "schedule": "cosine",
            },
        }, args.checkpoint)
        print(
            f"Diffusion epoch {epoch + 1}/{args.epochs} "
            f"noise_mse={total_loss / total_items:.5f} saved={args.checkpoint}",
            flush=True,
        )


@torch.no_grad()
def sample_latents(model: SlavikUNet, labels: torch.Tensor, steps: int,
                   guidance: float, device: torch.device) -> torch.Tensor:
    betas, alphas, alpha_bars = diffusion_schedule(steps, device)
    latent = torch.randn(
        len(labels), model.output.out_channels, 7, 7, device=device
    )
    null_labels = torch.full_like(labels, NULL_CLASS)
    model.eval()
    for index in reversed(range(steps)):
        timestep = torch.full((len(labels),), index, device=device, dtype=torch.long)
        if guidance == 1.0:
            predicted_noise = model(latent, timestep, labels)
        else:
            unconditional = model(latent, timestep, null_labels)
            conditional = model(latent, timestep, labels)
            predicted_noise = unconditional + guidance * (
                conditional - unconditional
            )
        beta = betas[index]
        alpha = alphas[index]
        alpha_bar = alpha_bars[index]
        latent = (latent - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        if index:
            alpha_bar_previous = alpha_bars[index - 1]
            posterior_variance = (
                beta * (1 - alpha_bar_previous) / (1 - alpha_bar)
            )
            latent = latent + posterior_variance.sqrt() * torch.randn_like(latent)
    return latent


def save_grid(images: torch.Tensor, labels: list[int], output: Path,
              columns: int = 5) -> None:
    images = images.detach().cpu().clamp(0, 1)
    cell = 32
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("L", (columns * cell, rows * (cell + 14)), color=24)
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        x = (index % columns) * cell
        y = (index // columns) * (cell + 14)
        pixels = (image[0].numpy() * 255).astype("uint8")
        tile = Image.fromarray(pixels, mode="L").resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y))
        draw.text((x + 12, y + cell), str(labels[index]), fill=255)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def generate(args: argparse.Namespace, device: torch.device) -> None:
    diffusion_checkpoint = checkpoint_load(args.checkpoint, device)
    config = diffusion_checkpoint["config"]
    autoencoder_checkpoint = checkpoint_load(args.autoencoder, device)
    ae = SlavikAutoencoder(**autoencoder_checkpoint["config"]).to(device)
    ae.load_state_dict(autoencoder_checkpoint["model"])
    ae.eval()
    model = SlavikUNet(
        config["latent_channels"], config["base_channels"], config["num_classes"]
    ).to(device)
    model.load_state_dict(diffusion_checkpoint["model"])
    model.eval()

    labels = (args.labels * math.ceil(args.num / len(args.labels)))[:args.num]
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    latent = sample_latents(
        model, label_tensor, config["diffusion_steps"], args.guidance, device
    )
    latent = latent * float(diffusion_checkpoint["latent_std"])
    latent = latent + float(diffusion_checkpoint["latent_mean"])
    images = (ae.decode(latent) + 1) / 2
    save_grid(images, labels, args.output, args.columns)
    print(f"Сэмплы сохранены: {args.output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST latent diffusion on SLA'vik cells (local, no downloads)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ae = subparsers.add_parser("train-ae", help="train the SLA'vik latent autoencoder")
    ae.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ae.add_argument("--checkpoint", type=Path, default=DEFAULT_MODELS / "slavik_mnist_ae.pt")
    ae.add_argument("--epochs", type=int, default=10)
    ae.add_argument("--limit", type=int, default=15_000)
    ae.add_argument("--val-limit", type=int, default=2_000)
    ae.add_argument("--batch-size", type=int, default=64)
    ae.add_argument("--latent-channels", type=int, default=4)
    ae.add_argument("--base-channels", type=int, default=32)
    ae.add_argument("--kl-weight", type=float, default=1e-5)
    ae.add_argument("--lr", type=float, default=1e-3)
    ae.add_argument("--seed", type=int, default=42)
    ae.add_argument("--device", default="auto")
    ae.add_argument("--resume", action="store_true")

    diffusion = subparsers.add_parser("train-diffusion", help="train the latent DDPM")
    diffusion.add_argument("--data", type=Path, default=DEFAULT_DATA)
    diffusion.add_argument("--autoencoder", type=Path, default=DEFAULT_MODELS / "slavik_mnist_ae.pt")
    diffusion.add_argument("--checkpoint", type=Path, default=DEFAULT_MODELS / "slavik_mnist_diffusion.pt")
    diffusion.add_argument("--epochs", type=int, default=20)
    diffusion.add_argument("--limit", type=int, default=15_000)
    diffusion.add_argument("--batch-size", type=int, default=64)
    diffusion.add_argument("--base-channels", type=int, default=32)
    diffusion.add_argument("--diffusion-steps", type=int, default=100)
    diffusion.add_argument("--class-dropout", type=float, default=0.1)
    diffusion.add_argument("--lr", type=float, default=2e-4)
    diffusion.add_argument("--seed", type=int, default=42)
    diffusion.add_argument("--device", default="auto")
    diffusion.add_argument("--resume", action="store_true")

    sample = subparsers.add_parser("sample", help="generate class-conditional digits")
    sample.add_argument("--autoencoder", type=Path, default=DEFAULT_MODELS / "slavik_mnist_ae.pt")
    sample.add_argument("--checkpoint", type=Path, default=DEFAULT_MODELS / "slavik_mnist_diffusion.pt")
    sample.add_argument("--labels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    sample.add_argument("--num", type=int, default=10)
    sample.add_argument("--columns", type=int, default=5)
    sample.add_argument("--guidance", type=float, default=2.0)
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument("--device", default="auto")
    sample.add_argument("--output", type=Path, default=ROOT / "outputs" / "slavik_mnist_samples.png")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "epochs") and args.epochs < 1:
        parser.error("--epochs must be positive")
    if hasattr(args, "limit") and args.limit < 0:
        parser.error("--limit must be zero (all data) or positive")
    if args.command == "train-diffusion" and not 0 <= args.class_dropout < 1:
        parser.error("--class-dropout must be in [0, 1)")
    if args.command == "train-diffusion" and args.diffusion_steps < 2:
        parser.error("--diffusion-steps must be at least 2")
    if args.command == "train-diffusion" and args.base_channels < 8:
        parser.error("--base-channels must be at least 8")
    if args.command in ("train-ae", "train-diffusion") and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.command == "sample":
        if args.num < 1 or args.columns < 1:
            parser.error("--num and --columns must be positive")
        if any(label < 0 or label > 9 for label in args.labels):
            parser.error("--labels must be digits 0 through 9")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    if args.command == "train-ae":
        train_autoencoder(args, device)
    elif args.command == "train-diffusion":
        train_diffusion(args, device)
    else:
        generate(args, device)


if __name__ == "__main__":
    main()
