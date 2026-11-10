"""SLA'vik в компактной decoder-only языковой модели.

Сравнивает ReLU, GELU, SLA'vik-A и SLA'vik-R2 в FFN одного и того же
Transformer. Посимвольное авторегрессионное обучение на русскоязычных
технических отчётах проекта или любом другом UTF-8 тексте.

Пример:
    python3 slavik_llm.py --steps 300 --seeds 0 1 2 \
        --acts relu gelu slavik_a slavik_r2

Результаты сохраняются после каждого прогона в JSON. Скрипт рассчитан на CPU,
но автоматически использует CUDA, если она доступна.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slavik_activations import SlavikA, SlavikR2

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = ROOT / "slavik_llm_assets"
DEFAULT_OUTPUT = ROOT / "experiments" / "llm_pilot.json"


def build_corpus(output_path: Path) -> str:
    """Собирает только исходные исследовательские MD, без новых LLM-отчётов."""
    names = [
        "TLN_обоснование_эффективности.md",
        "исследование_своя_формула.md",
        "изобретение_нормированный_консенсус.md",
        "ускорение_нормированного_консенсуса.md",
        "SLAvik_модульные_клетки.md",
        "SLAvik_итоговый_отчёт.md",
        "SLAvik_A_новизна_и_перенос.md",
        "SLAvik_VAE_пилот_CIFAR10.md",
        "SLAvik_R2_превосходство_ReLU.md",
        "SLAvik_полный_отчёт.md",
        "SLAvik_руководство_для_ИИ_агентов.md",
    ]
    sections = []
    for name in names:
        path = ROOT / name
        if path.is_file():
            sections.append(path.read_text(encoding="utf-8"))
    text = "\n\n".join(sections)
    if len(text) < 20_000:
        raise RuntimeError("Недостаточно локального текста; передайте --text-file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, act: str):
        super().__init__()
        self.act = act
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        if act == "slavik_a":
            self.activation = SlavikA(d_ff)
        elif act == "slavik_r2":
            self.activation = SlavikR2(d_ff)
        elif act in ("relu", "gelu"):
            self.activation = torch.nn.Identity()
        else:
            raise ValueError(f"Неизвестная активация: {act}")
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = self.fc1(x)
        if self.act == "relu":
            h = torch.relu(h)
        elif self.act == "gelu":
            h = F.gelu(h)
        else:
            h = self.activation(h)
        return self.fc2(h)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model должно делиться на heads")
        self.heads = heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        bsz, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_width = width // self.heads

        def split(t):
            return t.reshape(bsz, length, self.heads, head_width).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().reshape(bsz, length, width)
        return self.proj(attended)


class Block(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, heads: int, act: str):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, act)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TinySLALM(torch.nn.Module):
    """Компактный Transformer LM; это пилот, а не современная большая LLM."""

    def __init__(self, vocab_size: int, act: str, context: int = 64,
                 d_model: int = 64, d_ff: int = 128, layers: int = 2,
                 heads: int = 4):
        super().__init__()
        self.context = context
        self.token = torch.nn.Embedding(vocab_size, d_model)
        self.position = torch.nn.Embedding(context, d_model)
        self.blocks = torch.nn.ModuleList([
            Block(d_model, d_ff, heads, act) for _ in range(layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model)
        # Выходная проекция связана с embedding: меньше параметров.
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        _, length = tokens.shape
        if length > self.context:
            raise ValueError("Длина последовательности превышает context")
        pos = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.position(pos)[None]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = F.linear(x, self.token.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, length=240, temperature=0.8, top_k=20):
        self.eval()
        for _ in range(length):
            context = tokens[:, -self.context:]
            logits, _ = self(context)
            logits = logits[:, -1] / temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


class CharData:
    def __init__(self, text: str, train_chars: int, val_chars: int, device):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        encoded = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n_train = min(train_chars, len(encoded) - val_chars - 1)
        if n_train < 10_000:
            raise ValueError("Слишком мало train-символов")
        n_val = min(val_chars, len(encoded) - n_train - 1)
        self.train = torch.as_tensor(encoded[:n_train], dtype=torch.long, device=device)
        self.val = torch.as_tensor(encoded[n_train:n_train + n_val + 1],
                                   dtype=torch.long, device=device)
        self.vocab_size = len(chars)

    def batch(self, split: str, batch_size: int, context: int, generator):
        data = self.train if split == "train" else self.val
        offsets = torch.randint(0, len(data) - context - 1,
                                (batch_size,), generator=generator,
                                device="cpu").to(data.device)
        x = torch.stack([data[i:i + context] for i in offsets])
        y = torch.stack([data[i + 1:i + context + 1] for i in offsets])
        return x, y

    def decode(self, tokens):
        return "".join(self.itos[int(i)] for i in tokens)


def evaluate(model, data: CharData, batch_size: int, context: int,
             batches: int, generator):
    was_training = model.training
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = data.batch("val", batch_size, context, generator)
            logits, loss = model(x, y)
            losses.append(float(loss))
            correct += int((logits.argmax(-1) == y).sum())
            total += y.numel()
    if was_training:
        model.train()
    mean_loss = float(np.mean(losses))
    return {"val_loss": mean_loss, "val_ppl": math.exp(mean_loss),
            "val_token_accuracy": correct / total}


def train_one(act: str, seed: int, text: str, args, device) -> dict:
    # Одинаковая инициализация общих весов у всех активаций.
    torch.manual_seed(seed)
    data = CharData(text, args.train_chars, args.val_chars, device)
    model = TinySLALM(data.vocab_size, act, args.context, args.d_model,
                      args.d_ff, args.layers, args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    train_model = torch.compile(model, dynamic=None) if args.compile else model
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    eval_gen = torch.Generator(device="cpu").manual_seed(123_456)
    # Финальная метрика всегда получается на одних и тех же батчах,
    # независимо от периодичности промежуточной оценки и числа шагов.
    final_eval_gen = torch.Generator(device="cpu").manual_seed(987_654)
    prompt_gen_state = torch.get_rng_state()

    history = []
    total_start = time.perf_counter()
    train_seconds = 0.0
    first_step_seconds = 0.0
    for step in range(args.steps + 1):
        fraction = step / max(args.steps, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step % args.eval_every == 0 or step == args.steps:
            is_final = step == args.steps
            metrics = evaluate(
                model, data, args.batch_size, args.context,
                args.final_eval_batches if is_final else args.eval_batches,
                final_eval_gen if is_final else eval_gen)
            history.append({"step": step, **metrics})
            print(json.dumps({"act": act, "seed": seed, **history[-1]}), flush=True)
        if step == args.steps:
            break
        step_start = time.perf_counter()
        x, y = data.batch("train", args.batch_size, args.context, batch_gen)
        _, loss = train_model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_seconds = time.perf_counter() - step_start
        if step == 0:
            first_step_seconds = step_seconds
        train_seconds += step_seconds
    total_seconds = time.perf_counter() - total_start

    # Генерация начинается с одинакового RNG и одного промпта.
    torch.set_rng_state(prompt_gen_state)
    prompt = args.prompt
    tokens = torch.tensor([[data.stoi.get(c, 0) for c in prompt]],
                          dtype=torch.long, device=device)
    generated = model.generate(tokens, length=args.generate_chars,
                               temperature=0.8, top_k=20)
    sample = data.decode(generated[0].cpu())
    params = sum(p.numel() for p in model.parameters())
    return {"act": act, "seed": seed, "steps": args.steps,
            "parameters": params, "vocab_size": data.vocab_size,
            "train_chars": len(data.train), "val_chars": len(data.val) - 1,
            "train_seconds": train_seconds,
            "first_step_seconds": first_step_seconds,
            "steady_state_train_seconds": train_seconds - first_step_seconds,
            "total_seconds_with_eval": total_seconds,
            "compiled": args.compile,
            "train_tokens_per_second": args.steps * args.batch_size * args.context / train_seconds,
            "steady_state_tokens_per_second": ((args.steps - 1) * args.batch_size * args.context
                                                / max(train_seconds - first_step_seconds, 1e-9)),
            "final": history[-1], "history": history, "sample": sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--acts", nargs="+",
                        default=["relu", "gelu", "slavik_a", "slavik_r2"])
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--train-chars", type=int, default=240_000)
    parser.add_argument("--val-chars", type=int, default=24_000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=0.0001)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--final-eval-batches", type=int, default=32)
    parser.add_argument("--generate-chars", type=int, default=240)
    parser.add_argument("--prompt", default="Нейронная сеть")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        corpus_path = DEFAULT_CORPUS_DIR / "technical_russian.txt"
        text = (corpus_path.read_text(encoding="utf-8") if corpus_path.exists()
                else build_corpus(corpus_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.output.exists():
        rows = json.loads(args.output.read_text(encoding="utf-8"))
    done = {(r["act"], r["seed"], r["steps"]) for r in rows}
    config = {**vars(args), "text_file": str(args.text_file),
              "output": str(args.output), "device": str(device),
              "torch": torch.__version__, "corpus_chars": len(text)}
    args.output.with_suffix(".config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for seed in args.seeds:
        for act in args.acts:
            if (act, seed, args.steps) in done:
                print(f"Пропуск готового прогона: {act}, seed={seed}", flush=True)
                continue
            row = train_one(act, seed, text, args, device)
            rows.append(row)
            args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            print("SAVED", act, seed, row["final"], flush=True)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
