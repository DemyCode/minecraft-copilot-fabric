import argparse
import contextlib
import copy
import csv
import os
import pickle
from datetime import datetime
from pathlib import Path

import schedulefree
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MinecraftDataset
from diffusion import compute_loss, compute_accuracy
from model import UNetTransformer
from new_model import DiT3D
from viz import save_sample_viz, save_3d_viz


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--writcache_dir", default="writcache")
    p.add_argument("--npy_dir", default="npy")
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--min_fill", type=float, default=0.02)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--val_fraction", type=float, default=0.05)
    # training
    p.add_argument("--batch_size", type=int, default=4,
                   help="micro-batch size (per device, per forward pass)")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="accumulate this many micro-batches before each optimizer step; "
                        "effective batch size = batch_size * grad_accum_steps")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--air_weight", type=float, default=0.5)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--max_steps", type=int, default=500_000)
    # model (shared)
    p.add_argument("--model_type", choices=["unet", "dit"], default="unet")
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--time_dim", type=int, default=384)
    # unet-specific
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument("--num_res_blocks", type=int, default=3)
    p.add_argument("--transformer_layers", type=int, default=16)
    p.add_argument("--transformer_heads", type=int, default=8)
    # dit-specific
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--dit_depth", type=int, default=12)
    p.add_argument("--dit_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    # infra
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    return p.parse_args()


def _do_optimizer_step(optim, scaler, model, grad_clip):
    if scaler:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
    optim.zero_grad(set_to_none=True)


def train_epoch(
    model, ema_model, loader, optim, scaler, device, args, step, epoch
):
    model.train()
    optim.train()
    total_loss = 0.0
    n = 0
    accum = args.grad_accum_steps
    pending = 0  # micro-steps accumulated since last optimizer step

    optim.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=f"  train", leave=False)
    for batch in bar:
        blocks = batch["blocks"].to(device, non_blocking=True)
        cond = batch["condition_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda") if scaler else contextlib.nullcontext():
            # Divide loss by accum so gradients sum to the correct average
            loss = compute_loss(model, blocks, cond, args.air_weight, valid) / accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum  # log the unscaled loss
        n += 1
        pending += 1

        if pending == accum:
            _do_optimizer_step(optim, scaler, model, args.grad_clip)
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1.0 - args.ema_decay)
            pending = 0
            step += 1
            bar.set_postfix(loss=f"{total_loss/n:.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

            if step >= args.max_steps:
                break

    # flush any remaining accumulated gradients at epoch end
    if pending > 0:
        _do_optimizer_step(optim, scaler, model, args.grad_clip)
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.lerp_(p, 1.0 - args.ema_decay)
        step += 1

    return total_loss / max(1, n), step


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    losses = []
    t_levels = (0.2, 0.4, 0.6, 0.8)
    buckets = {t: {"non_air": [], "common": [], "rare": []} for t in t_levels}

    bar = tqdm(loader, desc=f"  val  ", leave=False)
    for batch in bar:
        blocks = batch["blocks"].to(device, non_blocking=True)
        cond = batch["condition_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        losses.append(compute_loss(model, blocks, cond, args.air_weight, valid).item())
        bar.set_postfix(loss=f"{sum(losses)/len(losses):.4f}")

        for t, b in buckets.items():
            na, co, ra = compute_accuracy(model, blocks, cond, t)
            b["non_air"].append(na)
            b["common"].append(co)
            b["rare"].append(ra)

    def avg(lst):
        return sum(lst) / max(1, len(lst))

    return {
        "loss": avg(losses),
        **{f"t{t}": {k: avg(v) for k, v in b.items()} for t, b in buckets.items()},
    }


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MinecraftDataset(
        data_dirs=args.data_dirs,
        chunk_size=args.chunk_size,
        min_fill=args.min_fill,
        writcache_dir=args.writcache_dir,
        npy_dir=args.npy_dir,
        max_files=args.max_files,
    )
    n_val = max(1, int(args.val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_type == "dit":
        model = DiT3D(
            vocab_size=dataset.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            patch_size=args.patch_size,
            depth=args.dit_depth,
            num_heads=args.dit_heads,
            mlp_ratio=args.mlp_ratio,
            time_dim=args.time_dim,
            chunk_size=args.chunk_size,
        ).to(device)
    else:
        model = UNetTransformer(
            vocab_size=dataset.vocab_size,
            embed_dim=args.embed_dim,
            base_channels=args.base_channels,
            num_res_blocks=args.num_res_blocks,
            time_dim=args.time_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            chunk_size=args.chunk_size,
        ).to(device)
    print(f"Model: {args.model_type}  —  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Batch: micro={args.batch_size}  accum={args.grad_accum_steps}  effective={args.batch_size * args.grad_accum_steps}")

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False).eval()

    # model = torch.compile(model)

    optim = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "step", "train_loss", "val_loss"]
            + [f"{m}_t{t}" for t in (0.2, 0.4, 0.6, 0.8) for m in ("non_air", "common", "rare")]
        )

    with open(run_dir / "vocab.pkl", "wb") as f:
        pickle.dump(
            {
                "block_to_idx": dataset.block_to_idx,
                "idx_to_block": dataset.idx_to_block,
                "vocab_size": dataset.vocab_size,
                "mask_idx": dataset.mask_idx,
            },
            f,
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def save_checkpoint(current_step, current_epoch, name=None):
        tag = name or f"step{current_step:07d}"
        path = run_dir / f"ckpt_{tag}.pt"
        optim.eval()
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "step": current_step,
                "epoch": current_epoch,
                "vocab_size": dataset.vocab_size,
                "args": vars(args),
            },
            path,
        )
        optim.train()
        tqdm.write(f"  saved → {path}")

    # ── Resume ────────────────────────────────────────────────────────────────
    step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"New weights (random init): {missing}")
        if unexpected:
            print(f"Dropped weights: {unexpected}")
        ema_model.load_state_dict(ckpt.get("ema_model", ckpt["model"]), strict=False)
        step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        try:
            optim.load_state_dict(ckpt["optim"])
        except (ValueError, RuntimeError):
            print("Optimizer state incompatible — fresh optimizer")
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed from step {step}, epoch {start_epoch}")

    # ── Fixed viz batch (loaded once, stays constant) ─────────────────────────
    _viz_batch = next(
        iter(DataLoader(val_set, batch_size=3, shuffle=False, num_workers=0))
    )
    viz_blocks = _viz_batch["blocks"].to(device)
    viz_cond = _viz_batch["condition_mask"].to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch = start_epoch
    best_val_loss = float("inf")
    while step < args.max_steps:
        print(f"\nEpoch {epoch}  (step {step}/{args.max_steps})")

        train_loss, step = train_epoch(
            model,
            ema_model,
            train_loader,
            optim,
            scaler,
            device,
            args,
            step,
            epoch,
        )

        optim.eval()
        metrics = validate(ema_model, val_loader, device, args)
        optim.train()
        model.train()

        acc_lines = "  ".join(
            f"t={t}  non_air={metrics[f't{t}']['non_air']:.3f}"
            f"  common={metrics[f't{t}']['common']:.3f}"
            f"  rare={metrics[f't{t}']['rare']:.3f}"
            for t in (0.2, 0.4, 0.6, 0.8)
        )
        print(f"  train_loss={train_loss:.4f}  val_loss={metrics['loss']:.4f}\n  {acc_lines}")

        viz_path = str(run_dir / f"viz_epoch{epoch:03d}.png")
        viz3d_path = str(run_dir / f"viz3d_epoch{epoch:03d}.html")
        save_sample_viz(ema_model, viz_blocks[:2], viz_cond[:2], viz_path, step)
        save_3d_viz(
            ema_model, viz_blocks[:2], viz_cond[:2], viz3d_path, step, dataset.idx_to_block
        )
        tqdm.write(f"  viz → {Path(viz_path).name}  {Path(viz3d_path).name}")

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, step, round(train_loss, 6), round(metrics["loss"], 6)]
                + [
                    round(metrics[f"t{t}"][m], 6)
                    for t in (0.2, 0.4, 0.6, 0.8)
                    for m in ("non_air", "common", "rare")
                ]
            )

        save_checkpoint(step, epoch, name=f"epoch{epoch:03d}")

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            save_checkpoint(step, epoch, name="best")
            tqdm.write(f"  *** new best val_loss={best_val_loss:.4f} → ckpt_best.pt")

        epoch += 1

    save_checkpoint(step, epoch, name="final")
    print("Training complete.")


if __name__ == "__main__":
    main()
