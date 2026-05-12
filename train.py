import argparse
import contextlib
import copy
import csv
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import schedulefree
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MinecraftDataset
from diffusion import compute_loss
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
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="micro-batch size (per device, per forward pass)",
    )
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="accumulate this many micro-batches before each optimizer step; "
        "effective batch size = batch_size * grad_accum_steps",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--max_steps", type=int, default=500_000)
    # model
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--dit_depth", type=int, default=12)
    p.add_argument("--dit_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    # infra
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--amp_dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="bf16 skips GradScaler (stable with schedulefree); fp16 uses GradScaler",
    )
    return p.parse_args()


def _do_optimizer_step(optim, scaler, model, grad_clip):
    """Returns True if the optimizer step was taken (False means NaN/inf grads, step skipped)."""
    if scaler:
        scaler.unscale_(optim)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        if scaler:
            scaler.update()  # update scale factor downward
        optim.zero_grad(set_to_none=True)
        return False
    if scaler:
        scaler.step(optim)
        scaler.update()
    else:
        optim.step()
    optim.zero_grad(set_to_none=True)
    return True


def train_epoch(
    model,
    ema_model,
    loader,
    optim,
    scaler,
    device,
    args,
    step,
    use_amp=False,
    amp_dtype=torch.bfloat16,
):
    model.train()
    optim.train()
    total_loss = 0.0
    n = 0
    skipped = 0
    accum = args.grad_accum_steps
    pending = 0  # micro-steps accumulated since last optimizer step

    optim.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=f"  train", leave=False)
    for batch in bar:
        blocks = batch["blocks"].to(device, non_blocking=True)
        cond = batch["condition_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        with (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        ):
            loss = compute_loss(model, blocks, cond, args.focal_gamma, valid) / accum

        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            optim.zero_grad(set_to_none=True)
            pending = 0
            continue

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum  # log the unscaled loss
        n += 1
        pending += 1

        if pending == accum:
            step_taken = _do_optimizer_step(optim, scaler, model, args.grad_clip)
            if step_taken:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.lerp_(p, 1.0 - args.ema_decay)
            else:
                skipped += 1
            pending = 0
            step += 1
            bar.set_postfix(
                loss=f"{loss.item() * accum:.4f}",
                avg=f"{total_loss / n:.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )

            if step >= args.max_steps:
                break

    # flush any remaining accumulated gradients at epoch end
    if pending > 0:
        step_taken = _do_optimizer_step(optim, scaler, model, args.grad_clip)
        if step_taken:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1.0 - args.ema_decay)
        step += 1

    if skipped:
        tqdm.write(f"  WARNING: {skipped} NaN/inf steps skipped this epoch")
    return total_loss / max(1, n), step


@torch.no_grad()
def validate(model, loader, device, args, use_amp=False, amp_dtype=torch.bfloat16):
    model.eval()
    loss_sum = 0.0
    n_losses = 0
    acc_sum = 0.0
    n_acc = 0
    vocab_size = model.vocab_size
    correct = torch.zeros(vocab_size, device=device)
    total = torch.zeros(vocab_size, device=device)

    non_air_idx = torch.arange(vocab_size, device=device) > 0

    bar = tqdm(loader, desc="  val  ", leave=False)
    for batch in bar:
        blocks = batch["blocks"].to(device, non_blocking=True)
        cond = batch["condition_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        with (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        ):
            loss_sum += compute_loss(model, blocks, cond, args.focal_gamma, valid).item()
        n_losses += 1

        # Per-block accuracy at t=0.5 with fixed seed for a reproducible metric
        B = blocks.shape[0]
        t_tensor = torch.full((B,), 0.5, device=device)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(0)
            noise_mask = (torch.rand(blocks.shape, device=device) < 0.5) & ~cond

        x_t = blocks.clone()
        x_t[noise_mask] = model.mask_idx
        with (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        ):
            preds = model(x_t, cond, t_tensor).argmax(dim=1)

        eval_mask = noise_mask & valid
        batch_acc = 0.0
        if eval_mask.any():
            true_b = blocks[eval_mask]
            hits = (preds[eval_mask] == true_b).float()
            correct.scatter_add_(0, true_b, hits)
            total.scatter_add_(0, true_b, torch.ones(len(true_b), device=device))
            non_air_eval = eval_mask & (blocks != 0)
            if non_air_eval.any():
                batch_acc = (preds[non_air_eval] == blocks[non_air_eval]).float().mean().item()
                acc_sum += batch_acc
                n_acc += 1

        bar.set_postfix(
            loss=f"{loss_sum / n_losses:.4f}",
            acc=f"{batch_acc:.4f}",
            avg_acc=f"{acc_sum / max(1, n_acc):.4f}",
        )

    return {
        "loss":      loss_sum / max(1, n_losses),
        "macro_acc": acc_sum / max(1, n_acc),
        "n_covered": int((total[non_air_idx] > 0).sum().item()),
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
    print(
        f"Model: DiT3D  —  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    print(
        f"Batch: micro={args.batch_size}  accum={args.grad_accum_steps}  effective={args.batch_size * args.grad_accum_steps}"
    )

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False).eval()

    # model = torch.compile(model)

    optim = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    # bf16 has sufficient dynamic range and does not need a GradScaler;
    # GradScaler + schedulefree interact badly (skipped steps corrupt y-buffer)
    scaler = (
        torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    )

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "step", "train_loss", "val_loss", "macro_acc", "n_covered"]
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
        try:
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
        finally:
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

    # ── Training summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Run dir        : {run_dir}")
    if args.resume:
        print(f"  Resumed from   : {args.resume}")
        print(f"  Starting step  : {step}  /  {args.max_steps}")
        print(f"  Starting epoch : {start_epoch}")
    else:
        print(f"  Training from scratch")
    print(f"  Dataset        : {n_train} train  /  {len(val_set)} val  ({len(dataset)} total)")
    print(f"  Vocab size     : {dataset.vocab_size} block types")
    print(f"  Chunk size     : {args.chunk_size}³")
    print(f"  Model          : DiT3D  depth={args.dit_depth}  heads={args.dit_heads}  "
          f"hidden={args.hidden_dim}  patch={args.patch_size}  "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    print(f"  Batch          : micro={args.batch_size}  accum={args.grad_accum_steps}  "
          f"effective={args.batch_size * args.grad_accum_steps}")
    print(f"  Optimizer      : AdamWScheduleFree  lr={args.lr}  wd={args.weight_decay}  "
          f"warmup={args.warmup_steps}")
    print(f"  Loss           : focal  gamma={args.focal_gamma}")
    print(f"  EMA decay      : {args.ema_decay}")
    amp_str = f"{args.amp_dtype.upper()}" if use_amp else "disabled"
    print(f"  AMP            : {amp_str}")
    print("=" * 60)
    print()

    run_config = {
        "run_dir": str(run_dir),
        "resumed_from": args.resume,
        "start_step": step,
        "start_epoch": start_epoch,
        "max_steps": args.max_steps,
        "dataset": {
            "dirs": args.data_dirs,
            "n_train": n_train,
            "n_val": len(val_set),
            "vocab_size": dataset.vocab_size,
            "chunk_size": args.chunk_size,
            "min_fill": args.min_fill,
        },
        "model": {
            "type": "DiT3D",
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "time_dim": args.time_dim,
            "patch_size": args.patch_size,
            "depth": args.dit_depth,
            "heads": args.dit_heads,
            "mlp_ratio": args.mlp_ratio,
        },
        "training": {
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "focal_gamma": args.focal_gamma,
            "ema_decay": args.ema_decay,
            "amp": use_amp,
            "amp_dtype": args.amp_dtype if use_amp else None,
        },
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    # ── Fixed viz batch (loaded once, stays constant) ─────────────────────────
    _viz_batch = next(
        iter(DataLoader(val_set, batch_size=3, shuffle=False, num_workers=0))
    )
    viz_blocks = _viz_batch["blocks"].to(device)
    viz_cond = _viz_batch["condition_mask"].to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch = start_epoch
    best_macro_acc = 0.0
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
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        optim.eval()
        metrics = validate(ema_model, val_loader, device, args, use_amp=use_amp, amp_dtype=amp_dtype)
        optim.train()
        model.train()

        print(
            f"  train_loss={train_loss:.4f}  val_loss={metrics['loss']:.4f}"
            f"  macro_acc={metrics['macro_acc']:.4f}  ({metrics['n_covered']} blocks covered)"
        )

        viz_path = str(run_dir / f"viz_epoch{epoch:03d}.png")
        viz3d_path = str(run_dir / f"viz3d_epoch{epoch:03d}.html")
        save_sample_viz(ema_model, viz_blocks[:2], viz_cond[:2], viz_path, step)
        save_3d_viz(
            ema_model,
            viz_blocks[:2],
            viz_cond[:2],
            viz3d_path,
            step,
            dataset.idx_to_block,
        )
        tqdm.write(f"  viz → {Path(viz_path).name}  {Path(viz3d_path).name}")

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    step,
                    round(train_loss, 6),
                    round(metrics["loss"], 6),
                    round(metrics["macro_acc"], 6),
                    metrics["n_covered"],
                ]
            )

        save_checkpoint(step, epoch, name=f"epoch{epoch:03d}")

        if metrics["macro_acc"] > best_macro_acc:
            best_macro_acc = metrics["macro_acc"]
            save_checkpoint(step, epoch, name="best")
            tqdm.write(f"  *** new best macro_acc={best_macro_acc:.4f} → ckpt_best.pt")

        epoch += 1

    save_checkpoint(step, epoch, name="final")
    print("Training complete.")


if __name__ == "__main__":
    main()
