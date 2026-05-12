import argparse
import os
import pickle

import numpy as np
import torch

from diffusion import sample
from eval_utils import extract_chunk, make_mask, metrics, save_viz
from new_model import DiT3D


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(run_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    a = ckpt.get("args", {})
    cs = a.get("chunk_size", 32)

    model = DiT3D(
        vocab_size=vocab["vocab_size"],
        embed_dim=a.get("embed_dim", 128),
        hidden_dim=a.get("hidden_dim", 512),
        patch_size=a.get("patch_size", 2),
        depth=a.get("dit_depth", 12),
        num_heads=a.get("dit_heads", 8),
        mlp_ratio=a.get("mlp_ratio", 4.0),
        time_dim=a.get("time_dim", 256),
        chunk_size=cs,
    ).to(device)

    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
    model.eval()
    return model, vocab, cs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--schematic", required=True)
    p.add_argument("--mask", default="non_air", choices=["non_air", "bottom", "top", "random", "shell"])
    p.add_argument("--cut_frac", type=float, default=0.5)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--output", default="eval.html")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, vocab, cs = load_model(args.checkpoint, device)
    block_to_idx = vocab["block_to_idx"]
    idx_to_block = vocab["idx_to_block"]

    print(f"Loading: {args.schematic}")
    from schematic_loader import load_any
    blocks_str = load_any(args.schematic)
    h, l, w = blocks_str.shape
    print(f"Schematic size: {h}×{l}×{w}  →  center-cropped to {cs}×{cs}×{cs}")

    chunk_str = extract_chunk(blocks_str, cs)
    unique, inverse = np.unique(chunk_str, return_inverse=True)
    original = np.array([block_to_idx.get(str(n), 0) for n in unique], dtype=np.int64)[inverse].reshape(chunk_str.shape)
    fill = (original != 0).mean()
    print(f"Fill ratio: {fill:.1%}  |  unique blocks: {len(np.unique(original))}")

    condition_mask = make_mask(cs, args.mask, args.cut_frac, indices=original)
    known_pct = condition_mask.mean()
    print(f"Mask: {args.mask}  given={known_pct:.0%}  to-reconstruct={1-known_pct:.0%}")

    orig_t = torch.from_numpy(original).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(condition_mask).unsqueeze(0).to(device)

    all_metrics = []
    last_result = None
    for run in range(args.n_runs):
        print(f"  Run {run+1}/{args.n_runs}...", end=" ", flush=True)
        with torch.no_grad():
            result_t = sample(model, orig_t, mask_t, num_steps=args.num_steps, temperature=args.temperature)
        result = result_t.squeeze(0).cpu().numpy()
        m = metrics(original, result, condition_mask)
        all_metrics.append(m)
        last_result = result
        print(f"rare={m['rare']:.3f}  non_air={m['non_air']:.3f}")

    print("\n--- Metrics at unknown positions ---")
    for key in ["exact", "non_air", "common", "rare", "hallucinated_frac", "missed_frac"]:
        vals = [m[key] for m in all_metrics]
        mean = np.mean(vals)
        if args.n_runs > 1:
            std = np.std(vals)
            print(f"  {key:<22} {mean:.3f} ± {std:.3f}")
        else:
            print(f"  {key:<22} {mean:.3f}")
    print(f"  {'n_unknown':<22} {all_metrics[0]['n_unknown']}")
    print(f"  {'n_non_air_unknown':<22} {all_metrics[0]['n_non_air_unknown']}")

    print(f"\nSaving viz: {args.output}")
    save_viz(original, condition_mask, last_result, args.output, idx_to_block)
    print("Done.")


if __name__ == "__main__":
    main()
