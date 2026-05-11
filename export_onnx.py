"""
Export a training checkpoint to ONNX + the vocab/meta JSON files expected by the Java mod.

Output directory will contain:
  model.onnx       — the EMA model
  vocab.json       — {idx_to_block: {str→str}, block_to_idx: {str→int}}
  meta.json        — {vocab_size, mask_idx, chunk_size}

Usage:
  uv run python export_onnx.py --checkpoint runs/.../ckpt_best.pt --output_dir onnx_export
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import torch

from model import UNetTransformer
from new_model import DiT3D


def _load(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with open(Path(checkpoint_path).parent / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    a = ckpt.get("args", {})
    cs = a.get("chunk_size", 32)
    if a.get("model_type", "unet") == "dit":
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
    else:
        model = UNetTransformer(
            vocab_size=vocab["vocab_size"],
            embed_dim=a.get("embed_dim", 192),
            base_channels=a.get("base_channels", 96),
            num_res_blocks=a.get("num_res_blocks", 3),
            time_dim=a.get("time_dim", 384),
            transformer_layers=a.get("transformer_layers", 16),
            transformer_heads=a.get("transformer_heads", 8),
            chunk_size=cs,
        ).to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
    model.eval()
    return model, vocab, cs


def export(checkpoint_path: str, output_dir: str, opset: int = 17):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, vocab, cs = _load(checkpoint_path, torch.device("cpu"))
    model.eval()

    # ── vocab.json ────────────────────────────────────────────────────────────
    # Java BlockMapper expects:
    #   idx_to_block: Map<String, String>  (index as string key → block name)
    #   block_to_idx: Map<String, Integer> (block name → index)
    vocab_json = {
        "idx_to_block": {str(k): v for k, v in vocab["idx_to_block"].items()},
        "block_to_idx": vocab["block_to_idx"],
    }
    vocab_path = out / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_json))
    print(f"vocab.json  → {len(vocab_json['block_to_idx'])} block types")

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta = {
        "vocab_size": vocab["vocab_size"],
        "mask_idx":   vocab["mask_idx"],
        "chunk_size": cs,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"meta.json   → vocab_size={meta['vocab_size']}  mask_idx={meta['mask_idx']}  chunk_size={cs}")

    # ── model.onnx ────────────────────────────────────────────────────────────
    # Dummy inputs matching the Java OnnxSession.forward signature:
    #   x_t            : long   [1, cs, cs, cs]
    #   condition_mask : bool   [1, cs, cs, cs]
    #   t              : float  [1]
    x_t   = torch.zeros(1, cs, cs, cs, dtype=torch.long)
    cond  = torch.zeros(1, cs, cs, cs, dtype=torch.bool)
    t     = torch.tensor([0.5])

    onnx_path = str(out / "model.onnx")
    print("Exporting ONNX… (this may take a minute)")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x_t, cond, t),
            onnx_path,
            input_names=["x_t", "condition_mask", "t"],
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={
                "x_t":            {0: "batch"},
                "condition_mask": {0: "batch"},
                "t":              {0: "batch"},
                "logits":         {0: "batch"},
            },
        )

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"model.onnx  → {onnx_path}  ({size_mb:.1f} MB)")
    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to .pt checkpoint file")
    p.add_argument("--output_dir", required=True, help="directory for ONNX + JSON outputs")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = p.parse_args()
    export(args.checkpoint, args.output_dir, args.opset)


if __name__ == "__main__":
    main()
