#!/usr/bin/env python3
"""Minecraft Copilot inference server — runs as a subprocess, communicates via stdin/stdout.

Protocol (big-endian binary):
  stdin per request : n*8 bytes (int64 x_t) + n bytes (uint8 condMask)
  stdout on startup : b'READY\n'
  stdout per request: (NUM_STEPS+1) iterations of [4B uint32 len][n*8B int64 x]
"""
import sys, struct, json
import numpy as np
from pathlib import Path

config_dir = Path(__file__).parent
try:
    meta = json.loads((config_dir / 'meta.json').read_text())
except FileNotFoundError as e:
    sys.stderr.write(f'ERROR: {e}\n')
    sys.exit(1)

cs        = int(meta['chunk_size'])
n         = cs ** 3
vocab     = int(meta['vocab_size'])
mask_idx  = int(meta['mask_idx'])
NUM_STEPS = 20

try:
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(config_dir / 'model.onnx'),
        providers=['CPUExecutionProvider'],
    )
except Exception as e:
    sys.stderr.write(f'ERROR loading model: {e}\n')
    sys.exit(1)

rng   = np.random.default_rng()
stdin  = sys.stdin.buffer
stdout = sys.stdout.buffer

def read_exact(nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = stdin.read(nbytes - len(buf))
        if not chunk:
            raise EOFError()
        buf += chunk
    return bytes(buf)

def send_x(x: np.ndarray):
    out = x.astype(np.dtype('>i8')).tobytes()
    stdout.write(struct.pack('>I', len(out)))
    stdout.write(out)
    stdout.flush()

stdout.write(b'READY\n')
stdout.flush()

while True:
    try:
        raw_x = read_exact(n * 8)
        raw_m = read_exact(n)
    except EOFError:
        break

    cond_blocks = np.frombuffer(raw_x, dtype=np.dtype('>i8')).astype(np.int64)
    cond_mask   = np.frombuffer(raw_m, dtype=np.uint8).astype(bool)
    x = np.where(cond_mask, cond_blocks, mask_idx).astype(np.int64)

    last_logits = None
    for step in range(NUM_STEPS):
        t_now  = 1.0 - step / NUM_STEPS
        t_next = 1.0 - (step + 1) / NUM_STEPS

        logits = sess.run(None, {
            'x_t':            x.reshape(1, cs, cs, cs),
            'condition_mask': cond_mask.reshape(1, cs, cs, cs),
            't':              np.array([t_now], dtype=np.float32),
        })[0].reshape(vocab, n)
        last_logits = logits

        x0          = np.argmax(logits, axis=0).astype(np.int64)
        unmask_prob = (t_now - t_next) / t_now if t_now > 1e-6 else 1.0
        still_masked = (~cond_mask) & (x == mask_idx)
        x = np.where(still_masked & (rng.random(n) < unmask_prob), x0, x)

        send_x(x)

    # Final pass: argmax any remaining masked positions
    if last_logits is not None:
        still_masked = (~cond_mask) & (x == mask_idx)
        if still_masked.any():
            x = np.where(still_masked, np.argmax(last_logits, axis=0).astype(np.int64), x)
    send_x(x)
