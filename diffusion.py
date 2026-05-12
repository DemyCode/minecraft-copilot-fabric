import torch
import torch.nn.functional as F


def compute_loss(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    condition_mask: torch.Tensor,
    focal_gamma: float = 2.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    B = blocks.shape[0]
    device = blocks.device

    t = torch.rand(B, device=device)

    unknown = ~condition_mask
    absorb_prob = t[:, None, None, None].expand(blocks.shape)
    noise_mask = (torch.rand(blocks.shape, device=device) < absorb_prob) & unknown

    x_t = blocks.clone()
    x_t[noise_mask] = model.mask_idx

    logits = model(x_t, condition_mask, t)

    loss_mask = noise_mask & valid_mask if valid_mask is not None else noise_mask

    if not loss_mask.any():
        return logits.sum() * 0.0

    flat_logits = logits.permute(0, 2, 3, 4, 1)[loss_mask]
    flat_targets = blocks[loss_mask]

    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** focal_gamma * ce

    # Average within each sample first so every schematic contributes equally,
    # regardless of how many valid voxels it has (small padded schematics vs full chunks).
    batch_idx = torch.where(loss_mask)[0]
    per_sample_sum = torch.zeros(B, device=device).scatter_add(0, batch_idx, focal)
    per_sample_count = loss_mask.flatten(1).sum(1).float()
    valid = per_sample_count > 0
    return (per_sample_sum[valid] / per_sample_count[valid]).mean()


def _sample_steps(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int,
    temperature: float,
    t_start: float = 1.0,
):
    device = condition.device
    B = condition.shape[0]
    mask_idx = model.mask_idx
    vocab_size = model.vocab_size

    x = condition.clone()
    unknown = ~condition_mask
    if t_start >= 1.0:
        x[unknown] = mask_idx
    else:
        noise = torch.rand(x.shape, device=device)
        x[unknown & (noise < t_start)] = mask_idx

    t_steps = torch.linspace(t_start, 0.0, num_steps + 1, device=device)

    for step in range(num_steps):
        still_masked = (x == mask_idx) & ~condition_mask
        if not still_masked.any():
            break

        t_now = t_steps[step].item()
        t_next = t_steps[step + 1].item()

        t_tensor = torch.full((B,), t_now, device=device)
        logits = model(x, condition_mask, t_tensor)

        flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        if temperature != 1.0:
            flat_logits = flat_logits / temperature
        probs = F.softmax(flat_logits, dim=-1)
        x0 = torch.multinomial(probs, 1).squeeze(1).reshape(B, *condition.shape[1:])

        if t_now > 1e-6:
            unmask_prob = (t_now - t_next) / t_now
            should_unmask = (torch.rand(x.shape, device=device) < unmask_prob) & still_masked
        else:
            should_unmask = still_masked

        x[should_unmask] = x0[should_unmask]
        yield step, x

    remaining = (x == mask_idx) & ~condition_mask
    if remaining.any():
        t_zero = torch.zeros(B, device=device)
        logits = model(x, condition_mask, t_zero)
        x[remaining] = logits.argmax(dim=1)[remaining]

    yield -1, x


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 100,
    temperature: float = 1.0,
    t_start: float = 1.0,
) -> torch.Tensor:
    for _, x in _sample_steps(model, condition, condition_mask, num_steps, temperature, t_start):
        pass
    return x
