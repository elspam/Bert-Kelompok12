import math
import random
import os
import yaml
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_optimizer(model: torch.nn.Module, config: dict) -> AdamW:
    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("lr", t_cfg.get("learning_rate", 8e-4)))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    betas = tuple(t_cfg.get("betas", (0.9, 0.999)))
    eps = float(t_cfg.get("eps", 1e-8))

    # Pisahkan parameter yang dikenakan weight decay dan yang tidak (bias & LayerNorm)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm1.weight", "norm2.weight", "final_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(optimizer_grouped_parameters, lr=lr, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    total_steps: int,
    warmup_steps: int = None,
) -> LambdaLR:
    """
    Membangun LR scheduler (cosine / linear) dengan prioritas warmup_steps absolut
    di atas warmup_ratio jika diisi.
    """
    t_cfg = config.get("training", {})
    scheduler_type = str(t_cfg.get("scheduler", "cosine")).lower()
    min_lr_ratio = float(t_cfg.get("min_lr_ratio", 0.05))  # Jaga agar LR ujung tidak mati ke 0

    # Prioritas 1: argumen fungsi warmup_steps
    # Prioritas 2: config["training"]["warmup_steps"]
    # Prioritas 3: config["training"]["warmup_ratio"] * total_steps
    cfg_warmup_steps = t_cfg.get("warmup_steps", None)
    if warmup_steps is not None and int(warmup_steps) > 0:
        num_warmup_steps = int(warmup_steps)
    elif cfg_warmup_steps is not None and int(cfg_warmup_steps) > 0:
        num_warmup_steps = int(cfg_warmup_steps)
    else:
        warmup_ratio = float(t_cfg.get("warmup_ratio", 0.06))
        num_warmup_steps = max(1, int(total_steps * warmup_ratio))

    # Pastikan warmup_steps tidak melampaui 50% total_steps pada eksperimen pendek
    num_warmup_steps = min(num_warmup_steps, max(1, int(total_steps * 0.5)))

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, total_steps - num_warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        if scheduler_type == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        else:
            # Linear decay
            return max(min_lr_ratio, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def compute_mlm_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> dict:
    """
    Menghitung metrik MLM pada posisi aktif (labels != ignore_index):
    - top1_correct, top5_correct, total_masked
    - accuracy (Top-1) & top5_accuracy (Top-5)
    """
    active_mask = labels != ignore_index
    total_masked = int(active_mask.sum().item())

    if total_masked == 0:
        return {
            "accuracy": 0.0,
            "top1_acc": 0.0,
            "top5_acc": 0.0,
            "top1_correct": 0,
            "top5_correct": 0,
            "total_masked": 0,
        }

    active_logits = logits[active_mask]  # (N_masked, V)
    active_labels = labels[active_mask]  # (N_masked,)

    # Top-1
    top1_preds = active_logits.argmax(dim=-1)
    top1_correct = int((top1_preds == active_labels).sum().item())

    # Top-5
    k = min(5, active_logits.size(-1))
    top5_preds = active_logits.topk(k, dim=-1).indices  # (N_masked, 5)
    top5_correct = int(
        (top5_preds == active_labels.unsqueeze(-1)).any(dim=-1).sum().item()
    )

    top1_acc = top1_correct / total_masked
    top5_acc = top5_correct / total_masked

    return {
        "accuracy": top1_acc,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "total_masked": total_masked,
    }


def compute_perplexity(loss: float) -> float:
    try:
        return float(math.exp(min(loss, 20.0)))
    except OverflowError:
        return float("inf")