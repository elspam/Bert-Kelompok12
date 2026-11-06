import os
import time
import torch
from torch.amp import autocast, GradScaler

from training.utils import (
    build_optimizer,
    build_scheduler,
    compute_mlm_metrics,
    compute_perplexity,
)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    ignore_index: int = -100,
) -> dict:
    model.train()
    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_masked_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

        metrics = compute_mlm_metrics(logits.detach(), labels, ignore_index=ignore_index)
        n_masked = metrics["total_masked"]

        if n_masked > 0 and loss is not None and not torch.isnan(loss):
            if use_amp:
                scaler.scale(loss).backward()
                # PENTING: unscale_ WAJIB dipanggil sebelum clip_grad_norm_
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * n_masked
            total_top1_correct += metrics["top1_correct"]
            total_top5_correct += metrics["top5_correct"]
            total_masked_tokens += n_masked

    avg_loss = total_loss / max(1, total_masked_tokens)
    avg_top1_acc = total_top1_correct / max(1, total_masked_tokens)
    avg_top5_acc = total_top5_correct / max(1, total_masked_tokens)
    ppl = compute_perplexity(avg_loss)

    return {
        "loss": avg_loss,
        "acc": avg_top1_acc,
        "top5_acc": avg_top5_acc,
        "ppl": ppl,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
    ignore_index: int = -100,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_masked_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

        metrics = compute_mlm_metrics(logits, labels, ignore_index=ignore_index)
        n_masked = metrics["total_masked"]

        if n_masked > 0 and loss is not None and not torch.isnan(loss):
            total_loss += loss.item() * n_masked
            total_top1_correct += metrics["top1_correct"]
            total_top5_correct += metrics["top5_correct"]
            total_masked_tokens += n_masked

    avg_loss = total_loss / max(1, total_masked_tokens)
    avg_top1_acc = total_top1_correct / max(1, total_masked_tokens)
    avg_top5_acc = total_top5_correct / max(1, total_masked_tokens)
    ppl = compute_perplexity(avg_loss)

    return {
        "loss": avg_loss,
        "acc": avg_top1_acc,
        "top5_acc": avg_top5_acc,
        "ppl": ppl,
    }


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device = None,
    tag: str = "",
) -> dict:
    """
    Menjalankan full training loop dengan dukungan:
    - Early stopping (patience dari config["training"]["patience"])
    - Pencatatan Top-1 & Top-5 accuracy di history
    - Penyimpanan checkpoint terbaik ke checkpoints/best_model{tag}.pt
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    t_cfg = config.get("training", {})
    d_cfg = config.get("data", {})

    epochs = int(t_cfg.get("epochs", 100))
    max_grad_norm = float(t_cfg.get("max_grad_norm", 1.0))
    use_amp = bool(t_cfg.get("fp16", True)) and (device.type == "cuda")
    ignore_index = int(d_cfg.get("ignore_index", -100))

    # Konfigurasi Early Stopping
    patience = t_cfg.get("patience", None)
    patience = int(patience) if (patience is not None and int(patience) > 0) else None
    min_delta = float(t_cfg.get("min_delta", 1e-4))

    checkpoint_dir = t_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if tag and not tag.startswith("_"):
        tag = f"_{tag}"
    best_ckpt_path = os.path.join(checkpoint_dir, f"best_model{tag}.pt")

    total_steps = len(train_loader) * epochs
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps=total_steps)
    scaler = GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_top5_acc": [],
        "val_top5_acc": [],
        "train_ppl": [],
        "val_ppl": [],
        "lr": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
        "best_val_top5_acc": 0.0,
        "best_val_ppl": float("inf"),
        "stopped_early": False,
        "checkpoint_path": best_ckpt_path,
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(
        f"=== Mulai Training {tag or 'default'} | Device: {device} | "
        f"Pre-LN: {getattr(model, 'pre_norm', False)} | Epochs: {epochs} | "
        f"Steps/Epoch: {len(train_loader)} | Total Steps: {total_steps} | "
        f"Patience: {patience} ==="
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            ignore_index=ignore_index,
        )

        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=use_amp,
            ignore_index=ignore_index,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["train_acc"].append(train_res["acc"])
        history["val_acc"].append(val_res["acc"])
        history["train_top5_acc"].append(train_res["top5_acc"])
        history["val_top5_acc"].append(val_res["top5_acc"])
        history["train_ppl"].append(train_res["ppl"])
        history["val_ppl"].append(val_res["ppl"])
        history["lr"].append(current_lr)

        # Cek perbaikan Val Loss untuk simpan best_model & Early Stopping
        improved = val_res["loss"] < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_res["loss"]
            epochs_no_improve = 0
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_res["loss"]
            history["best_val_acc"] = val_res["acc"]
            history["best_val_top5_acc"] = val_res["top5_acc"]
            history["best_val_ppl"] = val_res["ppl"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_res["loss"],
                    "val_acc": val_res["acc"],
                    "val_top5_acc": val_res["top5_acc"],
                    "val_ppl": val_res["ppl"],
                    "config": config,
                },
                best_ckpt_path,
            )
            mark = " * [BEST]"
        else:
            epochs_no_improve += 1
            mark = ""

        print(
            f"Epoch {epoch:03d}/{epochs:03d} ({elapsed:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_res['loss']:.4f} Acc@1: {train_res['acc']*100:.2f}% Acc@5: {train_res['top5_acc']*100:.2f}% | "
            f"Val Loss: {val_res['loss']:.4f} Acc@1: {val_res['acc']*100:.2f}% Acc@5: {val_res['top5_acc']*100:.2f}% PPL: {val_res['ppl']:.1f}{mark}"
        )

        if patience is not None and epochs_no_improve >= patience:
            print(
                f"\n[Early Stopping] Val loss tidak membaik selama {patience} epoch. "
                f"Berhenti di epoch {epoch}. Best epoch: {history['best_epoch']} "
                f"(Val Acc@1: {history['best_val_acc']*100:.2f}%, Acc@5: {history['best_val_top5_acc']*100:.2f}%)."
            )
            history["stopped_early"] = True
            break

    # Muat ulang bobot terbaik ke model sebelum fungsi selesai
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    return history