from __future__ import annotations
"""[BIMA] MLM accuracy, perplexity, dan loss tracker."""

"""
training/metrics.py
Perhitungan metrik evaluasi Masked Language Modeling (MLM) dari nol:
- Top-1 Masked Token Accuracy (hanya pada posisi labels != ignore_index)
- Perplexity (exp(loss)) dengan pengaman overflow
- EpochMetricTracker untuk akumulasi metrik eksak per token sepanjang epoch
"""



import math
import torch


def compute_mlm_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[int, int, float]:
    """
    Menghitung akurasi prediksi top-1 pada posisi token yang di-mask.

    Args:
        logits: Tensor float berukuran (B, L, V)
        labels: Tensor long berukuran (B, L) dengan nilai ignore_index (-100) pada token non-mask
        ignore_index: Nilai label yang diabaikan (default: -100 sesuai SPEC.md)

    Returns:
        Tuple (num_correct, num_masked_tokens, accuracy_float)
    """
    with torch.no_grad():
        mask = labels != ignore_index  # (B, L) boolean
        num_masked = int(mask.sum().item())

        if num_masked == 0:
            return 0, 0, 0.0

        predictions = logits.argmax(dim=-1)  # (B, L)
        num_correct = int((predictions[mask] == labels[mask]).sum().item())
        accuracy = float(num_correct) / float(num_masked)
        return num_correct, num_masked, accuracy


def compute_perplexity(avg_loss: float, max_loss_cap: float = 20.0) -> float:
    """
    Menghitung Perplexity = exp(avg_loss).
    Dilengkapi batas atas (max_loss_cap=20.0) untuk mencegah OverflowError
    ketika loss masih tinggi di awal pelatihan.
    """
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        return float("inf")
    safe_loss = min(float(avg_loss), max_loss_cap)
    return math.exp(safe_loss)


class EpochMetricTracker:
    """
    Pelacak metrik selama 1 epoch penuh.
    Mengakumulasi loss secara tertimbang terhadap jumlah token ter-mask (num_masked)
    agar rata-rata epoch akurat dan tidak bias terhadap batch yang memiliki sedikit mask.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.total_weighted_loss: float = 0.0
        self.total_correct: int = 0
        self.total_masked_tokens: int = 0
        self.valid_batches: int = 0
        self.skipped_batches: int = 0

    def update(
        self,
        loss_val: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[int, int, float]:
        """
        Memperbarui statistik dengan hasil satu batch.
        Mengembalikan (num_correct, num_masked, batch_acc).
        """
        num_correct, num_masked, batch_acc = compute_mlm_accuracy(
            logits, labels, ignore_index=self.ignore_index
        )

        if num_masked == 0 or math.isnan(loss_val) or math.isinf(loss_val):
            self.skipped_batches += 1
            return 0, 0, 0.0

        self.total_weighted_loss += float(loss_val) * num_masked
        self.total_correct += num_correct
        self.total_masked_tokens += num_masked
        self.valid_batches += 1
        return num_correct, num_masked, batch_acc

    def compute(self) -> dict[str, float | int]:
        """Menghasilkan ringkasan metrik di akhir epoch."""
        if self.total_masked_tokens == 0:
            avg_loss = 0.0
            accuracy = 0.0
            perplexity = float("inf")
        else:
            avg_loss = self.total_weighted_loss / float(self.total_masked_tokens)
            accuracy = float(self.total_correct) / float(self.total_masked_tokens)
            perplexity = compute_perplexity(avg_loss)

        return {
            "loss": round(avg_loss, 6),
            "accuracy": round(accuracy, 6),
            "perplexity": round(perplexity, 4),
            "masked_tokens": self.total_masked_tokens,
            "valid_batches": self.valid_batches,
            "skipped_batches": self.skipped_batches,
        }