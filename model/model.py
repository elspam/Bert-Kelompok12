"""[LUHUNG] Arsitektur BERT-from-Scratch untuk Masked Language Modeling (MLM).

Mengacu pada kontrak spesifikasi di SPEC.md v1.0:
- Komponen utama: MultiHeadAttention, EncoderLayer, BERTEncoder, MLMHead, BERT
- Kontrak input:
    input_ids      : LongTensor (B, L)
    attention_mask : LongTensor (B, L) — 1 = token valid, 0 = padding
    token_type_ids : LongTensor (B, L) — opsional, default 0
    labels         : LongTensor (B, L) — opsional, -100 pada posisi non-mask
- Kontrak output:
    logits         : FloatTensor (B, L, V)
    loss           : FloatTensor () scalar jika labels diberikan, selain itu None
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> nn.Module:
    """Mengembalikan modul fungsi aktivasi berdasarkan nama konfigurasi."""
    act = name.lower()
    if act == "gelu":
        return nn.GELU()
    if act == "relu":
        return nn.ReLU()
    raise ValueError(f"Fungsi aktivasi tidak didukung: {name!r}. Gunakan 'gelu' atau 'relu'.")


class MLMOutput(NamedTuple):
    """Container output BERT MLM yang kompatibel dengan tuple unpacking, atribut, maupun dict key.

    Contoh pemakaian di train.py:
        - Atribut : out = model(...); out.logits, out.loss
        - Unpack  : logits, loss = model(...)
        - Dict    : out["logits"], out["loss"]
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None

    def __getitem__(self, item: Union[int, slice, str]) -> Any:
        if isinstance(item, str):
            if item == "logits":
                return self.logits
            if item == "loss":
                return self.loss
            raise KeyError(f"Key {item!r} tidak ditemukan pada MLMOutput.")
        return super().__getitem__(item)


class BERTEmbeddings(nn.Module):
    """Gabungan Token Embedding + Learned Positional Embedding + Segment Embedding."""

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 256,
        max_seq_length: int = 128,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # Buffer position_ids (1, max_seq_length) agar otomatis ikut pindah device
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_length, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"Panjang sequence ({seq_len}) melebihi max_seq_length "
                f"({self.position_embeddings.num_embeddings})."
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = self.position_ids[:, :seq_len]

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + type_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttention(nn.Module):
    """Bidirectional Multi-Head Self-Attention untuk encoder BERT."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) harus habis dibagi num_heads ({num_heads})."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Ubah shape dari (B, L, H) menjadi (B, num_heads, L, head_dim)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        # Skor attention dua arah: (B, num_heads, L, L)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            # Terima mask (B, L) maupun yang sudah di-broadcast (B, 1, 1, L)
            if attention_mask.dim() == 2:
                expanded_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                expanded_mask = attention_mask
            else:
                raise ValueError(
                    f"attention_mask harus berdimensi 2 (B, L) atau 4 (B, 1, 1, L), "
                    f"diterima shape {tuple(attention_mask.shape)}."
                )

            # Gunakan nilai minimum sesuai dtype agar aman pada FP16 (AMP) maupun FP32
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(expanded_mask == 0, mask_value)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        output = self.out_proj(context)
        return output


class EncoderLayer(nn.Module):
    """Satu blok Transformer Encoder (Post-LN sesuai resep BERT asli)."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = _get_activation(activation)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Sub-layer 1: Bidirectional Multi-Head Self-Attention + Residual + LayerNorm
        attn_out = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.attn_layer_norm(hidden_states + self.attn_dropout(attn_out))

        # Sub-layer 2: Position-wise Feed-Forward Network + Residual + LayerNorm
        ffn_out = self.fc2(self.activation(self.fc1(hidden_states)))
        hidden_states = self.ffn_layer_norm(hidden_states + self.ffn_dropout(ffn_out))
        return hidden_states


class BERTEncoder(nn.Module):
    """Tumpukan N buah EncoderLayer dua arah."""

    def __init__(
        self,
        num_layers: int = 6,
        hidden_size: int = 256,
        num_heads: int = 8,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class MLMHead(nn.Module):
    """Prediction head untuk Masked Language Modeling sesuai SPEC.md §3.

    Arsitektur: Linear(H -> H) + GELU + LayerNorm + Linear(H -> V).
    """

    def __init__(
        self,
        hidden_size: int = 256,
        vocab_size: int = 30000,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = _get_activation(activation)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class BERT(nn.Module):
    """Model utama BERT untuk Masked Language Modeling (MLM) murni."""

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 128,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        tie_weights: bool = False,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.tie_weights = tie_weights

        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_length=max_seq_length,
            type_vocab_size=type_vocab_size,
            pad_token_id=pad_token_id,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = BERTEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
        self.mlm_head = MLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

        self.apply(self._init_weights)

        if self.tie_weights:
            self.mlm_head.decoder.weight = self.embeddings.word_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Inisialisasi bobot mengikuti standar BERT (Normal std=0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BERT":
        """Inisialisasi BERT langsung dari dictionary hasil load config.yaml."""
        model_cfg = config.get("model", config)
        return cls(
            vocab_size=model_cfg.get("vocab_size", 30000),
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            intermediate_size=model_cfg.get("intermediate_size", 1024),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "gelu"),
            max_seq_length=model_cfg.get("max_seq_length", 128),
            type_vocab_size=model_cfg.get("type_vocab_size", 2),
            pad_token_id=model_cfg.get("pad_token_id", 0),
            ignore_index=model_cfg.get("ignore_index", -100),
            tie_weights=model_cfg.get("tie_weights", False),
            layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-12)),
        )

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Menghitung total parameter model (untuk verifikasi estimasi SPEC.md §3)."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> MLMOutput:
        """Forward pass BERT MLM sesuai kontrak interface SPEC.md §4.

        Args:
            input_ids: LongTensor dengan shape (B, L).
            attention_mask: LongTensor dengan shape (B, L), 1 = valid, 0 = padding.
                Jika None, otomatis dibuat dari (input_ids != pad_token_id).
            token_type_ids: LongTensor dengan shape (B, L). Jika None, diisi 0.
            labels: LongTensor dengan shape (B, L). Posisi non-mask bernilai -100.

        Returns:
            MLMOutput(logits, loss):
                - logits: FloatTensor dengan shape (B, L, V)
                - loss: FloatTensor scalar () jika labels diberikan, selain itu None
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Pre-expand mask sekali menjadi (B, 1, 1, L) untuk seluruh layer encoder
        extended_mask = attention_mask[:, None, None, :]

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        encoded_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_mask,
        )
        logits = self.mlm_head(encoded_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=self.ignore_index,
            )

        return MLMOutput(logits=logits, loss=loss)


if __name__ == "__main__":
    # Sanity check cepat untuk verifikasi kontrak shape & jumlah parameter (SPEC.md §3 & §4)
    torch.manual_seed(42)
    dummy_model = BERT()
    total_params = dummy_model.count_parameters()

    B, L = 4, 128
    dummy_input_ids = torch.randint(5, dummy_model.vocab_size, (B, L))
    dummy_input_ids[:, -16:] = 0  # simulasi [PAD] di akhir sequence
    dummy_mask = (dummy_input_ids != 0).long()
    dummy_labels = torch.full((B, L), -100, dtype=torch.long)
    dummy_labels[:, 1:6] = torch.randint(5, dummy_model.vocab_size, (B, 5))

    out = dummy_model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_mask,
        labels=dummy_labels,
    )
    print(f"Total trainable parameters : {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"Logits shape & dtype       : {tuple(out.logits.shape)}, {out.logits.dtype}")
    print(f"Loss shape & value         : {tuple(out.loss.shape)}, {out.loss.item():.4f}")

