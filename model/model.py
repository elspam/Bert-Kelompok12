import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 256,
        max_seq_len: int = 128,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # Register buffer untuk position_ids (1, max_seq_len)
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        pos_ids = self.position_ids[:, :seq_len].expand(batch_size, seq_len)

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(pos_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = tok_emb + pos_emb + type_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size harus habis dibagi num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Proyeksi dan reshape ke (B, num_heads, S, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: (B, num_heads, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            # attention_mask: (B, S) -> (B, 1, 1, S), bernilai 1 untuk valid, 0 untuk PAD
            ext_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Gunakan nilai aman terhadap FP16 (-1e4) agar tidak overflow di float16
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(ext_mask == 0, fill_val)

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)  # (B, num_heads, S, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(hidden_size, intermediate_size, dropout)

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            # Pre-LN Transformer: LayerNorm sebelum Attention dan sebelum FFN
            attn_out = self.self_attn(self.norm1(x), attention_mask)
            x = x + self.dropout1(attn_out)

            ffn_out = self.ffn(self.norm2(x))
            x = x + self.dropout2(ffn_out)
        else:
            # Post-LN Transformer (perilaku lama default)
            attn_out = self.self_attn(x, attention_mask)
            x = self.norm1(x + self.dropout1(attn_out))

            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_out))
        return x


class MLMHead(nn.Module):
    def __init__(self, hidden_size: int = 256, vocab_size: int = 30000):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        max_seq_len: int = 128,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        tie_weights: bool = True,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ignore_index = ignore_index
        self.tie_weights = tie_weights
        self.pre_norm = pre_norm

        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                pre_norm=pre_norm,
            )
            for _ in range(num_layers)
        ])

        # LayerNorm final di ujung encoder khusus untuk Pre-LN
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-12) if pre_norm else nn.Identity()

        self.mlm_head = MLMHead(hidden_size=hidden_size, vocab_size=vocab_size)

        # Inisialisasi bobot ala BERT
        self.apply(self._init_weights)

        # Weight tying antara token_embeddings dan decoder MLMHead
        if self.tie_weights:
            self.mlm_head.decoder.weight = self.embeddings.token_embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config: dict) -> "BERT":
        m_cfg = config.get("model", {})
        d_cfg = config.get("data", {})
        return cls(
            vocab_size=m_cfg.get("vocab_size", d_cfg.get("vocab_size", 30000)),
            hidden_size=m_cfg.get("hidden_size", 256),
            num_layers=m_cfg.get("num_layers", 4),
            num_heads=m_cfg.get("num_heads", 4),
            intermediate_size=m_cfg.get("intermediate_size", 1024),
            max_seq_len=m_cfg.get("max_seq_len", d_cfg.get("max_seq_len", 128)),
            type_vocab_size=m_cfg.get("type_vocab_size", 2),
            dropout=m_cfg.get("dropout", 0.1),
            pad_token_id=d_cfg.get("pad_token_id", 0),
            ignore_index=d_cfg.get("ignore_index", -100),
            tie_weights=m_cfg.get("tie_weights", True),
            pre_norm=m_cfg.get("pre_norm", False),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        hidden_states = self.embeddings(input_ids, token_type_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.mlm_head(hidden_states)  # (B, 128, V)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }