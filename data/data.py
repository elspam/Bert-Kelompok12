"""
Data pipeline for BERT MLM pre-training (Member 1 — Data).

Contract summary (see SPEC.md for full details):
  Special token IDs: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
  Output batch keys: input_ids, attention_mask, token_type_ids, labels
  All tensors: shape (B, max_seq_len), dtype torch.long
  Unmasked positions in labels are set to -100.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS: List[Tuple[str, int]] = [
    ("[PAD]", 0),
    ("[UNK]", 1),
    ("[CLS]", 2),
    ("[SEP]", 3),
    ("[MASK]", 4),
]

IGNORE_INDEX: int = -100


class Vocab:
    """Word-level vocabulary with fixed special-token IDs (SPEC.md §3)."""

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        for tok, idx in SPECIAL_TOKENS:
            if token_to_id.get(tok) != idx:
                raise ValueError(
                    f"Special token {tok!r} must have ID {idx}, "
                    f"got {token_to_id.get(tok)}"
                )
        self.token_to_id: Dict[str, int] = dict(token_to_id)
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1

    @property
    def cls_token_id(self) -> int:
        return 2

    @property
    def sep_token_id(self) -> int:
        return 3

    @property
    def mask_token_id(self) -> int:
        return 4

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        unk = self.unk_token_id
        lut = self.token_to_id
        return [lut.get(t, unk) for t in tokens]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token.get(int(i), "[UNK]") for i in ids]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "token_to_id" in data:
            data = data["token_to_id"]
        return cls({k: int(v) for k, v in data.items()})


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def clean_line(line: str) -> str:
    """Lowercase + basic whitespace/token normalization for a single line."""
    line = line.strip()
    if not line or (line.startswith("=") and line.endswith("=")):
        return ""
    tokens = _TOKEN_RE.findall(line.lower())
    return " ".join(tokens)


def build_vocab(
    texts: Iterable[str],
    vocab_size: int = 30000,
    min_freq: int = 2,
) -> Vocab:
    """Build a word-level Vocab from an iterable of whitespace-tokenized lines."""
    counter: Counter[str] = Counter()
    for line in texts:
        counter.update(line.split())

    special_set = {t for t, _ in SPECIAL_TOKENS}
    for s in special_set:
        counter.pop(s, None)

    token_to_id: Dict[str, int] = {t: i for t, i in SPECIAL_TOKENS}
    budget = max(0, vocab_size - len(SPECIAL_TOKENS))

    for tok, freq in counter.most_common():
        if len(token_to_id) - len(SPECIAL_TOKENS) >= budget:
            break
        if freq < min_freq:
            break
        token_to_id[tok] = len(token_to_id)

    return Vocab(token_to_id)


def _chunk(tokens: List[str], max_body: int) -> List[List[str]]:
    return [tokens[i : i + max_body] for i in range(0, len(tokens), max_body)]


def mask_tokens(
    input_ids: torch.Tensor,
    vocab: Vocab,
    mlm_prob: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT 80/10/10 dynamic MLM masking on a 1D or 2D tensor of token IDs.

    Special tokens ([PAD], [UNK], [CLS], [SEP], [MASK]) are NEVER selected for masking.
    Unmasked positions in `labels` are set to -100.
    """
    ids = input_ids.clone()
    labels = input_ids.clone()

    # Exclude ALL special tokens including [UNK] (ID=1) from MLM masking targets
    special = (
        (input_ids == vocab.pad_token_id)
        | (input_ids == vocab.unk_token_id)
        | (input_ids == vocab.cls_token_id)
        | (input_ids == vocab.sep_token_id)
        | (input_ids == vocab.mask_token_id)
    )

    prob = torch.full(ids.shape, mlm_prob, dtype=torch.float)
    prob.masked_fill_(special, 0.0)
    masked_indices = torch.bernoulli(prob, generator=generator).bool()

    # Ensure at least 1 valid non-special token is masked per sequence if possible
    if ids.dim() == 1:
        if not masked_indices.any():
            valid_pos = (~special).nonzero(as_tuple=False).view(-1)
            if len(valid_pos) > 0:
                pick = valid_pos[
                    torch.randint(len(valid_pos), (1,), generator=generator).item()
                ]
                masked_indices[pick] = True

    labels[~masked_indices] = IGNORE_INDEX

    # 80% of the time, replace masked input tokens with [MASK]
    replace_prob = torch.rand(ids.shape, generator=generator)
    indices_replaced = (replace_indices := (replace_prob < 0.8)) & masked_indices
    ids[indices_replaced] = vocab.mask_token_id

    # 10% of the time, replace masked input tokens with random regular word (ID >= 5)
    indices_random = (replace_prob >= 0.8) & (replace_prob < 0.9) & masked_indices
    low_id = 5 if len(vocab) > 5 else 0
    random_words = torch.randint(
        low_id, len(vocab), ids.shape, dtype=torch.long, generator=generator
    )
    ids[indices_random] = random_words[indices_random]

    # Remaining 10%: keep original token unchanged
    return ids, labels


class MLMDataset(Dataset):
    """Pre-training dataset producing fixed-length padded MLM examples.

    Uses token-stream packing across lines so sequences are densely packed
    with bidirectional context up to `max_seq_len - 2` tokens.
    """

    def __init__(
        self,
        lines: Iterable[str],
        vocab: Vocab,
        max_seq_len: int = 128,
        mlm_prob: float = 0.15,
        pack_sequences: bool = True,
        min_seq_tokens: int = 8,
    ) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.mlm_prob = mlm_prob

        body = max_seq_len - 2  # reserve [CLS] and [SEP]
        self.examples: List[List[int]] = []

        if pack_sequences:
            buffer: List[str] = []
            for line in lines:
                toks = line.split()
                if not toks:
                    continue
                buffer.extend(toks)
                while len(buffer) >= body:
                    piece = buffer[:body]
                    buffer = buffer[body:]
                    self.examples.append(vocab.encode(piece))
            if len(buffer) >= min_seq_tokens:
                self.examples.append(vocab.encode(buffer[:body]))
        else:
            for line in lines:
                toks = line.split()
                if not toks:
                    continue
                for piece in _chunk(toks, body):
                    if len(piece) >= min_seq_tokens:
                        self.examples.append(vocab.encode(piece))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        body_ids = self.examples[idx]
        seq_ids = [self.vocab.cls_token_id] + body_ids + [self.vocab.sep_token_id]
        real_len = len(seq_ids)
        pad_len = self.max_seq_len - real_len

        input_ids = torch.tensor(
            seq_ids + [self.vocab.pad_token_id] * pad_len, dtype=torch.long
        )
        attention_mask = torch.tensor(
            [1] * real_len + [0] * pad_len, dtype=torch.long
        )
        token_type_ids = torch.zeros(self.max_seq_len, dtype=torch.long)

        masked_ids, labels = mask_tokens(input_ids, self.vocab, self.mlm_prob)

        return {
            "input_ids": masked_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def prepare_wikitext2(
    processed_dir: str | Path,
    vocab_size: int = 30000,
    min_freq: int = 2,
) -> Vocab:
    """Download WikiText-2 via HuggingFace `datasets`, clean it, and save artifacts."""
    from datasets import load_dataset  # lazy import so unit tests don't need it

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    split_map = {
        "train": "train_clean.txt",
        "validation": "validation_clean.txt",
        "test": "test_clean.txt",
    }

    cleaned_splits: Dict[str, List[str]] = {}
    for split, fname in split_map.items():
        cleaned = [c for raw in ds[split]["text"] if (c := clean_line(raw))]
        cleaned_splits[split] = cleaned
        out = processed_dir / fname
        with out.open("w", encoding="utf-8") as f:
            f.write("\n".join(cleaned) + "\n")

    vocab = build_vocab(
        cleaned_splits["train"], vocab_size=vocab_size, min_freq=min_freq
    )
    vocab.save(processed_dir / "vocab.json")
    return vocab


def build_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """Entry point used by Member 3 (`training/train.py`).

    If processed files are missing in `data.processed_dir`, automatically
    downloads and prepares WikiText-2 first.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    proc_dir = Path(data_cfg["processed_dir"])
    vocab_path = proc_dir / "vocab.json"
    train_path = proc_dir / "train_clean.txt"
    val_path = proc_dir / "validation_clean.txt"
    test_path = proc_dir / "test_clean.txt"

    if not (vocab_path.exists() and train_path.exists()):
        vocab = prepare_wikitext2(
            proc_dir,
            vocab_size=int(data_cfg.get("vocab_size", 30000)),
            min_freq=int(data_cfg.get("min_freq", 2)),
        )
    else:
        vocab = Vocab.load(vocab_path)

    max_len = int(data_cfg.get("max_seq_len", 128))
    mlm_prob = float(data_cfg.get("mlm_probability", data_cfg.get("mlm_prob", 0.15)))
    bs = int(train_cfg.get("batch_size", 32))
    nw = int(train_cfg.get("num_workers", data_cfg.get("num_workers", 2)))

    train_ds = MLMDataset(_read_lines(train_path), vocab, max_len, mlm_prob)
    val_ds = MLMDataset(_read_lines(val_path), vocab, max_len, mlm_prob)
    test_ds = MLMDataset(_read_lines(test_path), vocab, max_len, mlm_prob)

    common = dict(batch_size=bs, num_workers=nw, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, vocab


def create_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Helper returning (train_loader, val_loader, test_loader) for notebook & training."""
    train_loader, val_loader, test_loader, vocab = build_dataloaders(config)
    if "model" in config:
        config["model"]["vocab_size"] = len(vocab)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tr, va, te, vc = build_dataloaders(cfg)
    batch = next(iter(tr))
    print(f"Vocab size: {len(vc)}")
    for k, v in batch.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")