"""[BAGAS] Dataset, DataLoader, dan masking pipeline untuk MLM."""
"""
data/data.py — Data pipeline MLM (SPEC v1.0 §2, §4, §5).

Urutan pemakaian (jalankan dari root repo):
    cfg = load_config("config.yaml")
    lines = load_dataset(cfg)             # siapkan data/raw + data/processed
    tok = build_tokenizer(cfg)            # dari data/tokenizer.py (vocab dari train)
    ds = MLMDataset(lines["train"], tok, cfg, split="train")
    dl = make_dataloader(ds, cfg)

Kontrak output tiap sampel (dict, semua torch.long, shape (L,) dengan L=128):
    input_ids, attention_mask, token_type_ids, labels
Setelah batching: (B, L).
"""

import os
import re

import torch
from torch.utils.data import DataLoader, Dataset

from data.tokenizer import NUM_SPECIAL, SPLIT_FILES, WordTokenizer, build_tokenizer, load_config

SPLITS = ("train", "validation", "test")
RAW_FILES = {s: f"wikitext2_{s}.txt" for s in SPLITS}


# --------------------------------------------------------------------------- #
# 1. Download + cleaning
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def clean_line(line):
    """
    Bersihkan satu baris mentah WikiText. Return "" jika baris harus dibuang
    (kosong / judul diawali "=").
    Tanda baca TIDAK dirapatkan ke kata: hasil cukup di-.split().
    """
    line = line.strip()
    if not line or line.startswith("="):
        return ""
    line = re.sub(r"\s*@([-,.])@\s*", r"\1", line)
    line = _WS.sub(" ", line).strip()
    return line.lower()


def _hf_dataset_name(cfg):
    name = cfg["data"]["dataset_name"]
    # "wikitext" tanpa namespace dapat memicu HfUriError
    return name if "/" in name else f"Salesforce/{name}"


def _download_raw(cfg):
    """Unduh dengan library `datasets` (satu-satunya tempat ia dipakai)."""
    from datasets import load_dataset as hf_load_dataset  # import lokal sengaja

    raw_dir = cfg["paths"]["raw_data_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    ds = hf_load_dataset(_hf_dataset_name(cfg), cfg["data"]["dataset_config"])
    for split in SPLITS:
        path = os.path.join(raw_dir, RAW_FILES[split])
        with open(path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(row["text"].rstrip("\n") + "\n")
        print(f"[data] raw {split:<10} -> {path}")


def _clean_split(raw_path, clean_path):
    n_in, n_out = 0, 0
    with open(raw_path, "r", encoding="utf-8") as fin, open(clean_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            c = clean_line(line)
            if c:
                fout.write(c + "\n")
                n_out += 1
    print(f"[data] clean {os.path.basename(clean_path):<22} {n_in:>7,} -> {n_out:>7,} baris")


def load_dataset(cfg=None):
    """
    Pastikan data/processed/{train,validation,test}_clean.txt ada.
    - File bersih sudah ada  -> tidak unduh/bersihkan ulang.
    - Raw ada, bersih belum  -> bersihkan saja.
    - Raw belum ada          -> unduh dulu.
    Return: {split: list[str]} baris bersih (satu paragraf per baris).
    """
    cfg = cfg or load_config()
    raw_dir = cfg["paths"]["raw_data_dir"]
    proc_dir = cfg["paths"]["processed_data_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    clean_paths = {s: os.path.join(proc_dir, SPLIT_FILES[s]) for s in SPLITS}
    raw_paths = {s: os.path.join(raw_dir, RAW_FILES[s]) for s in SPLITS}

    if not all(os.path.exists(p) for p in clean_paths.values()):
        if not all(os.path.exists(p) for p in raw_paths.values()):
            _download_raw(cfg)
        for s in SPLITS:
            _clean_split(raw_paths[s], clean_paths[s])
    else:
        print("[data] file bersih sudah ada, lewati unduh/cleaning.")

    out = {}
    for s, p in clean_paths.items():
        with open(p, "r", encoding="utf-8") as f:
            out[s] = [ln.strip() for ln in f if ln.strip()]
    return out


# --------------------------------------------------------------------------- #
# 2. Masking MLM (fungsi murni -> mudah diuji)
# --------------------------------------------------------------------------- #
def mask_tokens(input_ids, tokenizer, cfg, generator=None):
    """
    input_ids: LongTensor (...,) asli (sudah ada [CLS]/[SEP]/[PAD]).
    Return (masked_input_ids, labels), keduanya LongTensor shape sama.

    - Kandidat = bukan [PAD]/[CLS]/[SEP]/[MASK].
    - Tiap kandidat dipilih dengan peluang mlm_probability (15%).
    - Terpilih: 80% -> [MASK], 10% -> ID acak di [5, len(tokenizer)), 10% tetap.
    - labels = token asli di semua posisi TERPILIH (termasuk yang dibiarkan),
      selain itu ignore_index (-100).
    """
    mlm = cfg["data"]["mlm"]
    p_select = mlm["mlm_probability"]
    p_mask = mlm["mask_replace_prob"]
    p_rand = mlm["random_replace_prob"]
    p_keep = mlm["keep_original_prob"]
    ignore_index = mlm["ignore_index"]
    if abs(p_mask + p_rand + p_keep - 1.0) > 1e-6:
        raise ValueError("mask_replace_prob + random_replace_prob + keep_original_prob harus = 1.")
    vocab_active = len(tokenizer)
    if vocab_active <= NUM_SPECIAL:
        raise ValueError("Vocab hanya berisi special token; tidak ada ID acak yang valid.")

    special = (
        (input_ids == tokenizer.pad_token_id)
        | (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
        | (input_ids == tokenizer.mask_token_id)
    )
    selected = (torch.rand(input_ids.shape, generator=generator) < p_select) & ~special

    labels = torch.full_like(input_ids, ignore_index)
    labels[selected] = input_ids[selected]

    u = torch.rand(input_ids.shape, generator=generator)
    to_mask = selected & (u < p_mask)
    to_rand = selected & (u >= p_mask) & (u < p_mask + p_rand)
    # sisanya (selected & u >= p_mask + p_rand) dibiarkan apa adanya

    random_ids = torch.randint(NUM_SPECIAL, vocab_active, input_ids.shape, generator=generator)
    masked = input_ids.clone()
    masked[to_mask] = tokenizer.mask_token_id
    masked[to_rand] = random_ids[to_rand]
    return masked, labels


# --------------------------------------------------------------------------- #
# 3. Dataset
# --------------------------------------------------------------------------- #
def _chunk(words, size):
    for i in range(0, len(words), size):
        yield words[i : i + size]


class MLMDataset(Dataset):
    """
    Tiap paragraf dipecah per (max_seq_length - 2) kata, di-encode menjadi
    [CLS] ... [SEP] + [PAD] sampai max_seq_length. Token asli disimpan di
    `self.token_ids` (N, L); masking dilakukan di __getitem__.

    split == "train"       : masking acak baru setiap sampel diambil.
    split == validation/test: masking dikunci per indeks (seed = seed config + idx).
    """

    def __init__(self, lines, tokenizer, cfg, split="train"):
        if split not in SPLITS:
            raise ValueError(f"split harus salah satu dari {SPLITS}")
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.split = split
        self.is_train = split == "train"
        self.max_seq_length = cfg["data"]["max_seq_length"]
        self.seed = cfg["project"]["seed"]
        body = self.max_seq_length - 2

        rows = []
        for line in lines:
            for piece in _chunk(line.split(), body):
                rows.append(
                    tokenizer.encode(
                        " ".join(piece),
                        add_special_tokens=True,
                        max_length=self.max_seq_length,
                        padding=True,
                    )
                )
        self.token_ids = torch.tensor(rows, dtype=torch.long)  # (N, L)

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, idx):
        original = self.token_ids[idx]  # (L,)
        attention_mask = (original != self.tokenizer.pad_token_id).long()

        generator = None
        if not self.is_train:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(idx))

        input_ids, labels = mask_tokens(original, self.tokenizer, self.cfg, generator)
        return {
            "input_ids": input_ids,  # (L,) long
            "attention_mask": attention_mask,  # (L,) long
            "token_type_ids": torch.zeros_like(original),  # (L,) long
            "labels": labels,  # (L,) long
        }


# --------------------------------------------------------------------------- #
# 4. DataLoader
# --------------------------------------------------------------------------- #
def make_dataloader(dataset, cfg, shuffle=None, num_workers=0):
    """
    Batch default -> dict dengan tensor (B, L) bertipe long.
    shuffle default: True untuk train, False untuk validation/test.
    """
    if shuffle is None:
        shuffle = dataset.is_train
    g = torch.Generator()
    g.manual_seed(cfg["training"]["seed"])
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    lines = load_dataset(cfg)
    tok = build_tokenizer(cfg)
    for split in SPLITS:
        ds = MLMDataset(lines[split], tok, cfg, split=split)
        batch = next(iter(make_dataloader(ds, cfg)))
        print(f"[{split}] sampel={len(ds):,} | " + ", ".join(f"{k}:{tuple(v.shape)}/{v.dtype}" for k, v in batch.items()))
