"""[BAGAS] build_tokenizer, encode/decode, dan manajemen vocabulary."""
"""
data/tokenizer.py — Word-level tokenizer dari nol (SPEC v1.0 §2).

Alur:
    build_tokenizer()  -> hitung frekuensi kata di train -> vocab (ID 0-4 terkunci)
    WordTokenizer      -> encode / decode / save / load
    report_oov()       -> rasio [UNK] untuk train / validation / test

Semua angka dibaca dari config.yaml (tidak ada hardcode angka spec).
Hanya library standar + PyYAML; tidak memakai transformers/AutoTokenizer.
"""

import json
import os
from collections import Counter

import yaml

# Nama file hasil cleaning (tidak ada di config.yaml; bukan angka spec).
SPLIT_FILES = {
    "train": "train_clean.txt",
    "validation": "validation_clean.txt",
    "test": "test_clean.txt",
}
NUM_SPECIAL = 5  # ID 0-4 terkunci; kata biasa mulai dari ID 5


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _special_maps(cfg):
    """Return (tokens_by_id, ids_by_token) dan validasi ID 0-4 terkunci."""
    tok = cfg["data"]["special_tokens"]
    ids = cfg["data"]["special_token_ids"]
    locked = {
        tok["pad_token"]: 0,
        tok["unk_token"]: 1,
        tok["cls_token"]: 2,
        tok["sep_token"]: 3,
        tok["mask_token"]: 4,
    }
    configured = {
        tok["pad_token"]: ids["pad_token_id"],
        tok["unk_token"]: ids["unk_token_id"],
        tok["cls_token"]: ids["cls_token_id"],
        tok["sep_token"]: ids["sep_token_id"],
        tok["mask_token"]: ids["mask_token_id"],
    }
    if configured != locked:
        raise ValueError(
            "ID special token di config.yaml tidak sama dengan ID terkunci "
            f"SPEC v1.0 ({locked}). Ini perubahan kategori (b): bawa ke rapat."
        )
    return locked


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
class WordTokenizer:
    def __init__(self, token_to_id, cfg):
        self.cfg = cfg
        self.special_to_id = _special_maps(cfg)
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        tok = cfg["data"]["special_tokens"]
        self.pad_token_id = self.special_to_id[tok["pad_token"]]
        self.unk_token_id = self.special_to_id[tok["unk_token"]]
        self.cls_token_id = self.special_to_id[tok["cls_token"]]
        self.sep_token_id = self.special_to_id[tok["sep_token"]]
        self.mask_token_id = self.special_to_id[tok["mask_token"]]
        self.max_seq_length = cfg["data"]["max_seq_length"]

        # Validasi: special token benar-benar di ID 0-4
        for t, i in self.special_to_id.items():
            if self.token_to_id.get(t) != i:
                raise ValueError(f"Special token {t} harus ber-ID {i} di vocab.")

    def __len__(self):
        """Ukuran vocab AKTIF (bisa < vocab_size config jika kata unik sedikit)."""
        return len(self.token_to_id)

    # ----- encode / decode ------------------------------------------------- #
    def encode(self, text, add_special_tokens=True, max_length=None, padding=False):
        """
        text -> list[int].
        add_special_tokens: [CLS] ... [SEP]
        max_length: potong (SEP tetap dipertahankan di akhir). None = tanpa potong.
        padding: jika True, pad dengan [PAD]=0 sampai tepat max_length
                 (default max_length = config max_seq_length).
        """
        if padding and max_length is None:
            max_length = self.max_seq_length

        ids = [self.token_to_id.get(w, self.unk_token_id) for w in text.split()]

        if add_special_tokens:
            if max_length is not None:
                ids = ids[: max(max_length - 2, 0)]
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        elif max_length is not None:
            ids = ids[:max_length]

        if padding:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """list[int] (atau tensor 1-D) -> string."""
        special_ids = set(self.special_to_id.values())
        words = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in special_ids:
                continue
            words.append(self.id_to_token.get(i, self.cfg["data"]["special_tokens"]["unk_token"]))
        return " ".join(words)

    # ----- simpan / muat --------------------------------------------------- #
    def save(self, path=None):
        path = path or self.cfg["paths"]["vocab_path"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "vocab_size_config": self.cfg["data"]["vocab_size"],
            "vocab_size_actual": len(self),
            "special_tokens": self.special_to_id,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return path

    @classmethod
    def load(cls, cfg, path=None):
        path = path or cfg["paths"]["vocab_path"]
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload["token_to_id"], cfg)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def build_tokenizer(cfg=None, train_path=None, save=True):
    """Bangun vocab dari train saja. Return WordTokenizer."""
    cfg = cfg or load_config()
    special_to_id = _special_maps(cfg)
    vocab_size = cfg["data"]["vocab_size"]
    train_path = train_path or os.path.join(
        cfg["paths"]["processed_data_dir"], SPLIT_FILES["train"]
    )

    counter = Counter()
    for line in _read_lines(train_path):
        counter.update(line.split())
    for t in special_to_id:  # jangan biarkan teks mengisi ID special token
        counter.pop(t, None)

    n_words = vocab_size - NUM_SPECIAL
    # urut frekuensi turun, tie-break alfabet -> deterministik
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n_words]

    token_to_id = dict(special_to_id)
    for offset, (word, _) in enumerate(ranked):
        token_to_id[word] = NUM_SPECIAL + offset

    tokenizer = WordTokenizer(token_to_id, cfg)
    print(f"[tokenizer] tipe kata unik di train : {len(counter):,}")
    print(f"[tokenizer] vocab aktif / config    : {len(tokenizer):,} / {vocab_size:,}")
    if len(tokenizer) < vocab_size:
        print(
            "[tokenizer] CATATAN: vocab aktif < vocab_size config. Embedding tetap "
            "berukuran 30000 (tidak perlu ubah SPEC), tetapi data.py harus memakai "
            "len(tokenizer) sebagai batas atas ID acak saat masking 10%."
        )
    if save:
        print(f"[tokenizer] vocab disimpan ke       : {tokenizer.save()}")
    return tokenizer


# --------------------------------------------------------------------------- #
# OOV
# --------------------------------------------------------------------------- #
def compute_oov(tokenizer, path):
    """Rasio OOV pada satu file (per token & per tipe unik)."""
    total, unk = 0, 0
    types, unk_types = set(), set()
    for line in _read_lines(path):
        for w in line.split():
            total += 1
            types.add(w)
            if w not in tokenizer.token_to_id:
                unk += 1
                unk_types.add(w)
    return {
        "total_tokens": total,
        "unk_tokens": unk,
        "oov_token_ratio": unk / total if total else 0.0,
        "unique_types": len(types),
        "unk_types": len(unk_types),
        "oov_type_ratio": len(unk_types) / len(types) if types else 0.0,
    }


def report_oov(tokenizer, cfg=None, save=True):
    """Hitung & cetak OOV untuk train/validation/test; simpan JSON ke logs_dir."""
    cfg = cfg or tokenizer.cfg
    pdir = cfg["paths"]["processed_data_dir"]
    report = {s: compute_oov(tokenizer, os.path.join(pdir, fn)) for s, fn in SPLIT_FILES.items()}

    print("\n=== Laporan OOV ([UNK]) ===")
    print(f"{'split':<12}{'token':>12}{'UNK token':>12}{'OOV token %':>14}{'OOV tipe %':>13}")
    for s, r in report.items():
        print(
            f"{s:<12}{r['total_tokens']:>12,}{r['unk_tokens']:>12,}"
            f"{r['oov_token_ratio']*100:>13.3f}%{r['oov_type_ratio']*100:>12.3f}%"
        )
    if save:
        logs_dir = cfg["paths"]["logs_dir"]
        os.makedirs(logs_dir, exist_ok=True)
        out = os.path.join(logs_dir, "oov_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nLaporan disimpan ke: {out}")
    return report


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    tok = build_tokenizer(cfg)
    report_oov(tok, cfg)

    # sanity check: round-trip + padding
    sample = next(_read_lines(os.path.join(cfg["paths"]["processed_data_dir"], SPLIT_FILES["train"])))
    ids = tok.encode(sample, max_length=cfg["data"]["max_seq_length"], padding=True)
    assert len(ids) == cfg["data"]["max_seq_length"]
    assert ids[0] == tok.cls_token_id
    print("\nContoh decode:", tok.decode(ids)[:120])
    reloaded = WordTokenizer.load(cfg)
    assert reloaded.token_to_id == tok.token_to_id
    print("Save/load vocab: OK")
