"""
Unit test kecil untuk data/data.py. Jalankan dari root repo:
    python -m pytest tests/test_data.py -q      (atau)      python tests/test_data.py
Tidak mengunduh dataset: memakai data sintetis + vocab sintetis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.data import MLMDataset, clean_line, make_dataloader, mask_tokens
from data.tokenizer import WordTokenizer, load_config

CFG = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
L = CFG["data"]["max_seq_length"]
IGN = CFG["data"]["mlm"]["ignore_index"]


def _tok(n_words=500):
    t2i = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    t2i.update({f"w{i}": 5 + i for i in range(n_words)})
    return WordTokenizer(t2i, CFG)


def _lines(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        k = int(torch.randint(3, 400, (1,), generator=g))  # sebagian > 126 kata -> terpecah
        ids = torch.randint(0, 500, (k,), generator=g).tolist()
        out.append(" ".join(f"w{i}" for i in ids))
    return out


def test_clean_line():
    assert clean_line(" = Judul = \n") == ""
    assert clean_line("   \n") == ""
    assert clean_line(" Well @-@ Known , 1 @,@ 000 and 3 @.@ 5 . ") == "well-known , 1,000 and 3.5 ."
    assert clean_line("A   B") == "a b"


def test_shapes_dtypes_and_keys():
    ds = MLMDataset(_lines(), _tok(), CFG, split="train")
    s = ds[0]
    assert set(s) == {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for v in s.values():
        assert v.shape == (L,) and v.dtype == torch.long
    batch = next(iter(make_dataloader(ds, CFG)))
    B = batch["input_ids"].shape[0]
    for k, v in batch.items():
        assert v.shape == (B, L) and v.dtype == torch.long, k


def test_non_mask_labels_are_ignore_index():
    tok = _tok()
    ds = MLMDataset(_lines(), tok, CFG, split="train")
    for i in range(len(ds)):
        s, orig = ds[i], ds.token_ids[i]
        sel = s["labels"] != IGN
        assert torch.equal(s["labels"][sel], orig[sel])          # label = token asli
        assert torch.equal(s["input_ids"][~sel], orig[~sel])     # non-pilih tidak berubah
        special = (orig == 0) | (orig == 2) | (orig == 3)
        assert not sel[special].any()                            # PAD/CLS/SEP tak pernah dipilih
        assert torch.equal(s["attention_mask"], (orig != 0).long())


def test_masking_ratios_and_random_range():
    tok = _tok()
    g = torch.Generator().manual_seed(123)
    ids = torch.randint(5, 505, (4000, L), generator=g)
    masked, labels = mask_tokens(ids, tok, CFG, generator=torch.Generator().manual_seed(1))
    sel = labels != IGN
    assert abs(sel.float().mean().item() - 0.15) < 0.01
    m = (masked[sel] == 4).float().mean().item()
    changed_rand = ((masked[sel] != 4) & (masked[sel] != ids[sel])).float().mean().item()
    assert abs(m - 0.80) < 0.02 and abs(changed_rand - 0.10) < 0.02
    rnd = masked[sel][(masked[sel] != 4) & (masked[sel] != ids[sel])]
    assert rnd.min() >= 5 and rnd.max() < len(tok)


def test_validation_locked_train_random():
    tok, lines = _tok(), _lines()
    val = MLMDataset(lines, tok, CFG, split="validation")
    assert torch.equal(val[3]["input_ids"], val[3]["input_ids"])
    assert torch.equal(val[3]["labels"], val[3]["labels"])
    tr = MLMDataset(lines, tok, CFG, split="train")
    i = int((tr.token_ids != 0).sum(1).argmax())  # sampel terpanjang
    assert not torch.equal(tr[i]["labels"], tr[i]["labels"])  # 126 token -> hampir pasti beda


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)