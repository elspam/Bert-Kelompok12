# SPEC.md — Spesifikasi Teknis BERT-from-Scratch

> **Status:** Draft v1.0 — menunggu rapat kelompok
> **Peran penyusun:** Ketua Kelompok / Prompt Engineer & LLM Navigator
> **Framework:** PyTorch
> **Metode:** Vibe Coding (kolaborasi aktif dengan LLM, tanpa copy-paste kode utuh)
> **Hardware asumsi:** Google Colab T4 (16GB VRAM) — *catatan: bagian `[HARDWARE]` di konteks proyek masih kosong; angka di dokumen ini diasumsikan untuk T4. Jika hardware final berbeda (mis. laptop GPU 4GB / CPU only), angka di §3 dan §5 wajib direvisi lewat rapat sebelum coding dimulai.*

---

## 1. Task Model

**Task utama yang direkomendasikan: Masked Language Modeling (MLM) murni.**

- **Definisi task:** Diberikan kalimat dengan sebagian token diganti `[MASK]`, model memprediksi token asli pada posisi yang di-mask. Ini adalah pre-training objective asli BERT dan paling selaras dengan arsitektur encoder dua arah yang menjadi inti proyek.
- **Contoh input-output konkret:**
  - Input: `"the cat sat on the [MASK] and purred"`
  - Target: token asli di posisi `[MASK]` → `"mat"` (atau token apa pun yang valid secara konteks)
  - Output model: distribusi probabilitas atas `vocab_size` untuk setiap posisi yang di-mask.
- **Metrik evaluasi:**
  - **Masked LM Accuracy** (akurasi top-1 pada posisi yang di-mask) — metrik utama.
  - **Masked LM Loss** (cross-entropy, `ignore_index=-100`) — metrik pendamping untuk memantau konvergensi.
  - **Perplexity** pada validation set (turunan dari loss) — untuk perbandingan antar-run.
- **Dataset yang cocok:** WikiText-2 (lihat §2). Alternatif: WikiText-103 (terlalu berat untuk T4 dalam waktu wajar), BookCorpus (lisensi & ukuran tidak praktis).
- **2 task alternatif yang dipertimbangkan:**
  1. **Sentiment Analysis (fine-tuning klasifikasi):** butuh label, menambah kompleksitas data pipeline & head klasifikasi; cocok sebagai *extension* setelah MLM berjalan, bukan task utama.
  2. **Next-Word Prediction / Causal LM:** tidak cocok untuk BERT dua arah (BERT bukan model autoregresif); akan memaksa arsitektur decoder yang menyimpang dari tujuan proyek "BERT dari nol".

> **Keputusan:** MLM murni. Sentiment analysis dicatat sebagai *stretch goal* opsional jika waktu tersisa, tetapi tidak masuk SPEC v1.

---

## 2. Dataset & Tokenizer

- **Nama dataset final:** **WikiText-2** (varian `wikitext-2-raw-v1`).
  - **Split:** train / validation / test (test dipakai sebagai held-out evaluasi akhir; validation untuk monitoring selama training).
  - **Sumber:** HuggingFace Datasets (`wikitext`), lisensi CC BY-SA.
  - **Ukuran perkiraan:** ~2M token total; ~36k baris train, ~3.7k baris validation, ~4.3k baris test.
- **Jenis tokenizer:** **Word-level tokenizer kustom** (dibangun dari nol oleh anggota Data Pipeline), bukan subword BPE.
  - **Alasan:** lebih sederhana untuk Vibe Coding, mudah di-debug, dan cukup untuk dataset ~2M token. BPE bisa diusulkan sebagai revisi SPEC v2 jika OOV terlalu tinggi.
- **`vocab_size`:** **30.000** (target; final ditentukan setelah analisis frekuensi token oleh Data Pipeline — nilai ini masuk kategori perubahan (a) angka/desain).
- **`max_seq_length`:** **128** token.
- **Special tokens & ID (ditetapkan di awal, tidak boleh berubah tanpa rapat):**

| Token   | ID    | Keterangan                     |
|---------|-------|--------------------------------|
| `[PAD]` | 0     | padding                        |
| `[UNK]` | 1     | unknown token (OOV)            |
| `[CLS]` | 2     | klasifikasi / awal sequence    |
| `[SEP]` | 3     | pemisah sequence               |
| `[MASK]`| 4     | token masking untuk MLM        |

> ID 0–4 **dikunci**. Token kosakata lain dimulai dari ID 5.

---

## 3. Arsitektur Model (angka saja, bukan kode)

| Komponen            | Nilai              |
|---------------------|--------------------|
| `hidden_size`       | 256                |
| `num_layers`        | 6                  |
| `num_heads`         | 8                  |
| `head_dim`          | 32 (= 256 / 8) ✔ habis dibagi |
| `intermediate_size` | 1024 (4 × hidden)  |
| `dropout`           | 0.1                |
| `activation`        | GELU               |
| `max_seq_length`    | 128                |
| `vocab_size`        | 30.000             |
| `type_vocab_size`   | 2 (segment A/B)    |

- **MLM Head:** linear projection `hidden_size → hidden_size` + GELU + LayerNorm, lalu linear `hidden_size → vocab_size`, dengan weight tying opsional ke embedding (diputuskan di rapat; default: **tidak** weight-tying agar debugging lebih transparan).
- **Perkiraan jumlah parameter:**
  - Embedding: `30.000 × 256 ≈ 7.7M`
  - Per EncoderLayer: attention ≈ `4 × 256² ≈ 0.26M`, FFN ≈ `2 × 256 × 1024 ≈ 0.52M`, total ≈ **0.8M/layer** × 6 ≈ **4.8M**
  - MLM Head: `256 × 256 + 256 × 30.000 ≈ 7.7M`
  - **Total ≈ 20–21 juta parameter** (tergantung weight tying).

---

## 4. Interface Antar Modul

- **Nama variabel input model (wajib, tidak boleh diganti):**
  - `input_ids` — `torch.LongTensor`
  - `attention_mask` — `torch.LongTensor` (1 = token valid, 0 = padding)
  - `labels` — `torch.LongTensor` (token asli di posisi mask; `-100` di posisi non-mask)
  - `token_type_ids` — `torch.LongTensor` (opsional; default nol)

- **Shape tensor di setiap boundary:**

| Boundary | Tensor | Shape | dtype |
|---|---|---|---|
| `data.py` → `model.py` | `input_ids` | `(B, L)` | `long` |
| `data.py` → `model.py` | `attention_mask` | `(B, L)` | `long` |
| `data.py` → `model.py` | `labels` | `(B, L)` | `long` |
| `data.py` → `model.py` | `token_type_ids` | `(B, L)` | `long` |
| `model.py` → `train.py` | `logits` | `(B, L, V)` | `float` |
| `model.py` → `train.py` | `loss` (scalar, jika `labels` diberikan) | `()` | `float` |

  - `B` = batch_size, `L` = max_seq_length (128), `V` = vocab_size (30.000).

- **Nilai `ignore_index` untuk loss:** **`-100`** (default PyTorch `CrossEntropyLoss`). Posisi non-mask dan padding di-set `-100` sehingga tidak berkontribusi ke loss.

- **Nama file & fungsi utama tiap modul:**

| File | Fungsi utama | Pemilik |
|---|---|---|
| `data.py` | `build_tokenizer()`, `load_dataset()`, `MLMDataset.__getitem__()`, `make_dataloader()` | Anggota Data Pipeline |
| `model.py` | `MultiHeadAttention`, `EncoderLayer`, `BERTEncoder`, `MLMHead`, `BERT.forward()` | Ketua |
| `train.py` | `train_one_epoch()`, `evaluate()`, `main()` | Anggota Training Loop |

---

## 5. Training Config

| Hyperparameter | Nilai | Catatan |
|---|---|---|
| `batch_size` | 32 | bisa diturunkan ke 16 jika OOM di T4 |
| `learning_rate` | 5e-4 | lebih tinggi dari BERT-base karena model kecil |
| `optimizer` | AdamW | standar untuk Transformer |
| `weight_decay` | 0.01 | regularisasi |
| `num_epochs` | 10 | early stopping via val loss |
| `warmup_steps` | 10% dari total steps | linear warmup |
| `lr_schedule` | linear decay setelah warmup | |
| `gradient_clip` | 1.0 | max-norm clipping |
| `mixed_precision` | FP16 (AMP) | opsional, direkomendasikan untuk T4 |
| `seed` | 42 | reproducibility |

- **Strategi masking MLM (15% token, aturan 80/10/10):**
  - Pilih 15% token acak dari sequence (bukan `[PAD]`, `[CLS]`, `[SEP]`).
  - Dari 15% tersebut: **80%** diganti `[MASK]`, **10%** diganti token acak dari vocab, **10%** dibiarkan apa adanya.
  - Posisi yang dipilih (termasuk yang dibiarkan) tetap masuk ke `labels`; posisi lain = `-100`.

---

## 6. Konvensi Penamaan

- **Nama file:** `data.py`, `model.py`, `train.py` (lowercase, snake_case).
- **Nama class:** `PascalCase` — `MLMDataset`, `MultiHeadAttention`, `EncoderLayer`, `BERTEncoder`, `MLMHead`, `BERT`.
- **Nama fungsi utama:** `snake_case` — `build_tokenizer`, `make_dataloader`, `train_one_epoch`, `evaluate`.
- **Nama variabel standar (wajib dipakai semua anggota):**
  - `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `logits`, `loss`
  - `batch_size`, `max_seq_length`, `vocab_size`, `hidden_size`, `num_heads`, `num_layers`
  - `ignore_index = -100`
- **Konvensi commit message (singkat):**
  - `feat(<modul>): <deskripsi singkat>` — fitur baru
  - `fix(<modul>): <deskripsi singkat>` — perbaikan bug
  - `spec: <deskripsi>` — perubahan SPEC.md
  - `docs: <deskripsi>` — dokumentasi / log LLM
  - Contoh: `feat(model): tambah MultiHeadAttention dua arah`

---

## 7. Aturan Perubahan

- Perubahan `SPEC.md` **hanya** boleh lewat rapat kelompok & persetujuan ketua.
- **Klasifikasi jenis perubahan:**
  - **(a) Angka/desain** (mis. `hidden_size`, `vocab_size`, `batch_size`, jumlah epoch) → keputusan rapat, dicatat di changelog SPEC.
  - **(b) Interface/logika lintas modul** (mis. nama variabel, shape tensor, `ignore_index`, signature fungsi) → rapat **+ verifikasi LLM** bahwa perubahan konsisten di ketiga modul, lalu update SPEC.
  - **(c) Arsitektur fundamental** (mis. ganti tokenizer ke BPE, ganti task ke klasifikasi, tambah layer) → **rapat besar + revisi SPEC v2**, wajib dokumentasi di `log.md` oleh Documentarian.
- Setiap perubahan wajib mencatat: tanggal, jenis (a/b/c), alasan, dan penanggung jawab.

---

## Justifikasi

### Task: MLM murni
Dipilih karena paling selaras dengan arsitektur BERT dua arah (tujuan inti proyek), tidak butuh label tambahan (menghemat kerja Data Pipeline), dan metriknya (akurasi top-1 + loss) mudah dimonitor di T4. Sentiment analysis ditunda karena menambah kompleksitas head & pipeline label — berisiko tinggi untuk timeline 4 orang dengan metode Vibe Coding.

### Dataset: WikiText-2
~2M token cukup untuk menunjukkan konvergensi MLM pada model ~20M parameter, dan cukup kecil untuk diload penuh di RAM Colab. WikiText-103 akan memaksa batch kecil / gradient accumulation yang memperlambat iterasi debugging — bertentangan dengan semangat Vibe Coding yang butuh feedback loop cepat.

### Tokenizer: word-level, vocab 30.000, max_len 128
Word-level lebih mudah di-debug dan dijelaskan di laporan dibanding BPE; untuk dataset ~2M token, OOV masih bisa ditoleransi dengan `[UNK]`. `max_seq_length=128` menekan biaya komputasi attention (O(L²)) sehingga satu epoch selesai dalam hitungan menit di T4. `vocab_size=30.000` adalah trade-off antara coverage dan ukuran embedding (7.7M parameter — masih wajar).

### Arsitektur: hidden 256, 6 layer, 8 head, FFN 1024
Konfigurasi ini ~20M parameter — cukup kecil untuk dilatih end-to-end di T4 dalam waktu wajar (target < 1 jam/epoch), tetapi cukup dalam (6 layer) untuk menunjukkan perilaku encoder dua arah. `hidden_size=256` habis dibagi `num_heads=8` (head_dim=32) sehingga tidak ada pembulatan yang menyulitkan debugging. Dropout 0.1 dan GELU mengikuti resep BERT asli agar perilaku mudah diprediksi saat debugging dengan LLM.

### Interface: `input_ids`/`attention_mask`/`labels`, `ignore_index=-100`
Nama variabel mengikuti konvensi HuggingFace/PyTorch sehingga LLM mudah memberikan saran debugging yang relevan (ini penting karena Vibe Coding bergantung pada kualitas interaksi LLM). `ignore_index=-100` adalah default `CrossEntropyLoss` PyTorch — meminimalkan risiko bug "loss meledak karena padding ikut dihitung". Shape `(B, L, V)` untuk logits adalah standar sehingga `train.py` bisa langsung memakai `CrossEntropyLoss` tanpa reshaping manual.

### Training config: AdamW, lr 5e-4, batch 32, 10 epoch
lr 5e-4 lebih tinggi dari BERT-base (2e-5) karena model jauh lebih kecil dan dilatih from scratch (bukan fine-tuning) — mempercepat konvergensi. AdamW + weight_decay 0.01 adalah standar Transformer yang stabil. Batch 32 dengan `max_len=128` muat di 16GB T4; jika OOM, turunkan ke 16 (perubahan kategori (a)). Warmup 10% + linear decay mencegah divergensi awal. Gradient clip 1.0 adalah pengaman murah terhadap exploding gradient — penting karena kita tidak punya tim tuning besar.

### Konvensi penamaan & aturan perubahan
Nama file/class/fungsi yang seragam memungkinkan LLM memberikan saran yang konsisten lintas sesi (kunci Vibe Coding). Aturan perubahan berjenjang (a/b/c) mencegah "spec drift" — risiko nyata ketika 4 orang berkolaborasi dengan LLM berbeda-beda; perubahan interface (b) diverifikasi LLM karena dampaknya lintas modul dan sulit dideteksi lewat review manual saja.

### 5 keputusan paling berisiko jika salah

1. **`vocab_size` (30.000) & jenis tokenizer (word-level).** Jika OOV terlalu tinggi, akurasi MLM akan plafon rendah dan sulit didiagnosis — bisa disalahartikan sebagai bug arsitektur. Dampak: berminggu-minggu debugging salah arah. Mitigasi: Data Pipeline wajib melaporkan statistik OOV sebelum training penuh.
2. **`max_seq_length = 128`.** Jika terlalu pendek untuk WikiText-2 (banyak paragraf panjang), konteks terpotong dan model belajar dari fragmen tidak wajar. Dampak: metrik rendah meski arsitektur benar. Mitigasi: analisis distribusi panjang sequence di rapat.
3. **Strategi masking 80/10/10 & `ignore_index=-100`.** Salah implementasi (mis. label di posisi non-mask tidak di-set -100) membuat loss "terlihat bagus" padahal model hanya menyalin input — bug senyap paling berbahaya. Dampak: kesimpulan evaluasi salah total. Mitigasi: unit test kecil di `data.py` + review LLM.
4. **Interface shape `(B, L, V)` logits & `(B, L)` labels.** Ketidakcocokan shape antara `model.py` dan `train.py` adalah sumber bug paling sering di proyek Transformer from scratch. Dampak: training crash / loss tidak masuk akal. Mitigasi: kontrak shape di §4 dijadikan acuan wajib; verifikasi LLM saat integrasi.
5. **`hidden_size=256` habis dibagi `num_heads=8`.** Jika angka ini diubah di rapat tanpa mengecek pembagian, `MultiHeadAttention` akan error reshape di tengah training — memakan waktu debugging. Dampak: delay signifikan. Mitigasi: aturan eksplisit di §3 + checklist review konsistensi oleh ketua.

---

> **Catatan akhir:** Dokumen ini adalah SPEC v1.0. Setiap angka yang ditandai sebagai kategori (a) atau (b) hanya boleh berubah lewat prosedur §7. Bagian `[HARDWARE]` di konteks proyek **wajib diisi** sebelum coding dimulai; jika hardware final bukan T4, §3 dan §5 harus direvisi di rapat pertama.