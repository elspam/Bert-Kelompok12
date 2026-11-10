# Log Sesi LLM

# LOG INTERAKSI LLM – KELOMPOK 12

## 1. Inisialisasi Proyek & Penyusunan SPEC.md

- **Tanggal:** 31/10/2026
- **Target Modul:** Inisialisasi Proyek & Penyusunan SPEC.md
- **Tautan Chat:** https://chat.deepseek.com/share/z9lch9mg0imsfiithi

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Memberikan konteks proyek BERT from Scratch, peran sebagai Prompt Engineer & LLM Navigator, batasan pengerjaan, serta kondisi hardware. Meminta LLM menghasilkan draft `SPEC.md` yang mencakup task, dataset, tokenizer, arsitektur, interface antar-modul, konfigurasi training, konvensi penamaan, aturan perubahan, dan justifikasi teknis.

**Alasan Khusus:**

Spesifikasi disusun terlebih dahulu sebelum implementasi agar seluruh anggota memiliki acuan yang sama. Hal ini bertujuan menghindari ketidaksesuaian interface, dimensi tensor, dan konfigurasi antar-modul selama pengerjaan paralel.

**Output Model:**

LLM menghasilkan draft `SPEC.md` berisi spesifikasi teknis BERT, termasuk pemilihan task, dataset, konfigurasi arsitektur, hyperparameter, interface antar-modul, serta justifikasi dan risiko implementasi.

**Keputusan Kelompok:**

Setelah peninjauan, kelompok sepakat untuk menggunakan SPEC.md sebagai acuan awal proyek

---

## 2. Struktur Folder Repository & Aturan Git Tim

- **Tanggal:** 01/11/2026
- **Target Modul:** Struktur Folder GitHub & Manajemen Kolaborasi Tim
- **Tautan Chat:** https://chat.deepseek.com/share/z9lch9mg0imsfiithi

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Meminta LLM menyusun struktur direktori proyek dengan pembagian kepemilikan file untuk empat anggota (LUHUNG, BAGAS, BIMA, AJENK), tabel hak akses, konfigurasi `.gitignore`, aturan Git (branch, commit message, alur kerja), serta justifikasi berdasarkan kebutuhan kolaborasi tim.

**Alasan Khusus:**

Struktur folder dan pembagian tugas ditentukan sebelum implementasi agar tidak terjadi konflik file. Aturan Git juga diperlukan untuk menjaga integrasi antar-modul dan melacak kontribusi masing-masing anggota.

**Output Model:**

LLM menghasilkan:

1. Pohon direktori beserta pembagian kepemilikan file.
2. Tabel tanggung jawab masing-masing anggota.
3. Konfigurasi `.gitignore` dan daftar file yang perlu di-commit.
4. Aturan branch, commit message, dan alur kerja Git.
5. Justifikasi struktur dan pembagian tugas.

**Keputusan Kelompok:**

Beberapa direktori yang dianggap berlebihan disederhanakan. Setelah struktur final disepakati, kelompok membuat repository GitHub yang dapat diakses seluruh anggota.

---

## 3. Dokumentasi Penggunaan GitHub & Google Colab

- **Tanggal:** 01/11/2026
- **Target Modul:** Penyusunan README.md dan Panduan Workflow Google Colab
- **LLM:** Claude AI
- **Tautan Chat:** https://claude.ai/share/2e144c88-c163-409e-be05-7eeecb1cea20

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Memberikan struktur direktori proyek dan pembagian tugas anggota. Meminta LLM menyusun `README.md` berisi panduan penggunaan proyek melalui Google Colab, mulai dari clone repository, persiapan lingkungan, menjalankan program, hingga melakukan commit, push, dan pull perubahan melalui GitHub.

**Alasan Khusus:**

Google Colab memiliki lingkungan kerja berbeda dari IDE lokal seperti VS Code. Dokumentasi diperlukan agar seluruh anggota memahami pengelolaan direktori, eksekusi program, dan sinkronisasi kode selama pengerjaan proyek.

**Output Model:**

LLM diminta menghasilkan README.md yang mencakup:

1. Persiapan Google Colab.
2. Clone repository GitHub.
3. Pengaturan direktori dan lingkungan proyek.
4. Instalasi dependensi dan menjalankan program.
5. Prosedur commit, push, dan pull.
6. Alur kolaborasi untuk menghindari konflik Git.

**Keputusan Kelompok:**
Readme disetujui dan dijadikan acuan untuk menjalankan project

---

## 4. Implementasi Konfigurasi & Arsitektur Model BERT

- **Tanggal:** 01/11/2026
- **Target Modul:** Implementasi `config.yaml`, `model/__init__.py`, dan `model/model.py`
- **LLM:** Claude AI
- **Tautan Chat:** [Link share Claude AI]

### Kronologi Prompt & Respons

**Prompt Kelompok:**

"Berdasarkan file SPEC.md yang telah disusun, bantu saya mengimplementasikan config.yaml, model/**init**.py, dan model.py sesuai spesifikasi tersebut."

Melampirkan isi `SPEC.md` sebagai acuan implementasi.

**Alasan Khusus:**

Memastikan implementasi konfigurasi dan arsitektur BERT sesuai dengan spesifikasi teknis yang telah disepakati, sehingga konsisten dengan interface dan kebutuhan modul lainnya.

**Output Model:**

LLM membantu menghasilkan:

1. `config.yaml` untuk menyimpan konfigurasi dan hyperparameter proyek.
2. `model/__init__.py` untuk mengekspor class atau komponen model.
3. `model/model.py` yang berisi implementasi arsitektur BERT sesuai `SPEC.md`, termasuk komponen attention, encoder, dan MLM Head.

**Keputusan Kelompok:**

Memeriksa kesesuaian model, init, dan spec untuk dijalankan apakah bisa dan sesuai dengan ekspektasi, jika ada yang kurang sesuai maka akan disesuaikan oleh anggota yang bertanggung jawab di bagian tuning dan preprocessing.

---

## 5. Data Pipeline: Review dan Uji Word-Level Tokenizer

- **Tanggal:** 04/11/2026
- **Target Modul:** `data/tokenizer.py` (build_tokenizer, WordTokenizer, report_oov)
- **Anggota:** Bagas
- **LLM:** Claude AI
- **Tautan Chat:** https://claude.ai/share/de34dcda-25ab-4abd-b68d-0ae0c21183b5

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Menyusun dan mengirimkan prompt terstruktur sesuai panduan `SYSTEM_PROMPT.md` (Master System Prompt, konteks sesi, dan peran Data Pipeline Bagas) untuk mereview serta memverifikasi kode `tokenizer.py` (class `WordTokenizer`, fungsi `build_tokenizer`, dan `report_oov`). Prompt menyertakan kode implementasi untuk diuji kepatuhannya terhadap batasan `SPEC.md` v1.0 serta `config.yaml`, lalu melampirkan output eksekusi terminal `python -m data.tokenizer`.

**Alasan Khusus:**

- Vocab dibangun dari data train saja agar tidak ada kebocoran data dari validation dan test.
- Semua angka dibaca dari `config.yaml` agar mengikuti SPEC v1.0.
- Rasio OOV dihitung karena SPEC mewajibkan Data Pipeline melaporkannya sebelum training penuh.
- Kode direview oleh LLM kedua dan diuji ke terminal sebelum dipakai, bukan langsung dipercaya.

**Output Model:**

Review kode `tokenizer.py` (class `WordTokenizer` dengan `encode`, `decode`, `save`, `load`; fungsi `build_tokenizer()` dan `report_oov()`). Kode dijalankan pada data tiruan (bukan WikiText asli) untuk uji awal: save/load vocab, ID kata mulai dari 5, padding tepat 128, dan ID asing saat decode menjadi `[UNK]`.

**Hasil Uji (`python -m data.tokenizer`, data WikiText-2 asli):**

- Tipe kata unik di train: 75.723
- Vocab aktif / config: 30.000 / 30.000
- OOV token: train 2,923% (57.354 dari 1.961.891), validation 5,617% (11.478 dari 204.347), test 6,545% (15.044 dari 229.871)
- OOV tipe: train 60,389%, validation 33,948%, test 35,875%
- Save/load vocab: OK

**Catatan Review Manusia:**

- `build_tokenizer()` selalu membangun ulang dan menimpa `vocab.json`; memuat vocab yang sudah ada memakai `WordTokenizer.load()`.
- `encode()` tidak memotong teks bila `max_length` tidak diisi, sehingga `data.py` harus memecah paragraf per 126 kata sebelum memanggilnya.
- Kode menambahkan OOV per tipe kata dan laporan JSON di luar daftar permintaan.
- Angka OOV validation dan test berbeda sekitar 0,01 sampai 0,03 poin persen dari hasil prototipe di Colab. Dugaan penyebab: cara mengurutkan kata berfrekuensi sama di batas 30.000 (belum dibuktikan).
- Memastikan keselarasan format prompt: format prompt yang disusun mandiri dipastikan telah sesuai dengan standar resmi tim di `SYSTEM_PROMPT.md` (Master System Prompt dengan konteks sesi terisi).

**Keputusan Kelompok:**

Tokenizer.py disetujui dan dijadikan acuan untuk menjalankan project

---

## 6. Data Pipeline: Review dan Uji `data/data.py`

- **Tanggal:** 04/11/2026
- **Target Modul:** `data/data.py` (clean_line, load_dataset, mask_tokens, MLMDataset, make_dataloader) dan `tests/test_data.py`
- **Anggota:** Bagas
- **LLM:** Claude AI
- **Tautan Chat:** https://claude.ai/share/de34dcda-25ab-4abd-b68d-0ae0c21183b5

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Menyusun dan mengirimkan prompt terstruktur mandiri berbasis template `SYSTEM_PROMPT.md` untuk mereview serta menguji implementasi modul `data/data.py` (`clean_line`, `load_dataset`, `mask_tokens`, `MLMDataset`, `make_dataloader`) dan suite pengujian `tests/test_data.py`. Prompt melampirkan kode yang telah dibuat untuk mengevaluasi konsistensi kontrak interface (`input_ids`, `attention_mask`, `token_type_ids`, `labels`), aturan masking 80/10/10, penanganan `ignore_index = -100`, serta melampirkan output eksekusi terminal `python -m data.data`.

**Alasan Khusus:**

- `data.py` memakai API `tokenizer.py` yang sudah jadi, sehingga `tokenizer.py` dilampirkan agar nama-namanya konsisten.
- `load_dataset()` dibuat mandiri (unduh, bersihkan, simpan) karena `data/raw/` dan `data/processed/` di-gitignore dan anggota lain tidak punya datanya.
- Semua angka dibaca dari `config.yaml` agar mengikuti SPEC v1.0.
- Kode dibandingkan dengan hasil prototipe Colab sebelum dipercaya, bukan langsung dipakai.

**Output Model:**

Review `data.py`: `clean_line` dan `load_dataset()` bertingkat (file bersih ada: lewati; raw ada: bersihkan; raw belum ada: unduh), `mask_tokens()` (15% kandidat, 80/10/10, token acak dari ID 5 sampai `len(tokenizer)`, label `-100`), `MLMDataset` (paragraf dipecah per 126 kata, masking dinamis untuk train dan dikunci per indeks untuk validation/test), serta `make_dataloader()`. Review juga mencakup lima unit test di `tests/test_data.py` (cleaning, shape/dtype/key, label non-mask, rasio masking, validation tetap dan train acak).

**Hasil Uji (`python -m data.data`, data WikiText-2 asli):**

- Kata unik di train: 75.723; vocab aktif 30.000 / 30.000
- Jumlah sampel: train 25.157, validation 2.628, test 3.029
- Tiap batch: `input_ids`, `attention_mask`, `token_type_ids`, `labels`, masing-masing `(32, 128)` bertipe `torch.int64`
- Pada run ini cleaning dilewati karena `data/processed/` sudah ada (hasil skrip cleaning sementara), sehingga angka sama persis dengan prototipe Colab.
- Hasil `python tests/test_data.py`: [ISI MANUAL]
- Hasil eksperimen pembanding (hapus/pindahkan `data/processed/` lalu jalankan ulang): [ISI MANUAL]

**Catatan Review Manusia:**

- **Temuan utama:** `clean_line` di `data.py` mengganti `@-@`/`@,@`/`@.@` tanpa menghapus spasi di sekelilingnya (`well - known`, `1 , 000`, `3 . 5`), berbeda dari cleaning prototipe yang sudah diverifikasi (`well-known`, `1,000`, `3.5`, memakai regex `\s*@([-,.])@\s*`). Diuji pada satu kalimat contoh: 11 token (regex) vs 17 token (`data.py`). Dampak pada OOV belum diukur pada data asli.
- Risiko reproduksi: anggota lain yang clone dan menjalankan dari nol akan mendapat teks bersih, vocab, dan OOV yang berbeda dari angka yang sudah dilaporkan.
- File tes mengharapkan hasil `clean_line` versi `data.py`, sehingga tes lolos walau cleaning berbeda dari prototipe.
- `mask_tokens` tidak punya pengaman minimal 1 token terpilih; batch yang semua labelnya `-100` berpotensi menghasilkan `NaN` pada `CrossEntropyLoss`. Peluangnya kecil, tetapi batch terakhir kecil (train 5 sampel, validation 4 sampel). Perlu disampaikan ke Bima.
- `WordTokenizer` di-import di `data.py` tetapi tidak dipakai.

**Penanganan Error / Debugging:**

Tidak ada pesan error pada sesi ini. Yang ditemukan adalah ketidaksesuaian logika (lihat Catatan Review Manusia), dan keputusan perbaikannya belum diambil.

**Keputusan Kelompok:**

`data.py` dan `test_data.py` disetujui dan dijadikan acuan untuk menjalankan project, dengan catatan penyesuaian:

1. `clean_line` disamakan dengan cleaning terverifikasi (regex `\s*@([-,.])@\s*`), sehingga hasilnya `well-known`, `1,000`, `3.5`. Versi awal memakai `.replace()` dan menghasilkan `well - known`, `1 , 000`, `3 . 5`, yang berbeda dari angka OOV yang sudah dilaporkan.
2. Ekspektasi `test_clean_line` diubah mengikuti poin 1.

---

## 7. Training Loop, Evaluasi Metrik MLM & Hyperparameter Tuning (`training/`)

LOG INTERAKSI LLM - KELOMPOK 12
• Tanggal Sesi: 05/11/2026
• Target Modul: `training/utils.py`, `training/metrics.py`, dan `training/train.py` (Training Loop, Evaluasi Metrik MLM, & Hyperparameter Tuning CLI — Anggota 3: Bima)
• Tautan Sesi Chat: [https://share.gemini.google/VLXMw7eaKgWB]

KRONOLOGI PROMPT & RESPONS:

1. TAHAP INISIASI (Meminta Kode Awal)

- Prompt Kelompok:
  "Kamu adalah asisten pair-programming PyTorch untuk proyek akademik BERT-from-Scratch (Kelompok 12) menggunakan metode Vibe Coding. Bangun dari nol (tanpa Trainer/scheduler instan HuggingFace) tiga file pada modul `training/` yang mematuhi kontrak `SPEC.md` v1.0 dan `config.yaml`:

1. `training/utils.py`: fungsi `set_seed(42)`, `get_device()`, `build_optimizer(model, cfg)` (AdamW dengan pemisahan parameter `no_decay` untuk bias dan LayerNorm), `build_scheduler(optimizer, total_steps, warmup_steps)` menggunakan `LambdaLR` untuk linear warmup 10% + linear decay, `save_checkpoint` & `load_checkpoint`, serta `TrainingLogger` (ekspor JSON & CSV ke `outputs/logs/`).
2. `training/metrics.py`: fungsi `compute_mlm_accuracy(logits, labels, ignore_index=-100)` yang menghitung akurasi top-1 hanya pada posisi `labels != -100` dan mengembalikan `(num_correct, num_masked, accuracy)`, fungsi `compute_perplexity(avg_loss)` dengan batas aman overflow (`max_loss_cap=20.0`), serta class `EpochMetricTracker` untuk akumulasi metrik tertimbang per token sepanjang epoch.
3. `training/train.py`: fungsi `train_one_epoch()` dan `evaluate()` dengan dukungan FP16 Mixed Precision (`torch.amp.autocast` & `GradScaler`), gradient clipping (`clip_grad_norm_`), pengaman _edge-case_ jika suatu batch tidak memiliki token ter-mask (`(labels != ignore_index).any() == False` atau `loss` bernilai `NaN`/`Inf` sesuai temuan Anggota 2), fungsi `plot_training_curves()` 3 panel ke `outputs/figures/`, serta `main()` dengan argumen CLI (`argparse`) untuk eksperimen _hyperparameter tuning_ (`--lr`, `--batch-size`, `--epochs`, `--weight-decay`, `--warmup-ratio`, `--grad-clip`, `--dropout`, `--no-amp`, `--max-steps`, `--resume`, `--eval-test`) tanpa mengubah `config.yaml` secara permanen."

- Alasan Khusus:
  1. Menjaga kepatuhan penuh terhadap kontrak lintas modul (`SPEC.md` §4–§5 & `SYSTEM_PROMPT.md`) dengan menyertakan signature asli dari `data/tokenizer.py`, `data/data.py`, dan `model/model.py` agar nama tensor (`input_ids`, `attention_mask`, `token_type_ids`, `labels`, `logits`, `loss`) dan dimensi `(B, L, V)` langsung kompatibel.
  2. Menindaklanjuti catatan review manusia dari Anggota 2 (Sesi 6) bahwa batch terakhir yang berukuran kecil berisiko memiliki seluruh `labels == -100`, sehingga diperlukan _guard_ eksplisit sebelum _backpropagation_ agar tidak memicu `NaN` pada `CrossEntropyLoss`.
  3. Mengakumulasi _MLM Accuracy_ dan _Loss_ secara tertimbang terhadap jumlah token ter-mask (`total_correct / total_masked_tokens`) di `EpochMetricTracker` agar metrik epoch eksak dan tidak bias terhadap ukuran batch terakhir.
  4. Menyediakan _override_ hyperparameter berbasis CLI (`apply_cli_overrides` dengan `copy.deepcopy`) agar Anggota 3 dapat menjalankan berbagai eksperimen tuning dan menyimpan log/plot terpisah tanpa melanggar aturan perubahan `config.yaml` milik Ketua.

- Output Model:
  LLM menghasilkan implementasi modular pada 3 file:
  1. `training/utils.py`: Implementasi `set_seed`, `get_device`, `build_optimizer` (memisahkan grup parameter `decay` vs `no_decay` untuk `bias` dan `LayerNorm`), `build_scheduler` berbasis `LambdaLR` (linear warmup diikuti linear decay), `save_checkpoint`/`load_checkpoint`, serta `TrainingLogger` (JSON & CSV).
  2. `training/metrics.py`: Implementasi `compute_mlm_accuracy` (mengabaikan `labels == -100`), `compute_perplexity` dengan _clamping_ `min(avg_loss, 20.0)`, dan `EpochMetricTracker`.
  3. `training/train.py`: Implementasi `train_one_epoch`, `evaluate`, `plot_training_curves` (3 panel: Loss, Accuracy, Perplexity), `parse_args`, `apply_cli_overrides`, dan `main()`.

  Cuplikan kode krusial (pengaman batch tanpa mask, FP16 AMP, & akumulasi metrik di `training/train.py` dan `training/metrics.py`):

  ```python
  # Guard di train_one_epoch (training/train.py):
  has_masked_tokens = bool((labels != ignore_index).any().item())
  if not has_masked_tokens:
      tracker.skipped_batches += 1
      continue

  with torch.amp.autocast(device_type=device.type, enabled=use_amp):
      out = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids,
          labels=labels,
      )
      logits = out.logits
      loss = out.loss

  if loss is None or torch.isnan(loss) or torch.isinf(loss):
      tracker.skipped_batches += 1
      continue
  ```

2. TAHAP PENANGANAN EROR / DEBUGGING

- Pesan Eror dari Notebook:

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "D:\Github Repo\Bert-Kelompok12\training\train.py", line 36, in <module>
    from model.model import BERT
  File "D:\Github Repo\Bert-Kelompok12\model\__init__.py", line 3, in <module>
    from .model import (
  File "D:\Github Repo\Bert-Kelompok12\model\model.py", line 35, in <module>
    class MLMOutput(NamedTuple):
RuntimeError: __class__ not set defining 'MLMOutput' as <class 'model.model.MLMOutput'>. Was __classcell__ propagated to type.__new__?
```

- Prompt Perbaikan ke LLM:
  "Saat saya menjalankan perintah `python -m training.train --config config.yaml --run-name bert_mlm_baseline --eval-test`, muncul pesan eror `RuntimeError: __class__ not set defining 'MLMOutput' as <class 'model.model.MLMOutput'>. Was __classcell__ propagated to type.__new__?` pada `class MLMOutput(NamedTuple)` di `model/model.py` baris 35–55 yang menggunakan `return super().__getitem__(item)`. Selain itu, di `training/train.py` baris 334 terdapat pemanggilan `build_tokenizer(cfg, texts=lines['train'])` sedangkan signature asli di `data/tokenizer.py` adalah `def build_tokenizer(cfg=None, train_path=None, save=True):` (tidak menerima argumen `texts`). Tolong jelaskan penyebab teknis eror `__classcell__` pada subclass `typing.NamedTuple` tersebut dan berikan kode perbaikan untuk `model/model.py` serta `training/train.py`."

- Analisis & Solusi Kelompok:
  1. **Akar Masalah `RuntimeError: __class__ not set defining 'MLMOutput'` (`model/model.py`):** Pada Python, pemanggilan `super()` tanpa argumen (`super().__getitem__(item)`) di dalam method suatu class membuat compiler menyisipkan closure implisit `__classcell__`. Namun, metaclass `typing.NamedTuple` (`NamedTupleMeta`) membangun class baru berbasis `collections.namedtuple` tanpa meneruskan `__classcell__` ke `type.__new__`, sehingga memicu `RuntimeError` saat modul `model.model` di-import oleh `training/train.py`.
  2. **Solusi pada `model/model.py`:** Mengganti `return super().__getitem__(item)` menjadi pemanggilan langsung ke class dasar `return tuple.__getitem__(self, item)` pada `MLMOutput.__getitem__`. Dengan cara ini, kompatibilitas akses atribut (`out.logits`), tuple unpacking (`logits, loss = out`), dan akses dictionary (`out["logits"]`) tetap terjaga sesuai `SPEC.md` tanpa memicu pembuatan `__classcell__`.
  3. **Solusi Integrasi Lintas Modul pada `training/train.py`:** Menyesuaikan pemanggilan `build_tokenizer(cfg, texts=lines["train"])` di fungsi `prepare_tokenizer_and_dataloaders()` menjadi `build_tokenizer(cfg)` karena fungsi `build_tokenizer` di `data/tokenizer.py` sudah membaca berkas `train_clean.txt` secara mandiri dari konfigurasi `cfg`.
  4. **Pelajaran Kelompok:** Saat menggabungkan kode dari beberapa anggota (Data, Model, dan Training), pengujian integrasi end-to-end (`python -m training.train`) sangat penting untuk menangkap perilaku spesifik metaclass Python (`NamedTuple`) dan menyelaraskan argumen fungsi lintas modul.

---

## 8. Pembuatan Notebook Eksperimen Pelatihan & Hyperparameter Tuning (`BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`)

LOG INTERAKSI LLM - KELOMPOK 12
• Tanggal Sesi: 05/11/2026
• Target Modul: `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb` (Notebook Eksekusi Training Loop, Eksperimen Hyperparameter Tuning, Visualisasi Komparasi, & Uji Inferensi MLM — Anggota 3: Bima)
• Tautan Sesi Chat: [https://share.gemini.google/CEIyGoSC1jMY]

KRONOLOGI PROMPT & RESPONS:

1. TAHAP INISIASI (Meminta Kode Awal)

- Prompt Kelompok:
  "Kamu adalah asisten pair-programming PyTorch untuk proyek akademik BERT-from-Scratch (Kelompok 12) menggunakan metode Vibe Coding. Saya telah melampirkan file `.zip` dari repositori proyek kami (`SPEC.md`, `config.yaml`, `data/`, `model/`, dan `training/`). Pelajari seluruh isi file tersebut, lalu buatkan file Jupyter Notebook (`BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`) untuk tugas Anggota 3 (Bima — Training Loop & Hyperparameter Tuning Engineer) tanpa menghubungkannya dengan GitHub (`git clone`/`git push` dilarang, gunakan ekstraksi `.zip` atau direktori proyek langsung) dan tanpa menulis ulang arsitektur dari nol di dalam sel notebook (import dari modul proyek). Susun dengan struktur:

1. Header & ringkasan kontrak spesifikasi (`SPEC.md` v1.0 & `config.yaml`).
2. Setup environment, pemeriksaan GPU CUDA, ekstraksi otomatis `.zip` bila dijalankan di Colab, dan pengecekan dependensi.
3. Inspeksi konfigurasi, pemuatan dataset WikiText-2 & `WordTokenizer`, serta _Sanity Check Forward Pass_ 1 batch tanpa menyimpan checkpoint (verifikasi jumlah parameter ~20.23M, shape `logits` `(B, 128, 30000)`, dan _initial loss_ $\approx \ln(30000) \approx 10.31$).
4. Pelatihan Utama (_Baseline Training_ sesuai `SPEC.md`: `lr=5e-4`, `batch_size=32`, `weight_decay=0.01`, `warmup_ratio=0.10`, `num_epochs=10`, `--eval-test`) beserta penampilan grafik `outputs/figures/training_curves.png`.
5. Eksperimen _Hyperparameter Tuning_ minimal 3 skenario (`tune_lr_1e4` dengan `lr=1e-4`, `tune_lr_1e3` dengan `lr=1e-3`, serta `tune_bs16_warmup05` dengan `batch_size=16` & `warmup_ratio=0.05`) menggunakan CLI override tanpa mengubah `config.yaml`, disertai pembersihan otomatis checkpoint sementara (`.pt`) agar disk Colab tidak penuh.
6. Tabel komparasi hasil seluruh eksperimen dari `outputs/logs/*.json` (disimpan juga ke `outputs/logs/hyperparameter_tuning_summary.csv`) dan visualisasi kurva gabungan (_Val Loss_, _Val Accuracy_, _Val Perplexity_) ke `outputs/figures/hyperparameter_tuning_comparison.png`.
7. Uji inferensi kualitatif (`predict_masked_sentence`) menggunakan checkpoint terbaik (`checkpoints/best_model.pt`) untuk menampilkan _Top-5 prediksi token_ beserta probabilitas _softmax_ pada kalimat berbahasa Inggris yang memiliki token `[MASK]`."

- Alasan Khusus:
  1. Memberikan arsip `.zip` proyek secara utuh kepada LLM agar seluruh pemanggilan fungsi di dalam notebook (`load_config`, `load_dataset`, `WordTokenizer`, `MLMDataset`, `make_dataloader`, `BERT.from_config`, `compute_mlm_accuracy`, `compute_perplexity`, dan CLI `training/train.py`) presisi 100% sesuai kode di repositori.
  2. Menghilangkan ketergantungan autentikasi GitHub (`GITHUB_TOKEN`) di dalam notebook dan mengintegrasikan _auto-mount_ Google Drive (`drive.mount('/content/drive')`) beserta fungsi `resolve_project_root()` agar notebook langsung mengakses folder proyek, data, dan `requirements.txt` dari Google Drive tanpa perlu upload manual ke runtime Colab.
  3. Menghapus tahap _smoke test_ yang menyimpan file checkpoint `.pt` berukuran besar (~242 MB per file) serta menambahkan penghapusan otomatis checkpoint sementara pada _loop_ tuning (`temp_ckpt.unlink()`) agar tidak menghabiskan kuota penyimpanan selama eksperimen multi-run.

- Output Model:
  LLM menghasilkan notebook `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb` yang terdiri dari 15 sel (Markdown & Code) mencakup setup environment, _sanity check_ 1 batch, eksekusi _baseline training_ dengan _live output streaming_ (`subprocess.Popen`), eksekusi 3 skenario _hyperparameter tuning_, agregasi tabel & grafik komparasi antar-run, serta fungsi inferensi kualitatif `predict_masked_sentence` untuk menguji prediksi Top-5 kata pada posisi `[MASK]`.

  Cuplikan kode krusial (eksekusi skenario _Hyperparameter Tuning_ & manajemen disk otomatis di `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`):

  ```python
  tuning_scenarios = [
      {"run_name": "tune_lr_1e4", "lr": 1e-4, "batch_size": 32, "warmup_ratio": 0.10, "weight_decay": 0.01},
      {"run_name": "tune_lr_1e3", "lr": 1e-3, "batch_size": 32, "warmup_ratio": 0.10, "weight_decay": 0.01},
      {"run_name": "tune_bs16_warmup05", "lr": 5e-4, "batch_size": 16, "warmup_ratio": 0.05, "weight_decay": 0.01},
  ]

  for idx, sc in enumerate(tuning_scenarios, start=1):
      cmd_tune = [
          sys.executable, "training/train.py",
          "--config", "config.yaml",
          "--run-name", sc["run_name"],
          "--lr", str(sc["lr"]),
          "--batch-size", str(sc["batch_size"]),
          "--warmup-ratio", str(sc["warmup_ratio"]),
          "--weight-decay", str(sc["weight_decay"]),
          "--epochs", str(TUNING_EPOCHS),
          "--eval-test",
      ]
      # Jalankan proses training & hapus checkpoint sementara setelah metrik tercatat di JSON/CSV
  ```

2. TAHAP PENANGANAN EROR / DEBUGGING & OPTIMASI AKURASI

- Pesan Eror / Kendala Performa dari Notebook:
  Pada eksekusi awal baseline (`tie_weights=False`, `lr_schedule="linear"`, tanpa pengecualian `[UNK]` pada masking MLM, dan tanpa _token-stream packing_), model berjalan tanpa crash namun menghasilkan **MLM Top-1 Accuracy yang sangat rendah** pada validation/test set karena matriks klasifikasi `mlm_head.decoder` (`30.000 x 256`) belajar terpisah dari `word_embeddings` dan token `[UNK]` (ID=1) ikut menjadi target prediksi masking.

- Prompt Perbaikan ke LLM:
  "Setelah saya menjalankan pelatihan di Google Colab, MLM Top-1 Accuracy pada validation/test masih sangat rendah. Tolong analisis akar penyebab rendahnya akurasi pada kode kami (`tie_weights: false`, bug masking token `[UNK]` ID=1 di `data/data.py`, efisiensi _token packing_ di `MLMDataset`, serta jadwal _learning rate_ `cosine` dan sinkronisasi `scheduler.step()` dengan `GradScaler` di `training/`), lalu berikan perbaikan kode lengkap beserta pembaruan sel eksperimen Hyperparameter Tuning dan Full Pre-training di `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`."

- Analisis & Solusi Kelompok:
  1. **Pengaktifan _Weight Tying_ (`tie_weights=True`):** Mengikat bobot `mlm_head.decoder.weight` dengan `embeddings.word_embeddings.weight` sehingga memangkas ~7,68 juta parameter redundan (dari ~20,23M menjadi ~12,55M parameter) dan menyatukan ruang representasi token input dengan proyeksi logit output.
  2. **Perbaikan Target Masking `[UNK]` & _Token-Stream Packing_ (`data/data.py`):** Mengecualikan `vocab.unk_token_id` (ID=1) dari kandidat masking di `mask_tokens()` agar model tidak dipaksa memprediksi token `[UNK]`, serta menerapkan _token-stream packing_ di `MLMDataset` agar setiap sekuens sepanjang `max_seq_len=128` padat dengan konteks dua arah (tidak terbuang menjadi `[PAD]`).
  3. **Penambahan _Cosine Annealing Schedule_ & Sinkronisasi AMP (`training/utils.py` & `training/train.py`):** Menambahkan opsi `lr_schedule="cosine"` pada `build_scheduler()` dan memeriksa `scaler.get_scale() >= old_scale` sebelum memanggil `scheduler.step()` agar jadwal LR tidak melompat ketika _GradScaler_ melewati step akibat _overflow_ gradien FP16.
  4. **Penyelarasan Integrasi Lintas Modul & Pembaruan Sel Eksperimen Notebook:**
     - Menambahkan dukungan pembacaan format `vocab.json` bersarang (`"token_to_id"`) pada `Vocab.load()` di `data/data.py` serta menyelaraskan pemanggilan `BERT.from_config(config)` di `training/train.py` dan `model/model.py`.
     - Memperbarui Bagian Eksperimen _Hyperparameter Tuning_ di `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb` untuk membandingkan 4 konfigurasi (`exp1_untied_linear_lr5e4`, `exp2_tied_linear_lr5e4`, `exp3_tied_cosine_lr5e4`, `exp4_tied_cosine_lr8e4`), dilanjutkan dengan _Full Pre-training_ 15 Epoch (`best_final`) menggunakan konfigurasi pemenang tuning.

---

## 9. Diagnosa & Penanganan _Unigram Plateau_ (Pre-LN Transformer, Early Stopping, Top-5 Accuracy & Long Training)

LOG INTERAKSI LLM - KELOMPOK 12
• Tanggal Sesi: 06/11/2026
• Target Modul: `model/model.py`, `training/utils.py`, `training/train.py`, `config.yaml`, dan `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb` (Penanganan _Unigram Plateau_ & Optimasi Pelatihan Panjang — Anggota 3: Bima)
• Tautan Sesi Chat: [https://share.gemini.google/xRWjAo4HFvmt]

KRONOLOGI PROMPT & RESPONS:

1. TAHAP INISIASI (Meminta Kode Awal / Strategi Optimasi Lanjutan)

- Prompt Kelompok:
  "Kamu adalah asisten pair-programming PyTorch untuk proyek BERT-from-Scratch (Kelompok 12), metode Vibe Coding. Saya lampirkan `.zip` proyek (berisi `config.yaml`, `data/data.py`, `model/model.py`, `training/train.py`, `training/utils.py`, dan notebook `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`). Saya Anggota 3 (Training Loop & Hyperparameter Tuning Engineer).

### HASIL RUN SAAT INI (masih rendah)

- Full training terbaik (tie_weights=True, cosine, lr=8e-4, batch 64, 15 epoch): Val Acc 16.70%, Test Acc 16.47%, Val PPL 373.
- 244 step/epoch × 15 epoch = hanya 3.660 step total; 1 epoch = 3,8 detik di T4.
- Train Acc (16.0%) < Val Acc (16.7%), train loss masih turun -> underfitting.
- Uji inferensi `[MASK]`: prediksi Top-1 SELALU "the" (5–12%), lalu ",", "in", "of" untuk semua kalimat -> model terjebak di unigram plateau dan belum memakai konteks dua arah.
- 4 eksperimen tuning hanya berbeda 14.4%–16.0% karena semuanya tertahan di plateau yang sama.

### TUGAS

1. Konfirmasi atau bantah diagnosis "undertrained + unigram plateau". Periksa juga apakah ada bug tersembunyi yang membuat model tidak memakai konteks (attention mask, positional embedding, label -100, masking 80/10/10, FP16).
2. Perbaiki kode agar akurasi naik signifikan, dengan perubahan berikut:
   a. `model/model.py`: tambahkan opsi `pre_norm` (Pre-LN Transformer: LayerNorm sebelum attention/FFN + LayerNorm final setelah encoder). Default `false` agar perilaku lama tetap sama; baca dari `config["model"]["pre_norm"]` di `BERT.from_config`.
   b. `training/utils.py`: `build_scheduler` menerima `warmup_steps` absolut (prioritas di atas `warmup_ratio` jika diisi). Tambahkan `compute_mlm_metrics` Top-5 accuracy.
   c. `training/train.py`: dukung early stopping (`patience` dari config), catat Top-5 accuracy di history, dan tetap simpan `best_model{tag}.pt`.
   d. `config.yaml`: usulkan default baru untuk run panjang (mis. `batch_size: 32`, `epochs: 100–150`, `warmup_steps: 1000–2000`, `lr` 5e-4–1e-3, `pre_norm: true`). Tandai mana yang merupakan perubahan SPEC kategori (a)/(c).
3. Perbarui notebook (tetap akses langsung dari Google Drive `/content/drive/MyDrive`, tanpa GitHub):
   - Eksperimen tuning singkat (~20 epoch) membandingkan: Post-LN vs Pre-LN, lr 5e-4 vs 1e-3, batch 32 vs 64.
   - Full training panjang (100–150 epoch + early stopping) pada konfigurasi pemenang.
   - Plot Top-1 & Top-5 accuracy, serta uji `[MASK]` ulang untuk membuktikan prediksi tidak lagi selalu "the".
4. Jelaskan target akurasi yang realistis untuk WikiText-2 word-level (vocab 30k, model ~12M parameter) supaya ekspektasi kelompok sesuai.

Kontrak `SPEC.md` wajib tetap dipatuhi: nama `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `logits`, `loss`; shape `(B, 128)` & `(B, 128, V)`; special token ID 0–4; `ignore_index=-100`."

- Alasan Khusus:
  1. Total 3.660 step pada arsitektur Post-LN terlalu pendek sehingga gradien belum membentuk representasi _self-attention_ kontekstual dan model tertahan di _unigram plateau_ (hanya menebak frekuensi kata tertinggi seperti `"the"`).
  2. Arsitektur **Pre-LN (`pre_norm=True`)** menyediakan jalur identitas langsung (_clean residual stream_) dari output ke embedding sehingga model jauh lebih cepat keluar dari _unigram plateau_ dan stabil pada `lr=8e-4` hingga `1e-3`.
  3. Menurunkan `batch_size` dari `64` ke `32` menggandakan jumlah update per epoch (~488 step/epoch), dan menaikkannya hingga maksimal `120` epoch dengan _Early Stopping_ (`patience=15`) memastikan model konvergen penuh tanpa membuang waktu komputasi jika _validation loss_ sudah mendatar.
  4. Menambahkan metrik **Top-5 Accuracy** sangat penting pada MLM word-level ber-vocab 30.000 kata karena banyak posisi `[MASK]` memiliki beberapa sinonim/kandidat kata yang sama-sama valid secara konteks.

- Output Model:
  LLM memperbarui `model/model.py`, `training/utils.py`, `training/train.py`, `config.yaml`, dan menyusun 5 sel utama pada `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`:
  ```python
  # Pre-LN vs Post-LN di TransformerEncoderLayer (model/model.py):
  if self.pre_norm:
      attn_out = self.self_attn(self.norm1(x), attention_mask)
      x = x + self.dropout1(attn_out)
      ffn_out = self.ffn(self.norm2(x))
      x = x + self.dropout2(ffn_out)
  else:
      attn_out = self.self_attn(x, attention_mask)
      x = self.norm1(x + self.dropout1(attn_out))
      ffn_out = self.ffn(x)
      x = self.norm2(x + self.dropout2(ffn_out))
  ```

2. TAHAP PENANGANAN EROR / DEBUGGING

- Pesan Eror dari Notebook:
  Saat menyelaraskan sel baru notebook dengan modul `data/data.py`, `model/__init__.py`, dan `training/__init__.py`:
  1. Sel notebook memanggil `from data.data import create_dataloaders` yang mengembalikan 3 nilai `(train_loader, val_loader, test_loader)`, sedangkan di `data/data.py` baru tersedia `build_dataloaders(cfg)` yang mengembalikan 4 nilai.
  2. `model/__init__.py` dan `training/__init__.py` masih meng-import nama fungsi/class lama (`BERTEncoder`, `run_training`).
- Prompt Perbaikan ke LLM:
  "Selaraskan ekspor fungsi di `data/data.py` (`create_dataloaders`), `model/__init__.py`, dan `training/__init__.py` agar kompatibel penuh dengan ke-5 sel baru pada `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`."
- Analisis & Solusi Kelompok:
  Menambahkan fungsi pembungkus `create_dataloaders(config)` pada `data/data.py` yang mengembalikan `(train_loader, val_loader, test_loader)` sekaligus menyinkronkan `config["model"]["vocab_size"]`, serta memperbarui daftar `__all__` di `model/__init__.py` dan `training/__init__.py`.

---

.... <LANJUT YA GESSS>

LOG INTERAKSI LLM - KELOMPOK 12
• Tanggal Sesi: 06/11/2026
• Target Modul: `model/model.py`, `training/utils.py`, `training/train.py`, `config.yaml`, dan `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb` (Penanganan _Unigram Plateau_ & Optimasi Pelatihan Panjang — Anggota 3: Bima)
• Tautan Sesi Chat: [https://share.gemini.google/xRWjAo4HFvmt]

KRONOLOGI PROMPT & RESPONS:

1. TAHAP INISIASI (Meminta Kode Awal / Strategi Optimasi Lanjutan)

- Prompt Kelompok:
  "Pada hasil run 15 epoch (`batch_size=64`, 244 step/epoch = 3.660 total step), Val Acc mencapai 16.70% (Test Acc 16.47%, Val PPL 373), namun Train Acc (16.00%) masih di bawah Val Acc dan pada uji inferensi `[MASK]` prediksi Top-1 selalu `'the'`, `','`, `'in'`, `'of'` untuk semua kalimat. Tolong konfirmasi diagnosis _undertrained + unigram plateau_, lalu perbaiki kode:

1. `model/model.py`: tambahkan opsi `pre_norm` (Pre-LN Transformer: `LayerNorm` sebelum Attention/FFN + `final_norm` di ujung encoder).
2. `training/utils.py`: dukung `warmup_steps` absolut di `build_scheduler` (dengan `min_lr_ratio=0.05`) serta perhitungan **Top-1 & Top-5 MLM Accuracy** di `compute_mlm_metrics`.
3. `training/train.py`: dukung _Early Stopping_ (`patience` & `min_delta`), catat Top-1 & Top-5 accuracy di `history`, dan simpan checkpoint terbaik.
4. `config.yaml` & `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`: perbarui menjadi 5 sel utama yang berjalan dari Google Drive (Setup & Reload, Eksperimen Tuning 20 Epoch Post-LN vs Pre-LN & BS 64 vs 32, Full Training 120 Epoch + Early Stopping, Plot Top-1 & Top-5 Accuracy, dan Uji Inferensi Kontekstual `[MASK]`)."

- Alasan Khusus:
  1. Total 3.660 step pada arsitektur Post-LN terlalu pendek sehingga gradien belum membentuk representasi _self-attention_ kontekstual dan model tertahan di _unigram plateau_ (hanya menebak frekuensi kata tertinggi seperti `"the"`).
  2. Arsitektur **Pre-LN (`pre_norm=True`)** menyediakan jalur identitas langsung (_clean residual stream_) dari output ke embedding sehingga model jauh lebih cepat keluar dari _unigram plateau_ dan stabil pada `lr=8e-4` hingga `1e-3`.
  3. Menurunkan `batch_size` dari `64` ke `32` menggandakan jumlah update per epoch (~488 step/epoch), dan menaikkannya hingga maksimal `120` epoch dengan _Early Stopping_ (`patience=15`) memastikan model konvergen penuh tanpa membuang waktu komputasi jika _validation loss_ sudah mendatar.
  4. Menambahkan metrik **Top-5 Accuracy** sangat penting pada MLM word-level ber-vocab 30.000 kata karena banyak posisi `[MASK]` memiliki beberapa sinonim/kandidat kata yang sama-sama valid secara konteks.

- Output Model:
  LLM memperbarui `model/model.py`, `training/utils.py`, `training/train.py`, `config.yaml`, dan menyusun 5 sel utama pada `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`:
  ```python
  # Pre-LN vs Post-LN di TransformerEncoderLayer (model/model.py):
  if self.pre_norm:
      attn_out = self.self_attn(self.norm1(x), attention_mask)
      x = x + self.dropout1(attn_out)
      ffn_out = self.ffn(self.norm2(x))
      x = x + self.dropout2(ffn_out)
  else:
      attn_out = self.self_attn(x, attention_mask)
      x = self.norm1(x + self.dropout1(attn_out))
      ffn_out = self.ffn(x)
      x = self.norm2(x + self.dropout2(ffn_out))
  ```

2. TAHAP PENANGANAN EROR / DEBUGGING

- Pesan Eror dari Notebook:
  Saat menyelaraskan sel baru notebook dengan modul `data/data.py`, `model/__init__.py`, dan `training/__init__.py`:
  1. Sel notebook memanggil `from data.data import create_dataloaders` yang mengembalikan 3 nilai `(train_loader, val_loader, test_loader)`, sedangkan di `data/data.py` baru tersedia `build_dataloaders(cfg)` yang mengembalikan 4 nilai.
  2. `model/__init__.py` dan `training/__init__.py` masih meng-import nama fungsi/class lama (`BERTEncoder`, `run_training`).
- Prompt Perbaikan ke LLM:
  "Selaraskan ekspor fungsi di `data/data.py` (`create_dataloaders`), `model/__init__.py`, dan `training/__init__.py` agar kompatibel penuh dengan ke-5 sel baru pada `BERT_Kelompok12_Anggota3_Training_dan_Tuning.ipynb`."
- Analisis & Solusi Kelompok:
  Menambahkan fungsi pembungkus `create_dataloaders(config)` pada `data/data.py` yang mengembalikan `(train_loader, val_loader, test_loader)` sekaligus menyinkronkan `config["model"]["vocab_size"]`, serta memperbarui daftar `__all__` di `model/__init__.py` dan `training/__init__.py`.

---

.... <LANJUT YA GESSS>

## 10. Pemeriksaan Repository dan Pembaruan Dokumentasi

- **Tanggal:** 07/11/2026
- **Target Modul:** Integrasi hasil pekerjaan anggota dan dokumentasi proyek.
- **Tautan Chat:** https://claude.ai/share/7d5b8cc4-1c26-4f00-ac1b-b2bfa7c2bc9c

**Prompt Kelompok:**
Meminta bantuan untuk proses pengintegrasian seluruh hasil pekerjaan tim dalam sebuah laporan dan dokumentasi.

**Alasan Khusus:**
Diperlukan panduan untuk mengintegrasikan hasil pekerjaan seluruh anggota kelompok ke dalam dokumentasi proyek agar setiap proses pengembangan dapat ditelusuri dengan jelas. Selain itu, perlu dipastikan bahwa interaksi dengan LLM, proses debugging, perubahan kode, dan hasil implementasi terdokumentasi secara konsisten serta sesuai dengan kondisi repository yang sebenarnya.

**Output Model:**
LLM memberikan panduan integrasi dokumentasi proyek yang mencakup:

1. Struktur repository untuk memisahkan file implementasi dan dokumentasi.

2. Matriks keterlacakan yang menghubungkan komponen, anggota penanggung jawab, file kode, entri log, bukti percakapan LLM, dan bagian laporan.

3. Alur kerja dokumentasi, mulai dari pengumpulan bukti percakapan, pencatatan log, verifikasi riwayat commit, hingga penyusunan laporan.

4. Lima pertanyaan panduan untuk mengumpulkan informasi dari setiap anggota mengenai implementasi, prompt, debugging, evaluasi saran LLM, dan hasil akhir.

5. Checklist pemeriksaan konsistensi antara dokumentasi, kode program, hiperparameter, dan bukti interaksi LLM.

**Keputusan Kelompok:**
Kelompok menggunakan panduan tersebut sebagai acuan dalam proses integrasi dan penyusunan dokumentasi akhir. Dokumentasi difokuskan pada tiga bagian utama, yaitu log.md, laporan.md, dan folder llm_chats/. Setiap anggota diminta menyediakan bukti interaksi LLM beserta ringkasan proses pengerjaannya agar hasil implementasi dapat diverifikasi dan dicantumkan secara tepat dalam laporan akhir.
