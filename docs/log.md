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

## 4. Data Pipeline: Review dan Uji Word-Level Tokenizer

- **Tanggal:** 04/11/2026
- **Target Modul:** `data/tokenizer.py` (build_tokenizer, WordTokenizer, report_oov)
- **Anggota:** Bagas
- **LLM:** Claude AI
- **Tautan Chat:** https://claude.ai/share/de34dcda-25ab-4abd-b68d-0ae0c21183b5

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Di chat ini: meminta bantuan menyusun prompt sesi tokenizer sesuai format `SYSTEM_PROMPT.md` (Master System Prompt, konteks sesi, prompt peran Bagas), menanyakan asal format prompt tersebut, menempelkan kode `tokenizer.py` hasil LLM lain untuk direview, lalu menempelkan output `python -m data.tokenizer`.
Prompt ke LLM pembuat kode (chat lain): [ISI MANUAL: salin prompt yang benar-benar dikirim]

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
- Penanya mempertanyakan asal format prompt; dijelaskan bahwa formatnya gabungan dari `SYSTEM_PROMPT.md` dan tambahan dari LLM, sehingga prompt resmi tim adalah Master System Prompt dengan konteks sesi yang diisi.

**Keputusan Kelompok:**

Tokenizer.py disetujui dan dijadikan acuan untuk menjalankan project

---

## 5. Data Pipeline: Review dan Uji `data/data.py`

- **Tanggal:** 04/11/2026
- **Target Modul:** `data/data.py` (clean_line, load_dataset, mask_tokens, MLMDataset, make_dataloader) dan `tests/test_data.py`
- **Anggota:** Bagas
- **LLM:** Claude AI
- **Tautan Chat:** https://claude.ai/share/de34dcda-25ab-4abd-b68d-0ae0c21183b5

### Kronologi Prompt & Respons

**Prompt Kelompok:**

Di chat ini: meminta bantuan menyusun prompt sesi `data.py` sesuai format `SYSTEM_PROMPT.md`, lalu menempelkan `tests/test_data.py` dan `data/data.py` hasil LLM lain untuk direview, dan menempelkan output `python -m data.data`.
Prompt ke LLM pembuat kode (chat lain): [ISI MANUAL: salin prompt yang benar-benar dikirim]

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

.... <LANJUT YA GESSS>
