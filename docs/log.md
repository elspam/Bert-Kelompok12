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

.... <LANJUT YA GESSS>
