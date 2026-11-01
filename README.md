# BERT From Scratch — Kelompok 12

Implementasi BERT (MLM) dari nol menggunakan PyTorch, dikerjakan oleh kelompok dan dijalankan di **Google Colab**.

README ini adalah panduan kerja: dari clone repo, menjalankan di Colab, sampai aturan commit dan pull supaya tidak saling menimpa pekerjaan.

---

## 1. Pembagian Tugas (Pemilik File)

| Anggota    | Area                | File / Folder                                                                                         |
| ---------- | ------------------- | ----------------------------------------------------------------------------------------------------- |
| **LUHUNG** | Model & spesifikasi | `model/`, `config.yaml`, `SPEC.md`, `SYSTEM_PROMPT.md`, `README.md`, `requirements.txt`, `.gitignore` |
| **BAGAS**  | Data                | `data/data.py`, `data/tokenizer.py`                                                                   |
| **BIMA**   | Training            | `training/`, `outputs/figures/`                                                                       |
| **AJENK**  | Dokumentasi         | `docs/log.md`, `docs/laporan.md`, `docs/llm_chats/`                                                   |

> **Aturan emas:** edit hanya file milikmu. Jika perlu mengubah file orang lain, kabari pemiliknya atau minta dia yang mengubah. Ini cara paling efektif menghindari merge conflict.

---

## 2. Perbedaan Colab vs VS Code (Baca Dulu)

Colab bukan komputer pribadi, melainkan mesin virtual sementara. Konsekuensinya:

| Hal               | VS Code (lokal)     | Colab                                                      |
| ----------------- | ------------------- | ---------------------------------------------------------- |
| File              | Permanen di laptop  | **Hilang** saat runtime putus/direset                      |
| Folder kerja      | Terbuka terus       | Harus `git clone` ulang tiap sesi baru                     |
| Login Git         | Tersimpan di laptop | Harus diatur ulang tiap sesi                               |
| Pindah folder     | `cd` biasa          | Pakai `%cd` (bukan `!cd`, karena tidak bertahan antar sel) |
| Perintah terminal | Langsung diketik    | Diawali `!` di dalam sel notebook                          |
| GPU               | Tergantung laptop   | Gratis (terbatas), perlu diaktifkan manual                 |

Aturan praktis:

1. **Kode** disimpan di GitHub (lewat commit + push).
2. **Checkpoint model & dataset besar** disimpan di Google Drive (karena di-`gitignore`, tidak masuk GitHub).
3. **Commit dan push sering**. Kalau runtime mati sebelum push, pekerjaan hilang.

---

## 3. Setup Awal (Sekali Saja per Anggota)

### 3.1 Akun & Akses

1. Pastikan punya akun GitHub.
2. Minta Luhung (pemilik repo) menambahkan kamu sebagai **Collaborator** (Settings → Collaborators).
3. Terima undangan lewat email/notifikasi GitHub.

### 3.2 Buat Personal Access Token (PAT)

GitHub **tidak menerima password** untuk `git push` dari Colab. Gunakan token:

1. GitHub → foto profil → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**.
2. Beri nama, misalnya `colab-bert`, dan masa berlaku (misalnya 30–90 hari).
3. Centang scope **`repo`**.
4. Klik **Generate token**, lalu **salin tokennya sekarang** (hanya muncul sekali).

> ⚠️ Token = password. **Jangan** ditulis langsung di notebook, di chat grup, atau di-commit.

### 3.3 Simpan Token di Colab Secrets

1. Di Colab, klik ikon **kunci 🔑** di sidebar kiri (Secrets).
2. **Add new secret**:
   - Name: `GITHUB_TOKEN`
   - Value: _(tempel token)_
3. Aktifkan toggle **Notebook access**.

Secrets ini bersifat pribadi per akun Google, jadi tiap anggota mengisi sendiri.

---

## 4. Menjalankan di Colab (Setiap Sesi Baru)

Buat notebook Colab pribadi (misalnya `kerja_<nama>.ipynb`) dan jalankan sel berikut **berurutan**. Notebook ini cukup disimpan di Google Drive masing-masing, tidak perlu di-commit.

### Sel 1: Aktifkan GPU

Menu **Runtime → Change runtime type → T4 GPU → Save**. Lalu cek:

```python
!nvidia-smi
```

### Sel 2: Mount Google Drive (untuk checkpoint & data)

```python
from google.colab import drive
drive.mount('/content/drive')

# Folder penyimpanan permanen di Drive (otomatis dibuat)
import os
DRIVE_DIR = '/content/drive/MyDrive/bert-from-scratch'
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/data_raw', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/data_processed', exist_ok=True)
```

### Sel 3: Clone repo (dengan token)

Ganti `NAMA_PEMILIK_REPO` dengan username GitHub pemilik repo.

```python
from google.colab import userdata
import os

TOKEN = userdata.get('GITHUB_TOKEN')
USER  = 'NAMA_PEMILIK_REPO'
REPO  = 'bert-from-scratch'

if not os.path.exists(f'/content/{REPO}'):
    !git clone https://{TOKEN}@github.com/{USER}/{REPO}.git /content/{REPO}

%cd /content/{REPO}
```

> Jika repo sudah ada di sesi yang sama (runtime belum direset), sel ini tidak meng-clone ulang.

### Sel 4: Identitas Git (wajib sebelum commit)

Isi dengan nama dan email **akun GitHub-mu sendiri**:

```python
!git config user.name  "Nama Kamu"
!git config user.email "emailgithub@kamu.com"
```

### Sel 5: Pindah ke branch kerjamu & ambil update terbaru

```python
# Pertama kali: buat branch pribadi
!git checkout -b <nama>/<topik>      # contoh: bagas/tokenizer

# Sesi berikutnya: pindah ke branch yang sudah ada
# !git checkout bagas/tokenizer

# Ambil perubahan terbaru dari main
!git fetch origin
!git merge origin/main
```

### Sel 6: Install dependensi

```python
!pip install -q -r requirements.txt
```

> Torch sudah terpasang di Colab. Jika `requirements.txt` memaksa versi torch tertentu dan bentrok, diskusikan dengan Luhung. Jangan menimpa torch bawaan Colab sembarangan.

### Sel 7: Hubungkan data & checkpoint ke Drive

Folder `data/raw`, `data/processed`, dan `checkpoints` di-`gitignore`, jadi isinya hilang tiap sesi. Gunakan symlink ke Drive supaya permanen:

```python
import os, shutil

def link(target, link_path):
    if os.path.islink(link_path):
        return
    if os.path.exists(link_path):
        shutil.rmtree(link_path)   # hapus folder kosong bawaan repo (.gitkeep)
    os.symlink(target, link_path)

link(f'{DRIVE_DIR}/data_raw',       'data/raw')
link(f'{DRIVE_DIR}/data_processed', 'data/processed')
link(f'{DRIVE_DIR}/checkpoints',    'checkpoints')
```

Sekarang apa pun yang ditulis ke `checkpoints/` otomatis tersimpan di Drive.

> Jika `git status` mendeteksi `.gitkeep` terhapus, abaikan saja dan **jangan di-commit** (`git checkout -- data/raw/.gitkeep` bila perlu).

### Sel 8: Jalankan kode

Selalu jalankan dari **root repo** (`/content/bert-from-scratch`) supaya import antar-folder bekerja.

```python
# Contoh: tes cepat tokenizer / data (sesuaikan dengan fungsi yang ada)
!python -m data.tokenizer

# Contoh: training (sesuaikan argumen dengan train.py)
!python -m training.train --config config.yaml
```

Tips:

- Pakai `python -m folder.file` (bukan `python folder/file.py`) agar `from model import ...` tidak error.
- Jika mengubah file `.py` lewat commit/pull lalu ingin mencoba di notebook yang sama, **restart runtime** atau gunakan:
  ```python
  %load_ext autoreload
  %autoreload 2
  ```

---

## 5. Cara Mengedit Kode di Colab

Ada dua pendekatan. Pilih yang paling nyaman, tapi tetap konsisten.

### Opsi A (Disarankan): Edit di VS Code lokal, jalankan di Colab

1. Clone repo di laptop, kerjakan di VS Code, lalu commit dan push ke branch-mu.
2. Di Colab, jalankan `!git pull` (atau Sel 3 dan 5) untuk mengambil kodenya, lalu run.

Kelebihan: nyaman mengedit, riwayat Git rapi, Colab hanya sebagai "mesin eksekusi".

### Opsi B: Edit langsung di Colab

1. Buka file lewat panel **Files** (kiri) → klik dua kali file `.py` → edit di editor Colab.
2. Atau tulis ulang file dari sel dengan `%%writefile`:
   ```python
   %%writefile data/tokenizer.py
   # isi file lengkap di sini
   ```
   ⚠️ `%%writefile` **menimpa seluruh file**. Pastikan isinya lengkap.
3. Simpan, jalankan, lalu **commit dan push sebelum menutup Colab**.

---

## 6. Alur Commit & Push

Semua perintah dijalankan di sel Colab (diawali `!`) atau di terminal VS Code (tanpa `!`).

### 6.1 Cek status

```python
!git status
```

### 6.2 Tambahkan hanya file milikmu

```python
!git add data/data.py data/tokenizer.py
```

Hindari `git add .` kecuali sudah yakin isinya, karena bisa ikut membawa file yang tidak diinginkan (checkpoint, log, data).

### 6.3 Commit dengan pesan jelas

Format: `<tipe>(<area>): <deskripsi singkat>`

```python
!git commit -m "feat(data): tambah masking 80/10/10 pada pipeline MLM"
```

| Tipe       | Kapan dipakai                                 |
| ---------- | --------------------------------------------- |
| `feat`     | Fitur baru                                    |
| `fix`      | Perbaikan bug                                 |
| `docs`     | Perubahan dokumentasi/log/laporan             |
| `refactor` | Merapikan kode tanpa ubah fungsi              |
| `chore`    | Hal teknis (config, requirements, .gitignore) |

Contoh lain:

- `fix(training): perbaiki perhitungan perplexity`
- `docs(log): tambah sesi LLM 3 ke log.md`
- `feat(model): implementasi MultiHeadAttention`

### 6.4 Push ke branch-mu

```python
# Push pertama kali untuk branch baru
!git push -u origin <nama>/<topik>

# Push berikutnya
!git push
```

### 6.5 Buka Pull Request (PR)

1. Buka repo di GitHub → muncul banner **Compare & pull request** → klik.
2. Base: `main`, compare: branch-mu.
3. Tulis ringkasan singkat perubahanmu → **Create pull request**.
4. Minta minimal 1 anggota lain me-review, lalu **Merge**.
5. Setelah merge, update `main` lokal (lihat bagian 7).

> Jangan push langsung ke `main` kecuali disepakati kelompok.

---

## 7. Alur Pull (Mengambil Update Teman)

### Sebelum mulai bekerja (setiap sesi)

```python
!git checkout main
!git pull origin main

# Kembali ke branch kerjamu dan gabungkan update terbaru
!git checkout <nama>/<topik>
!git merge main
```

### Jika ada commit yang belum di-push dan `git pull` ditolak

```python
!git stash                # simpan sementara perubahan yang belum di-commit
!git pull origin main
!git stash pop            # kembalikan perubahan
```

### Menangani Merge Conflict

Conflict muncul saat dua orang mengubah baris yang sama. Git menandai file seperti ini:

```
<<<<<<< HEAD
versi kamu
=======
versi teman
>>>>>>> origin/main
```

Langkah penyelesaian:

1. Buka file bermasalah, pilih/gabungkan versi yang benar, **hapus ketiga penanda** (`<<<<<<<`, `=======`, `>>>>>>>`).
2. Simpan, lalu:
   ```python
   !git add <file_yang_diselesaikan>
   !git commit -m "fix: selesaikan merge conflict pada <file>"
   !git push
   ```
3. Jika bingung, **jangan paksa**. Tanya pemilik file.

> Karena tiap orang punya file sendiri (bagian 1), conflict seharusnya jarang terjadi. Yang paling rawan: `requirements.txt`, `config.yaml`, dan `README.md`. Kabari Luhung sebelum mengubahnya.

---

## 8. Apa yang Boleh dan Tidak Boleh Masuk Git

| Masuk Git ✅                                   | JANGAN masuk Git ❌                |
| ---------------------------------------------- | ---------------------------------- |
| Semua file `.py`, `.md`, `.yaml`, `.txt`       | `data/raw/`, `data/processed/`     |
| `outputs/figures/*.png` (grafik loss/accuracy) | `checkpoints/*.pt`, `*.pth`        |
| `.gitkeep`                                     | `outputs/logs/`                    |
| `docs/llm_chats/*` (transkrip chat LLM)        | Token GitHub, API key, file `.env` |

Jika Git tetap melacak file yang seharusnya di-ignore:

```python
!git rm -r --cached checkpoints/
!git commit -m "chore: berhenti melacak checkpoints"
```

---

## 9. Checklist Ringkas

**Mulai sesi Colab**

- [ ] Aktifkan GPU
- [ ] Mount Drive
- [ ] Clone repo (dengan token dari Secrets)
- [ ] Set `git config user.name` dan `user.email`
- [ ] `git checkout main` → `git pull` → pindah ke branch → `git merge main`
- [ ] `pip install -r requirements.txt`
- [ ] Symlink `data/` dan `checkpoints/` ke Drive

**Selesai bekerja**

- [ ] `git status` → `git add` file milikmu
- [ ] `git commit -m "tipe(area): deskripsi"`
- [ ] `git push`
- [ ] Buka PR jika fitur sudah siap
- [ ] Checkpoint sudah tersimpan di Drive
- [ ] (AJENK) Catat sesi LLM di `docs/log.md`

---

## 10. Troubleshooting

| Masalah                                  | Penyebab & solusi                                                                                   |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `fatal: Authentication failed`           | Token salah/kedaluwarsa atau scope `repo` belum dicentang. Buat token baru dan perbarui di Secrets. |
| `Secret not found: GITHUB_TOKEN`         | Toggle **Notebook access** belum aktif atau nama secret salah.                                      |
| `Permission denied` / `403` saat push    | Belum diundang sebagai collaborator, atau mencoba push ke `main` yang diproteksi.                   |
| `ModuleNotFoundError: model` / `data`    | Tidak berada di root repo. Jalankan `%cd /content/bert-from-scratch` dan pakai `python -m ...`.     |
| `Author identity unknown`                | Belum menjalankan `git config user.name/email` (Sel 4).                                             |
| `rejected: non-fast-forward`             | Ada commit baru di remote. `git pull --rebase origin <branch>` lalu push lagi.                      |
| File hilang setelah Colab restart        | Normal. Clone ulang dan ambil data/checkpoint dari Drive. Pastikan sudah push.                      |
| `CUDA out of memory`                     | Kecilkan `batch_size` atau `max_seq_len` di `config.yaml` (koordinasi dengan Luhung).               |
| Perubahan kode tidak terbaca di notebook | Restart runtime atau gunakan `%autoreload 2`.                                                       |

---

## 11. Aturan Main Kelompok

1. Satu orang = satu branch per topik; jangan kerja bareng di branch yang sama tanpa koordinasi.
2. Pull dari `main` **sebelum** mulai bekerja, push **sebelum** menutup Colab.
3. Jangan pernah commit token, password, atau data mentah.
4. Hyperparameter hanya diubah lewat `config.yaml` (bukan hardcode di kode) dan lewat Luhung.
5. Setiap sesi dengan LLM dicatat oleh Ajenk di `docs/log.md`, transkripnya diarsipkan di `docs/llm_chats/`. Kirim transkripmu ke Ajenk setelah sesi.
