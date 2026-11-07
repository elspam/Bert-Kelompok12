# Laporan Akhir Proyek
# VIBE CODING GENERATIVE AI FROM SCRATCH 
# IMPLEMENTASI BERT FROM SCRATCH MENGGUNAKAN PYTORCH 

Repository: https://github.com/elspam/Bert-Kelompok12 

# 1. DESKRIPSI PROYEK
Proyek ini bertujuan mengimplementasikan model BERT (Bidirectional Encoder Representations from Transformers) from scratch menggunakan PyTorch dengan pendekatan Vibe Coding. Model dikembangkan untuk menjalankan tugas Masked Language Modeling (MLM), yaitu memprediksi token yang disembunyikan berdasarkan konteks kalimat.

Dataset yang digunakan adalah WikiText-2 Raw dengan word-level tokenizer dan strategi masking 80/10/10.

# 2. PEMBAGIAN TUGAS
Anggota | Tanggung Jawab
Luhung  | Ketua, dan prompt engineering
Bagas   | Data pipeline dan tokenizer
Bima    | Training loop dan hyperparameter tuning
Ajenk   | Dokumentasi interaksi LLM dan Error Handling

# 3. TEKNOLOGI YANG DIGUNAKAN
- Python
- PyTorch
- Google Collab 
- Google Drive
- Github
- LLM

# 4. IMPLEMENTASI PROYEK
Pengembangan proyek dibagi menjadi tiga komponen utama:
1. **Data Pipeline:** Meliputi pengolahan dataset WikiText-2, pembuatan vocabulary, tokenisasi, dan proses MLM masking.
2. **Model Architecture:** Implementasi BERT menggunakan Transformer Encoder, Multi-Head Self-Attention, Feed Forward Network, dan MLM Head.
3. **Training:** Meliputi training loop, optimizer AdamW, cosine scheduler, perhitungan loss, serta evaluasi model.

Konfigurasi model dan training dikelola melalui file config.yaml.

# 5. PENGGUNAAN LLM DAN DEBUGGING
LLM dimanfaatkan untuk membantu penyusunan spesifikasi, perancangan struktur repository, implementasi kode, pemeriksaan komponen, serta penyelesaian kendala teknis.

Setiap interaksi penting didokumentasikan dengan mencatat prompt, alasan penggunaan, output model, dan keputusan kelompok. Proses debugging juga mencakup identifikasi error, evaluasi solusi LLM, serta pengujian hasil perbaikan.

# 6. HASIL DAN EVALUASI
Hasil pengembangan meliputi implementasi data pipeline, tokenizer, arsitektur BERT, dan training loop. Pengujian model dilakukan melalui Masked Language Modeling Loss, Top-1 Accuracy, Top-5 Accuracy, dan Perplexity.

Hasil evaluasi akhir akan dilengkapi berdasarkan proses training dan pengujian pada Google Colab.

# 7. DOKUMENTASI
Dokumentasi proyek dikelola melalui folder docs/ dengan struktur:
1. log.md: Catatan interaksi LLM, output, keputusan kelompok dan proses debugging
2. laporan.md: Ringkasan pengembangan dan hasil proyek
3. llm_chats/: Bukti percakapan dengan LLM


<!-- [AJENK] Laporan akhir proyek BERT From Scratch -->

