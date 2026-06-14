# 🧭 Daily Compass

Daily Compass adalah sistem rekomendasi keputusan harian mahasiswa berdasarkan kondisi energi, budget keuangan, dan beban aktivitas akademis menggunakan dua model Jaringan Syaraf Tiruan (JST) yang diimplementasikan dari awal (_from scratch_) dengan **NumPy**:

1.  **Perceptron**: Untuk mengklasifikasikan tingkat risiko hari ini (_Safe Day_ atau _Risky Day_).
2.  **Learning Vector Quantization (LVQ)**: Untuk mengklasifikasikan kondisi fisik harian mahasiswa menjadi tiga kelas (_Bugar_, _Normal_, atau _Lelah_).

---

## 🎨 Keunggulan Fitur & Tampilan Baru

Proyek ini telah diperbarui menjadi standar **Masterpiece Pro** dengan fitur-fitur unggulan berikut:

- **Premium Dark Glassmorphic UI**: Antarmuka modern bernuansa gelap dengan efek kaca transparan dan font Outfit yang sangat elegan.
- **Dynamic Neural Network Plotter**: Visualisasi grafis struktur jaringan saraf secara dinamis langsung di dalam aplikasi (Perceptron dan LVQ).
- **Neon Evaluation Curves**: Grafik penurunan error Perceptron dan tingkat ketidakcocokan (Mismatch Rate) LVQ yang dirancang khusus mengikuti skema warna tema gelap aplikasi.
- **Modular Architecture**: Pemisahan kode bersih antara visualisasi/UI (`app.py`), model matematika (`models/`), normalisasi data (`utils.py`), dan dataset latih (`data.py`).

---

## 📁 Struktur Folder Proyek

```text
daily_compass_modular/
├── app.py                      # Antarmuka Web Streamlit (Premium UI & Visualisasi)
├── data.py                     # Dataset Latih Perceptron & LVQ
├── utils.py                    # Normalisasi Vektor Input & Logika Keputusan
├── requirements.txt            # Dependensi Pustaka Python (Streamlit, NumPy, dll)
├── DRAF_LAPORAN_UAS.md         # Draf Laporan Akademis Lengkap untuk UAS JST
└── models/
    ├── __init__.py             # Inisialisasi Paket Model
    ├── perceptron.py           # Kelas Matematika Perceptron (Bipolar Classifier)
    └── lvq.py                  # Kelas Matematika LVQ (Competitive Classifier)
```

---

## ⚙️ Panduan Instalasi & Pengoperasian

1.  Pastikan Anda telah memasang **Python 3.8+** di sistem Anda.
2.  Buka terminal/PowerShell di folder proyek ini:
    ```bash
    cd daily_compass_modular
    ```
3.  Pasang semua pustaka dependensi yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    ```
4.  Jalankan aplikasi web Streamlit:
    ```bash
    streamlit run app.py
    ```
5.  Peramban Anda akan terbuka secara otomatis pada alamat lokal: `http://localhost:8501`.

---
