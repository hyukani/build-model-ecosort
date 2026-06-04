# EcoSort AI - Notebook Soft-Voting Ensemble 🤖♻️

Selamat datang di repositori **EcoSort AI** khusus untuk model **Ensemble Soft-Voting**. Notebook dan modul di dalam folder ini berfokus pada penggabungan dua arsitektur *Deep Learning*, yaitu **EfficientNetB3** sebagai model utama dan **EfficientNetV2B0** sebagai model sekunder, untuk mencapai akurasi klasifikasi sampah yang optimal dan stabil.

Proyek ini merupakan bagian dari **Capstone Project Coding Camp 2026** yang didukung oleh **DBS Foundation** dengan tema **Sustainable Living & Responsible Consumption**.

**ID Tim:** CC26-PSU272

---

## 📌 Daftar Isi

1. [Ringkasan Proyek](#-ringkasan-proyek)
2. [Struktur Direktori Drive dan Repositori](#%EF%B8%8F-struktur-direktori-drive-dan-repositori)
3. [Arsitektur Model Ensemble](#-arsitektur-model-ensemble)
4. [Konfigurasi Eksperimen](#%EF%B8%8F-konfigurasi-eksperimen)
5. [Eksplorasi Data dan Dataset](#-eksplorasi-data-dan-dataset)
6. [Langkah Penggunaan Google Colab](#-langkah-penggunaan-google-colab)
7. [Hasil Kerja dan Deliverables](#-hasil-kerja-dan-deliverables)
8. [Anggota Tim CC26-PSU272](#-anggota-tim-cc26-psu272)

---

## 🌍 Ringkasan Proyek

**EcoSort AI** adalah sistem cerdas berbasis **Computer Vision** yang berfungsi untuk mengklasifikasikan jenis sampah secara otomatis ke dalam 6 kategori utama berdasarkan standar KLHK. Sistem ini juga dirancang untuk memberikan rekomendasi daur ulang interaktif serta informasi lokasi bank sampah terdekat, khususnya untuk wilayah Kota Surabaya.

Model pada proyek ini menggunakan metode **Soft-Voting Ensemble** dengan menggabungkan keluaran probabilitas dari dua model *pre-trained backbone*. Pendekatan ini digunakan untuk meminimalkan *variance error* dan meningkatkan performa akurasi dibandingkan penggunaan model tunggal.

---

## 🗂️ Struktur Direktori Drive dan Repositori

### Path Google Drive Sebelum Running

Pastikan Anda membuat pintasan (*shortcut*) folder bersama ke Drive utama dengan struktur berikut:

```text
MyDrive/
└── Capstone Coding Camp 2026/
    ├── Dataset_Cleaned/              -> Kumpulan dataset gambar 6 kategori
    ├── ecosort_model/                -> Output training dan model tunggal
    │   └── ensemble_soft_voting/     -> Output akhir model ensemble
    └── coba_coba_sampah/             -> Foto random user untuk external test
```

### Struktur Repositori Git

Struktur repositori proyek adalah sebagai berikut:

```text
├── ensemble_soft_voting/                  # Berkas bobot/model final soft-voting
├── main_model/                            # Berkas model utama EfficientNetB3
├── secondary_model/                       # Berkas model sekunder EfficientNetV2B0
├── .gitattributes                         # Pengaturan Git LFS untuk file model besar
├── README.md                              # Dokumentasi proyek
├── ecosort_model_soft_voting_ensemble.ipynb
├── experiment_summary.csv                 # Log hasil eksperimen model tunggal
├── experiment_summary_ensemble.csv        # Log hasil eksperimen model ensemble
└── metadata_split.csv                     # Pembagian dataset train, validation, dan test
```

---

## 🧠 Arsitektur Model Ensemble

Model ensemble ini menggabungkan probabilitas prediksi atau **softmax output** dari dua model arsitektur melalui mekanisme rata-rata berbobot atau **Soft-Voting**.

1. **Main Backbone:** `EfficientNetB3`  
   Model utama dikonfigurasi dengan resolusi input **300 × 300 piksel** agar lebih sensitif terhadap detail visual, seperti kilap kaca, serat kertas, tekstur organik, dan kemasan plastik.

2. **Secondary Backbone:** `EfficientNetV2B0`  
   Model sekunder digunakan sebagai model komplementer yang lebih efisien dan cepat, sehingga dapat membantu meningkatkan stabilitas prediksi.

Secara umum, proses ensemble dilakukan dengan menggabungkan probabilitas prediksi dari kedua model, kemudian memilih kelas dengan nilai probabilitas gabungan tertinggi.

---

## ⚙️ Konfigurasi Eksperimen

Parameter konfigurasi inti yang digunakan dalam notebook `ecosort_model_soft_voting_ensemble.ipynb` adalah sebagai berikut:

```python
SEED = 42
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16

# Pembagian Dataset Stratified Split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Kategori Sampah
CLASS_NAMES = ['B3', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']
NUM_CLASSES = 6
```

---

## 📊 Eksplorasi Data dan Dataset

Dataset yang digunakan berjumlah **1.197 gambar** setelah proses pembersihan data atau *data cleaning*. Dataset telah disusun secara seimbang pada setiap kelas dan telah melewati pengecekan gambar rusak serta duplikasi.

| Nama Kelas | Jumlah Gambar | Karakteristik Data |
| --- | ---: | --- |
| **B3** | 199 | Sampah elektronik, baterai, bohlam |
| **Glass** | 200 | Botol kaca, beling, jar |
| **Metal** | 200 | Kaleng minuman, aluminium foil |
| **Organic** | 199 | Sisa makanan, daun, ranting |
| **Paper** | 200 | Kardus, koran, kertas dokumen |
| **Plastic** | 199 | Botol plastik, kantong kresek, cup |
| **Total** | **1.197** | **Aman dari gambar rusak dan duplikat** |

---

## 🚀 Langkah Penggunaan Google Colab

1. **Buka Notebook**  
   Akses file `ecosort_model_soft_voting_ensemble.ipynb` melalui Google Colab.

2. **Ubah Runtime**  
   Aktifkan akselerator **GPU T4 Colab** melalui menu:

   ```text
   Runtime -> Change runtime type -> T4 GPU
   ```

3. **Mount Google Drive**  
   Jalankan kode berikut untuk menghubungkan Google Drive dengan Google Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Jalankan Pipeline Notebook**  
   Ikuti alur sel mulai dari pengecekan duplikat gambar menggunakan **MD5 hashing**, pembagian data secara terstratifikasi, pembuatan **TF Data Pipeline** menggunakan `resize_with_pad`, hingga proses pelatihan dan evaluasi model ensemble.

5. **Evaluasi Model**  
   Setelah proses training selesai, lakukan evaluasi menggunakan data test untuk melihat performa model berdasarkan akurasi, *loss*, dan metrik klasifikasi lainnya.

---

## 📦 Hasil Kerja dan Deliverables

Deliverables utama dari proyek ini meliputi:

- **Model AI Terlatih**  
  Model disimpan pada direktori `ecosort_model/ensemble_soft_voting` dengan target akurasi minimal **85%**.

- **Data Log Eksperimen**  
  File log eksperimen terdiri dari:
  - `metadata_split.csv`
  - `experiment_summary.csv`
  - `experiment_summary_ensemble.csv`

- **Notebook Training Ensemble**  
  File `ecosort_model_soft_voting_ensemble.ipynb` berisi keseluruhan proses mulai dari preprocessing, training, evaluasi, hingga penyimpanan model.

- **Integrasi Web**  
  Model disiapkan untuk di-*deploy* agar dapat melayani *request inference* dari aplikasi **Front-end React + Vite** melalui **RESTful API** pada sisi *Back-end*.

---

## 👥 Anggota Tim CC26-PSU272

| Nama | Role |
| --- | --- |
| Muhammad Dafa Alvin Zuhdi | Full-Stack Web Developer |
| Muhammad Aqsa Firdaus | Data Scientist |
| Putri Manika Rukmamaya | AI Engineer |
| Maysahayu Artika Maharani | AI Engineer |
| Muhamad Alfa Reza Gobel | Data Scientist |
| Catherine Natalia Koeswandono | Full-Stack Web Developer |

---

## 📄 Lisensi

Dokumen ini disusun untuk kebutuhan dokumentasi proyek Capstone Project Coding Camp 2026. Penggunaan dan distribusi kode mengikuti kebijakan repositori proyek.

---

**EcoSort AI © 2026 — Pilihan Cerdas untuk Bumi yang Lebih Hijau.**
