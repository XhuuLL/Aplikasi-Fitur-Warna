# 🎨 Aplikasi Fitur Warna — Computer Vision Web App

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  Aplikasi berbasis web yang mengimplementasikan teknik <strong>Computer Vision</strong> dan <strong>Machine Learning</strong> untuk klasifikasi jenis buah, jenis bunga, dan analisis tingkat kematangan buah secara otomatis melalui ekstraksi fitur warna <strong>HSV</strong> dan fitur bentuk <strong>Hu Moments</strong> dengan algoritma <strong>K-Nearest Neighbors (KNN)</strong>.
</p>

---

## 📑 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Tech Stack](#-tech-stack)
- [Persyaratan Sistem](#-persyaratan-sistem)
- [Panduan Instalasi](#-panduan-instalasi)
  - [1. Clone Repositori](#1-clone-repositori)
  - [2. Buat Virtual Environment](#2-buat--aktifkan-virtual-environment)
  - [3. Instal Dependensi](#3-instal-dependensi)
  - [4. Persiapan Dataset](#4-persiapan-dataset)
  - [5. Training Model](#5-training-model-ai)
  - [6. Jalankan Aplikasi](#6-jalankan-aplikasi)
- [Struktur Direktori](#-struktur-direktori)
- [Penjelasan Fitur Ekstraksi](#-penjelasan-fitur-ekstraksi)
- [Troubleshooting](#-troubleshooting)
- [Penulis](#-penulis)

---

## ✨ Fitur Utama

| No | Fitur | Deskripsi | Model |
|----|-------|-----------|-------|
| 1 | 🍎 **Klasifikasi Buah** | Mengenali 14 jenis buah dari foto yang diunggah | `model_buah.pkl` |
| 2 | 🌻 **Klasifikasi Bunga** | Mengenali 5 jenis bunga dari foto yang diunggah | `model_bunga.pkl` |
| 3 | 🍌 **Kematangan Pisang** | Mendeteksi fase kematangan pisang (Belum Matang, Matang, Sangat Matang, Busuk) | `model_kematangan.pkl` |
| 4 | 🍅 **Kematangan Tomat** | Mendeteksi fase kematangan tomat (Mentah, Setengah Matang, Matang) | `model_tomat.pkl` |

**Fitur Tambahan pada Setiap Halaman:**
- 📷 Upload foto **atau** ambil langsung dari **webcam**
- 🎨 Visualisasi transformasi citra **RGB → HSV** secara *real-time*
- 📊 Grafik distribusi intensitas warna **(Hue, Saturation, Value)**
- 🏷️ Card hasil klasifikasi dengan desain dinamis sesuai prediksi

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────┐
│                   INPUT (Gambar)                    │
│          Upload Foto / Webcam Capture               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              1. PREPROCESSING                       │
│         Resize citra ke 100×100 piksel              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            2. EKSTRAKSI FITUR                       │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │  Histogram HSV   │  │  Hu Moments (Bentuk)    │  │
│  │  8×8×8 bins      │  │  7 invariant moments    │  │
│  │  = 512 fitur     │  │  log-scale normalized   │  │
│  └──────────────────┘  └─────────────────────────┘  │
│         Digabung menjadi feature vector (519 fitur) │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           3. KLASIFIKASI (KNN k=5)                  │
│    Mencocokkan feature vector terdekat dari         │
│    dataset referensi menggunakan jarak Euclidean    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               OUTPUT (Prediksi)                     │
│      Label klasifikasi + Visualisasi HSV Chart      │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Komponen | Teknologi | Kegunaan |
|----------|-----------|----------|
| **Frontend** | Streamlit | Antarmuka web interaktif |
| **Image Processing** | OpenCV (`cv2`) | Manipulasi & ekstraksi fitur citra |
| **Machine Learning** | scikit-learn (KNN) | Algoritma klasifikasi |
| **Data Handling** | NumPy, Pandas | Pemrosesan data numerik |
| **Model Persistence** | Joblib | Menyimpan & memuat model `.pkl` |
| **Image I/O** | Pillow (PIL) | Membaca gambar dari upload |
| **UI Enhancement** | streamlit-option-menu | Navigasi sidebar kustom |

---

## 📋 Persyaratan Sistem

Sebelum memulai instalasi, pastikan perangkat lunak berikut sudah terinstal di komputer Anda:

| Software | Versi Minimum | Link Download |
|----------|---------------|---------------|
| **Python** | 3.8 atau lebih baru | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | Terbaru (bawaan Python) | Sudah termasuk dalam instalasi Python |
| **Git** | Versi terbaru | [git-scm.com/downloads](https://git-scm.com/downloads/) |

> **💡 Tips:** Saat menginstal Python di Windows, pastikan untuk mencentang opsi **"Add Python to PATH"** agar perintah `python` dan `pip` bisa dijalankan dari terminal mana saja.

### Verifikasi Instalasi

Buka **Command Prompt / Terminal** dan jalankan perintah berikut untuk memastikan semua sudah terinstal:

```bash
python --version
# Output contoh: Python 3.11.9

pip --version
# Output contoh: pip 24.0 from ...

git --version
# Output contoh: git version 2.45.1
```

> ⚠️ Jika salah satu perintah di atas gagal, artinya software tersebut belum terinstal atau belum ditambahkan ke PATH.

---

## 🚀 Panduan Instalasi

Ikuti langkah-langkah berikut secara berurutan dari awal hingga aplikasi siap dipakai.

### 1. Clone Repositori

Buka terminal/command prompt, navigasi ke folder tempat Anda ingin menyimpan proyek, lalu jalankan:

```bash
# Clone repositori dari GitHub
git clone https://github.com/XhuuLL/Aplikasi-Fitur-Warna.git

# Masuk ke direktori proyek
cd "Aplikasi Fitur Warna"
```

> 📝 **Catatan:** Ganti `https://github.com/XhuuLL/Aplikasi-Fitur-Warna.git` dengan URL repositori GitHub Anda yang sebenarnya.

---

### 2. Buat & Aktifkan Virtual Environment

Virtual environment digunakan agar library yang diinstal tidak bentrok dengan proyek Python lain di komputer Anda.

**Membuat virtual environment:**

```bash
python -m venv venv
```

**Mengaktifkan virtual environment:**

<table>
<tr>
<th>Sistem Operasi</th>
<th>Perintah</th>
</tr>
<tr>
<td><strong>Windows (CMD)</strong></td>
<td>

```bash
venv\Scripts\activate
```

</td>
</tr>
<tr>
<td><strong>Windows (PowerShell)</strong></td>
<td>

```bash
venv\Scripts\Activate.ps1
```

</td>
</tr>
<tr>
<td><strong>macOS / Linux</strong></td>
<td>

```bash
source venv/bin/activate
```

</td>
</tr>
</table>

Setelah berhasil diaktifkan, Anda akan melihat tanda `(venv)` di awal baris terminal:

```
(venv) D:\Project-Python\Aplikasi Fitur Warna>
```

> ⚠️ **Penting:** Pastikan virtual environment selalu aktif (ada tanda `(venv)`) sebelum menjalankan perintah `pip install` atau menjalankan aplikasi.

---

### 3. Instal Dependensi

Setelah virtual environment aktif, instal semua library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Perintah ini akan menginstal library berikut secara otomatis:

| Library | Versi | Fungsi |
|---------|-------|--------|
| `streamlit` | 1.33.0 | Framework web app |
| `opencv-python` | 4.9.0.80 | Pemrosesan citra digital |
| `numpy` | 1.26.4 | Operasi array & matriks |
| `pillow` | 10.3.0 | Pembacaan file gambar |
| `scikit-learn` | 1.4.2 | Algoritma Machine Learning (KNN) |
| `joblib` | 1.4.0 | Serialisasi model `.pkl` |

> 💡 Library `streamlit-option-menu` dan `pandas` juga diperlukan. Jika belum terinstal otomatis, jalankan:
> ```bash
> pip install streamlit-option-menu pandas
> ```

**Verifikasi instalasi berhasil:**

```bash
pip list
```

Pastikan semua library di atas muncul dalam daftar output.

---

### 4. Persiapan Dataset

> ⚠️ **PENTING:** Folder dataset **tidak disertakan** dalam repositori karena ukuran file yang besar (sudah dikecualikan melalui `.gitignore`). Anda harus menyiapkan dataset secara mandiri.

#### 4.1 Struktur Folder yang Dibutuhkan

Buat struktur folder berikut di dalam direktori proyek:

```
📁 Aplikasi Fitur Warna/
│
├── 📁 dataset/
│   ├── 📁 buah/                    ← Gambar buah (untuk klasifikasi jenis)
│   │   ├── 📁 Alpukat/             ← Berisi gambar alpukat (.jpg/.png)
│   │   ├── 📁 Apel/
│   │   ├── 📁 Blackberry/
│   │   ├── 📁 Cherimoya/
│   │   ├── 📁 Cherry/
│   │   ├── 📁 Jeruk/
│   │   ├── 📁 Kurma/
│   │   ├── 📁 Melon Oranye/
│   │   ├── 📁 Pear/
│   │   ├── 📁 Pepaya/
│   │   ├── 📁 Pir Berduri/
│   │   ├── 📁 Pisang/
│   │   ├── 📁 Raspberry/
│   │   └── 📁 Strawberry/
│   │
│   └── 📁 flowers/                 ← Gambar bunga (untuk klasifikasi jenis)
│       ├── 📁 daisy/
│       ├── 📁 dandelion/
│       ├── 📁 rose/
│       ├── 📁 sunflower/
│       └── 📁 tulip/
│
└── 📁 dataset_kematangan/          ← Gambar untuk deteksi kematangan
    ├── 📁 unripe/                  ← Pisang belum matang
    ├── 📁 ripe/                    ← Pisang matang
    ├── 📁 overripe/                ← Pisang sangat matang
    ├── 📁 rotten/                  ← Pisang busuk / rusak
    │
    └── 📁 Tomato Ripeness.v2i.multiclass/
        └── 📁 train/              ← Dataset tomat dari Roboflow
            ├── 📁 not_ripe/        ← Tomat mentah
            ├── 📁 half_ripe/       ← Tomat setengah matang
            └── 📁 ripe/            ← Tomat matang
```

#### 4.2 Sumber Dataset

Anda dapat memperoleh dataset gambar dari sumber berikut:

| Dataset | Sumber | Keterangan |
|---------|--------|------------|
| **Buah** | [Kaggle](https://www.kaggle.com/) / Google Images | Kumpulkan minimal 20-50 gambar per jenis buah |
| **Bunga** | [Kaggle - Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) | Dataset 5 jenis bunga populer |
| **Kematangan Pisang** | Koleksi mandiri | Foto pisang pada berbagai fase kematangan |
| **Kematangan Tomat** | [Roboflow - Tomato Ripeness](https://universe.roboflow.com/) | Download dalam format **Multiclass Classification** |

#### 4.3 Sortir Dataset Tomat dari Roboflow (Opsional)

Jika Anda mengunduh dataset tomat dari Roboflow dalam format CSV, gunakan script `sortir_dataset.py` untuk merapikan gambar ke dalam sub-folder secara otomatis:

```bash
python sortir_dataset.py
```

> 📝 Script ini akan membaca file `_classes.csv` dan memindahkan gambar ke folder `not_ripe/`, `half_ripe/`, dan `ripe/` secara otomatis.

---

### 5. Training Model AI

Sebelum aplikasi dapat digunakan untuk memprediksi, Anda **wajib melatih model** terlebih dahulu. Proses ini akan menghasilkan file `.pkl` yang berisi model Machine Learning yang sudah dilatih.

Jalankan perintah berikut **satu per satu** di terminal:

```bash
# ① Training model klasifikasi buah (menghasilkan model_buah.pkl)
python train_model.py

# ② Training model kematangan pisang (menghasilkan model_kematangan.pkl)
python train_kematangan.py

# ③ Training model kematangan tomat (menghasilkan model_tomat.pkl)
python train_tomat.py
```

**Output yang diharapkan (contoh):**

```
🚀 Memulai training dari folder: dataset/buah...
   📂 Mengekstrak data Warna & Bentuk: Alpukat
   📂 Mengekstrak data Warna & Bentuk: Apel
   ...
✨ Training selesai! Akurasi Model dengan Bentuk+Warna: 92.50%
💾 Model baru berhasil disimpan sebagai model_buah.pkl
```

**Verifikasi training berhasil:**

Pastikan file-file berikut sudah terbuat di root folder proyek:

| File | Ukuran Perkiraan | Model Untuk |
|------|------------------|-------------|
| `model_buah.pkl` | ~18 MB | Klasifikasi jenis buah |
| `model_bunga.pkl` | ~7 MB | Klasifikasi jenis bunga |
| `model_kematangan.pkl` | ~19 MB | Kematangan pisang |
| `model_tomat.pkl` | ~220 KB | Kematangan tomat |

> ⚠️ **Catatan:** `train_model.py` hanya men-train model buah. Untuk model bunga, jalankan script yang sama tetapi ubah path dataset ke `dataset/flowers` dan nama output menjadi `model_bunga.pkl`, atau buat script training terpisah dengan logika serupa.

---

### 6. Jalankan Aplikasi

Setelah semua langkah di atas selesai, jalankan aplikasi dengan perintah:

```bash
streamlit run main.py
```

**Output di terminal:**

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Browser akan **otomatis terbuka** dan menampilkan halaman beranda aplikasi. Jika tidak terbuka otomatis, buka browser secara manual dan akses:

```
http://localhost:8501
```

**Untuk menghentikan aplikasi**, tekan `Ctrl + C` di terminal.

---

## 📂 Struktur Direktori

```
📁 Aplikasi Fitur Warna/
│
├── 📄 main.py                  # Entry point - Halaman Beranda
├── 📄 requirements.txt         # Daftar dependensi Python
├── 📄 .gitignore               # Pengecualian file untuk Git
├── 📄 README.md                # Dokumentasi proyek (file ini)
├── 🖼️ zbungan.jpg              # Gambar hero section halaman beranda
│
├── 📁 pages/                   # Halaman-halaman Streamlit
│   ├── 📄 buah.py              # Halaman klasifikasi jenis buah
│   ├── 📄 bunga.py             # Halaman klasifikasi jenis bunga
│   └── 📄 kematangan.py        # Halaman analisis kematangan buah
│
├── 📄 train_model.py           # Script training KNN (buah: HSV + Hu Moments)
├── 📄 train_kematangan.py      # Script training kematangan pisang (HSV only)
├── 📄 train_tomat.py           # Script training kematangan tomat (HSV + Hu Moments)
├── 📄 sortir_dataset.py        # Script utilitas sortir dataset CSV → folder
│
├── 🧠 model_buah.pkl           # Model terlatih - klasifikasi buah
├── 🧠 model_bunga.pkl          # Model terlatih - klasifikasi bunga
├── 🧠 model_kematangan.pkl     # Model terlatih - kematangan pisang
├── 🧠 model_tomat.pkl          # Model terlatih - kematangan tomat
│
├── 📁 dataset/                 # Dataset training (tidak diupload ke GitHub)
│   ├── 📁 buah/                # 14 subfolder jenis buah
│   └── 📁 flowers/             # 5 subfolder jenis bunga
│
├── 📁 dataset_kematangan/      # Dataset kematangan (tidak diupload ke GitHub)
│   ├── 📁 unripe/
│   ├── 📁 ripe/
│   ├── 📁 overripe/
│   ├── 📁 rotten/
│   └── 📁 Tomato Ripeness.v2i.multiclass/
│
└── 📁 venv/                    # Virtual environment (tidak diupload ke GitHub)
```

---

## 🔬 Penjelasan Fitur Ekstraksi

### Histogram Warna HSV

Citra input dikonversi dari ruang warna **BGR** ke **HSV** (*Hue, Saturation, Value*). Kemudian dihitung histogram 3D dengan konfigurasi:

- **Hue (H):** 8 bins — merepresentasikan jenis warna (merah, kuning, hijau, dll.)
- **Saturation (S):** 8 bins — merepresentasikan kepekatan warna
- **Value (V):** 8 bins — merepresentasikan kecerahan

Total fitur warna = **8 × 8 × 8 = 512 fitur** yang dinormalisasi ke rentang [0, 1].

### Hu Moments (Fitur Bentuk)

Digunakan pada model buah dan tomat untuk menambah akurasi. Hu Moments menghasilkan **7 momen invarian** yang bersifat:

- **Translation invariant** — tidak berubah saat objek digeser
- **Scale invariant** — tidak berubah saat objek diperbesar/diperkecil
- **Rotation invariant** — tidak berubah saat objek diputar

Nilai Hu Moments ditransformasi menggunakan **log-scale** dan dinormalisasi ke [0, 1].

### Feature Vector Gabungan

```
Feature Vector = [HSV Histogram (512)] + [Hu Moments (7)] = 519 fitur
```

---

## 🔧 Troubleshooting

<details>
<summary><strong>❌ Error: "File 'model_buah.pkl' tidak ditemukan"</strong></summary>

**Penyebab:** Model belum dilatih (training belum dijalankan).

**Solusi:** Jalankan script training terlebih dahulu:
```bash
python train_model.py
python train_kematangan.py
python train_tomat.py
```
</details>

<details>
<summary><strong>❌ Error: "ModuleNotFoundError: No module named 'streamlit'"</strong></summary>

**Penyebab:** Library belum terinstal atau virtual environment belum aktif.

**Solusi:**
1. Pastikan virtual environment sudah aktif (ada tanda `(venv)`)
2. Instal ulang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
</details>

<details>
<summary><strong>❌ Error: "Folder dataset tidak ditemukan"</strong></summary>

**Penyebab:** Folder dataset belum dibuat atau belum diisi gambar.

**Solusi:** Buat struktur folder sesuai panduan di [Persiapan Dataset](#4-persiapan-dataset) dan isi dengan gambar yang sesuai.
</details>

<details>
<summary><strong>❌ Error saat menjalankan di PowerShell: "execution of scripts is disabled"</strong></summary>

**Penyebab:** Kebijakan eksekusi PowerShell memblokir aktivasi virtual environment.

**Solusi:** Jalankan perintah berikut di PowerShell sebagai Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Kemudian coba aktifkan venv kembali.
</details>

<details>
<summary><strong>❌ Akurasi model rendah</strong></summary>

**Penyebab:** Jumlah gambar dalam dataset terlalu sedikit atau kualitas gambar tidak konsisten.

**Solusi:**
- Tambah jumlah gambar per kategori (minimal 50-100 gambar)
- Pastikan gambar berkualitas baik dan objek terlihat jelas
- Pastikan gambar sudah masuk ke folder yang benar
</details>

---

## 📝 Ringkasan Perintah (Quick Start)

```bash
# 1. Clone & masuk direktori
git clone https://github.com/XhuuLL/Aplikasi-Fitur-Warna.git
cd "Aplikasi Fitur Warna"

# 2. Setup virtual environment
python -m venv venv
venv\Scripts\activate              # Windows CMD
# source venv/bin/activate         # macOS/Linux

# 3. Install dependensi
pip install -r requirements.txt

# 4. (Siapkan dataset di folder dataset/ dan dataset_kematangan/)

# 5. Training model
python train_model.py
python train_kematangan.py
python train_tomat.py

# 6. Jalankan aplikasi
streamlit run main.py
```

---

## 👤 Penulis

**Akhmad Fatkhul Arifin**

<p align="center">
  © 2026 <strong>Vision AI</strong> — Sistem Identifikasi Buah & Bunga Otomatis
</p>