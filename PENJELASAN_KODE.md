# 📄 Dokumentasi Aplikasi Fitur Warna
## Penjelasan Kode & Fungsi untuk Presentasi

---

## 📁 Struktur Proyek

```
Aplikasi Fitur Warna/
├── dataset/                          # Dataset klasifikasi jenis objek
│   ├── buah/                         # 14 jenis buah (Alpukat, Apel, Jeruk, dll.)
│   └── flowers/                      # 5 jenis bunga (daisy, rose, sunflower, dll.)
├── dataset_kematangan/               # Dataset klasifikasi kematangan
│   ├── unripe/                       # Gambar buah belum matang
│   ├── ripe/                         # Gambar buah matang
│   ├── overripe/                     # Gambar buah sangat matang
│   ├── rotten/                       # Gambar buah busuk/rusak
│   └── Tomato Ripeness.v2i.multiclass/
│       └── train/                    # Dataset tomat dari Roboflow
│           ├── not_ripe/             # Tomat mentah
│           ├── half-ripe/            # Tomat setengah matang
│           └── ripe/                 # Tomat matang
├── train_model.py                    # Script training klasifikasi jenis buah & bunga
├── train_kematangan.py               # Script training kematangan buah (umum)
├── train_tomat.py                    # Script training kematangan tomat
└── zbungan.jpg                       # Contoh gambar uji
```

---

## 🧠 Algoritma yang Digunakan: K-Nearest Neighbors (KNN)

### Apa itu KNN?
KNN (K-Nearest Neighbors) adalah algoritma **Machine Learning** yang mengklasifikasikan data baru berdasarkan **kedekatan** (kemiripan) dengan data yang sudah dikenal.

### Cara Kerja:
1. Ambil data baru yang ingin diklasifikasikan
2. Hitung **jarak** data baru tersebut ke semua data training
3. Ambil **K tetangga terdekat** (dalam proyek ini K=5)
4. Klasifikasikan berdasarkan **voting mayoritas** dari K tetangga tersebut

### Mengapa K=5?
- Ganjil → Menghindari hasil seri (tie)
- Cukup kecil → Cepat dalam komputasi
- Cukup besar → Tidak terlalu sensitif terhadap noise

---

## 📝 Penjelasan Setiap File

---

### 1️⃣ `train_model.py` — Training Klasifikasi Jenis Buah & Bunga

**Tujuan:** Melatih model untuk mengenali **14 jenis buah** dan **5 jenis bunga** berdasarkan warna dan bentuknya. Script ini menjalankan training **dua kali** untuk menghasilkan dua model terpisah.

#### Fungsi-Fungsi:

##### `extract_features(image_path)`
Mengekstrak **dua jenis fitur** dari setiap gambar:

| Fitur | Penjelasan | Kegunaan |
|-------|-----------|----------|
| **Histogram Warna HSV** | Menghitung distribusi warna dalam ruang HSV (Hue, Saturation, Value) dengan 8×8×8 = 512 bin | Membedakan warna dominan setiap jenis buah |
| **Hu Moments** | 7 nilai yang mendeskripsikan bentuk geometri objek, bersifat invariant terhadap rotasi dan skala | Membedakan bentuk buah (bulat, lonjong, dll.) |

**Langkah-langkah dalam fungsi ini:**
1. Baca gambar dengan `cv2.imread()`
2. Resize ke 100×100 pixel (agar seragam)
3. Konversi ke ruang warna HSV
4. Hitung histogram warna → **512 fitur warna**
5. Konversi ke grayscale → thresholding Otsu
6. Hitung Hu Moments → **7 fitur bentuk**
7. Gabungkan menjadi **519 fitur total**

##### `train_and_save_model(dataset_path, model_name)`
Melatih model KNN dan menyimpan hasilnya.

**Langkah-langkah:**
1. Cek apakah folder dataset ada
2. Loop setiap subfolder (nama folder = label/jenis buah)
3. Ekstrak fitur dari setiap gambar `.jpg`, `.png`, `.jpeg`
4. Split data: **80% training, 20% testing**
5. Latih model KNN dengan `n_neighbors=5`
6. Hitung akurasi pada data test
7. Simpan model ke file `.pkl` dengan `joblib`

**Dataset yang digunakan:**
- `dataset/buah/` → 14 subfolder (Alpukat, Apel, Blackberry, Cherry, Jeruk, Kurma, Melon Oranye, Pear, Pepaya, Pir Berduri, Pisang, Raspberry, Strawberry, Cherimoya) → menghasilkan `model_buah.pkl`
- `dataset/flowers/` → 5 subfolder (daisy, dandelion, rose, sunflower, tulip) → menghasilkan `model_bunga.pkl`

---

### 2️⃣ `train_kematangan.py` — Training Kematangan Buah (Umum)

**Tujuan:** Melatih model untuk mengenali **tingkat kematangan** buah secara umum.

#### Fungsi-Fungsi:

##### `extract_color_histogram(image_path)`
Mengekstrak fitur **warna saja** (tanpa bentuk) dari gambar.

| Langkah | Penjelasan |
|---------|-----------|
| Baca gambar | `cv2.imread()` |
| Resize | 100×100 pixel |
| Konversi HSV | `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)` |
| Hitung histogram | 8×8×8 = 512 bin HSV |
| Normalisasi | `cv2.normalize()` |
| Flatten | Ubah ke array 1D (512 angka) |

> **Mengapa hanya warna?** Karena untuk menentukan kematangan, **perubahan warna** adalah indikator utama (hijau → kuning → coklat/hitam).

##### `train_and_save_maturity_model(dataset_path, model_name)`
Melatih model KNN untuk kematangan.

**Mapping Label (Bahasa Indonesia):**
| Folder Dataset | Label Prediksi |
|---------------|---------------|
| `unripe` | Belum Matang |
| `ripe` | Matang |
| `overripe` | Sangat Matang |
| `rotten` | Busuk / Rusak |

**Perbedaan dengan `train_model.py`:**
- Hanya menggunakan fitur **warna** (bukan warna + bentuk)
- Label di-mapping ke **Bahasa Indonesia**
- Hanya folder yang ada di `LABEL_MAP` yang diproses (lebih selektif)

---

### 3️⃣ `train_tomat.py` — Training Kematangan Tomat

**Tujuan:** Melatih model khusus untuk mengenali **3 tingkat kematangan tomat**.

#### Fungsi-Fungsi:

##### `extract_features(image_path)`
Sama persis dengan `train_model.py` — mengekstrak **warna + bentuk** (519 fitur total).

##### `train_and_save_tomato_model(dataset_path, model_name)`
Melatih model KNN khusus tomat.

**Mapping Label:**
| Folder Dataset | Label Prediksi |
|---------------|---------------|
| `not_ripe` | Mentah |
| `half-ripe` | Setengah Matang |
| `ripe` | Matang |

**Dataset:** Berasal dari **Roboflow** (Tomato Ripeness v2i multiclass), disimpan di `dataset_kematangan/Tomato Ripeness.v2i.multiclass/train/`

---

## 🔧 Library yang Digunakan

| Library | Fungsi | Contoh Penggunaan |
|---------|--------|------------------|
| `cv2` (OpenCV) | Pemrosesan gambar (baca, resize, ubah warna, histogram) | `cv2.imread()`, `cv2.cvtColor()`, `cv2.calcHist()` |
| `numpy` | Operasi matematika array/matriks | `np.hstack()`, `np.sign()`, `np.log10()` |
| `scikit-learn` | Algoritma Machine Learning (KNN, split data, akurasi) | `KNeighborsClassifier()`, `train_test_split()` |
| `joblib` | Menyimpan dan memuat model yang sudah dilatih | `joblib.dump()`, `joblib.load()` |
| `os` | Navigasi folder dan file system | `os.listdir()`, `os.path.join()` |

---

## 🔄 Alur Kerja (Pipeline)

```
┌─────────────────┐
│  Gambar Dataset  │
│  (.jpg/.png)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Preprocessing          │
│  • Resize 100×100       │
│  • Konversi ke HSV      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Ekstraksi Fitur        │
│  • Histogram Warna (512)│
│  • Hu Moments (7)*      │
│  *hanya untuk buah/tomat│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Split Data             │
│  • 80% Training         │
│  • 20% Testing          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Training KNN (K=5)     │
│  • Fit model            │
│  • Hitung akurasi       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Simpan Model (.pkl)    │
│  • model_buah.pkl       │
│  • model_bunga.pkl      │
│  • model_kematangan.pkl │
│  • model_tomat.pkl      │
└─────────────────────────┘
```

---

## 📊 Perbandingan Keempat Model

| Aspek | Model Buah | Model Bunga | Model Kematangan | Model Tomat |
|-------|-----------|------------|-----------------|------------|
| **Script** | `train_model.py` | `train_model.py` | `train_kematangan.py` | `train_tomat.py` |
| **Tujuan** | Klasifikasi jenis buah | Klasifikasi jenis bunga | Klasifikasi kematangan | Klasifikasi kematangan tomat |
| **Fitur** | Warna + Bentuk (519) | Warna + Bentuk (519) | Warna saja (512) | Warna + Bentuk (519) |
| **Jumlah Kelas** | 14 jenis buah | 5 jenis bunga | 4 tingkat kematangan | 3 tingkat kematangan |
| **Dataset** | `dataset/buah/` | `dataset/flowers/` | `dataset_kematangan/` | Roboflow Tomato Ripeness |
| **Output Model** | `model_buah.pkl` | `model_bunga.pkl` | `model_kematangan.pkl` | `model_tomat.pkl` |
| **Label** | Nama folder | Nama folder | Bahasa Indonesia | Bahasa Indonesia |

---

## 🎯 Ruang Warna HSV

Alasan menggunakan **HSV** bukan **RGB**:

| Komponen | Penjelasan | Keunggulan |
|----------|-----------|------------|
| **H** (Hue) | Jenis warna (0-180°) | Tidak terpengaruh pencahayaan |
| **S** (Saturation) | Kepekatan warna (0-255) | Membedakan warna pucat vs cerah |
| **V** (Value) | Kecerahan (0-255) | Menangani variasi pencahayaan |

> RGB sangat sensitif terhadap perubahan cahaya, sedangkan HSV memisahkan informasi warna dari kecerahan, sehingga lebih robust untuk klasifikasi berbasis warna.

---

## 💾 Format Model: `.pkl` (Pickle)

- Model yang sudah dilatih disimpan dalam format **Pickle** menggunakan library `joblib`
- File `.pkl` berisi seluruh state model KNN (data training, parameter, dll.)
- Bisa di-load kembali dengan `joblib.load("model.pkl")` untuk melakukan prediksi tanpa perlu training ulang
- Keuntungan: **cepat** untuk load dan tidak perlu training ulang setiap kali program dijalankan

---

## ✅ Ringkasan

Aplikasi ini menggunakan pendekatan **Computer Vision + Machine Learning** untuk:
1. **Mengenali 14 jenis buah** berdasarkan warna dan bentuk → `model_buah.pkl`
2. **Mengenali 5 jenis bunga** berdasarkan warna dan bentuk → `model_bunga.pkl`
3. **Mendeteksi 4 tingkat kematangan** buah berdasarkan perubahan warna → `model_kematangan.pkl`
4. **Mengklasifikasikan 3 tingkat kematangan tomat** secara spesifik → `model_tomat.pkl`

Semua model menggunakan algoritma **KNN (K=5)** dengan fitur yang diekstrak dari gambar menggunakan **OpenCV**, dan disimpan menggunakan **joblib** agar bisa digunakan kembali untuk prediksi.

### Total Cakupan Dataset:
| Dataset | Jumlah Kelas | Script |
|---------|-------------|--------|
| Buah | 14 jenis | `train_model.py` |
| Bunga | 5 jenis | `train_model.py` |
| Kematangan Umum | 4 level | `train_kematangan.py` |
| Kematangan Tomat | 3 level | `train_tomat.py` |
| **Total** | **26 kelas** | **3 script** |
