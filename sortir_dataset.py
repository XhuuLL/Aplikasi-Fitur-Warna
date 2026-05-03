import os
import pandas as pd
import shutil

# Lokasi folder
FOLDER_TRAIN = r"D:\Project-Python\Aplikasi Fitur Warna\dataset_kematangan\Tomato Ripeness.v2i.multiclass\train"
FILE_CSV = os.path.join(FOLDER_TRAIN, "_classes.csv")

print(f"Membaca file _classes.csv dari:\n{FILE_CSV}")

try:
    # Membaca file CSV
    df = pd.read_csv(FILE_CSV)
except FileNotFoundError:
    print(f"Error: File _classes.csv tidak ditemukan di {FOLDER_TRAIN}")
    exit()

df.columns = df.columns.str.strip()

daftar_kelas = df.columns[1:]
print(f" Ditemukan kelas: {list(daftar_kelas)}")

for nama_kelas in daftar_kelas:
    path_folder_kelas = os.path.join(FOLDER_TRAIN, nama_kelas.strip())
    os.makedirs(path_folder_kelas, exist_ok=True)

print("🚀 Mulai memindahkan gambar. Tunggu sebentar...")

berhasil = 0
for index, baris in df.iterrows():
    nama_file = baris['filename'].strip()

    for nama_kelas in daftar_kelas:
        if baris[nama_kelas] == 1: 
            sumber = os.path.join(FOLDER_TRAIN, nama_file)
            tujuan = os.path.join(FOLDER_TRAIN, nama_kelas.strip(), nama_file)

            if os.path.exists(sumber):
                shutil.move(sumber, tujuan)
                berhasil += 1
            break 

print(f"Selesai! Berhasil merapikan {berhasil} gambar ke dalam foldernya masing-masing.")