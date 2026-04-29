import os
import cv2
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_color_histogram(image_path):
    """Mengekstrak fitur warna HSV menjadi array angka (Histogram)"""
    image = cv2.imread(image_path)
    if image is None: return None
    image = cv2.resize(image, (100, 100))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def train_and_save_model(dataset_path, model_name):
    print(f"\n Memulai training dari folder: {dataset_path}...")
    X, y = [], []
    
    if not os.path.exists(dataset_path):
        print(f"Folder {dataset_path} tidak ditemukan!")
        return

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"   Mengekstrak data: {folder_name}")
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                fitur = extract_color_histogram(image_path)
                if fitur is not None:
                    X.append(fitur)
                    y.append(folder_name)

    if len(X) == 0:
        print("Tidak ada gambar valid untuk ditraining!")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih model KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Uji akurasi
    akurasi = accuracy_score(y_test, model.predict(X_test))
    print(f" Training selesai! Akurasi Model: {akurasi * 100:.2f}%")
    
    # Simpan model
    joblib.dump(model, model_name)
    print(f" Model berhasil disimpan sebagai {model_name}\n")

if __name__ == "__main__":
    train_and_save_model("dataset/buah", "model_buah.pkl")
    train_and_save_model("dataset/flowers", "model_bunga.pkl")