import os
import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(image_path):
    """Mengekstrak fitur Warna (HSV) + Bentuk (Hu Moments)"""
    image = cv2.imread(image_path)
    if image is None: return None
    
    img_resized = cv2.resize(image, (100, 100))
    
    # Fitur Warna (HSV)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    color_features = hist.flatten()
    
    # Fitur Bentuk (Hu Moments)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log scale & normalisasi
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    cv2.normalize(hu_moments, hu_moments, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Gabungkan Warna dan Bentuk
    combined_features = np.hstack((color_features, hu_moments))
    return combined_features

def train_and_save_tomato_model(dataset_path, model_name):
    LABEL_MAP = {
        "not_ripe": "Mentah",
        "half_ripe": "Setengah Matang",
        "ripe": "Matang"
    }

    print(f"\n Memulai training kematangan Tomat dari: {dataset_path}...")
    X, y = [], []
    
    if not os.path.exists(dataset_path):
        print(f"Folder {dataset_path} tidak ditemukan!")
        return

    # Looping folder sesuai mapping
    for folder_name, indo_label in LABEL_MAP.items():
        folder_path = os.path.join(dataset_path, folder_name)
        
        if os.path.isdir(folder_path):
            print(f" Mengekstrak data: {indo_label}...")
            count = 0
            for image_name in os.listdir(folder_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, image_name)
                    fitur = extract_features(image_path)
                    if fitur is not None:
                        X.append(fitur)
                        y.append(indo_label)
                        count += 1
            print(f" Berhasil memproses {count} gambar.")

    if len(X) == 0:
        print("Tidak ada gambar valid untuk ditraining!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Melatih Model KNN dengan total {len(X)} data...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    akurasi = accuracy_score(y_test, model.predict(X_test))
    print(f"Training selesai! Akurasi Model Tomat: {akurasi * 100:.2f}%")
    
    joblib.dump(model, model_name)
    print(f"💾 Model berhasil disimpan sebagai {model_name}\n")

if __name__ == "__main__":
    # Lokasi dataset Tomat yang baru saja kamu sortir
    PATH_TOMAT = r"D:\Project-Python\Aplikasi Fitur Warna\dataset_kematangan\Tomato Ripeness.v2i.multiclass\train"
    
    # Jalankan training dan simpan sebagai 'model_tomat.pkl'
    train_and_save_tomato_model(PATH_TOMAT, "model_tomat.pkl")