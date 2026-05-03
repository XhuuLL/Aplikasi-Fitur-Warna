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
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    color_features = hist.flatten()

    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Menghitung bentuk geometri
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    cv2.normalize(hu_moments, hu_moments, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    combined_features = np.hstack((color_features, hu_moments))
    
    return combined_features

def train_and_save_model(dataset_path, model_name):
    print(f"\n🚀 Memulai training dari folder: {dataset_path}...")
    X, y = [], []
    
    if not os.path.exists(dataset_path):
        print(f"Folder {dataset_path} tidak ditemukan!")
        return

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"   📂 Mengekstrak data Warna & Bentuk: {folder_name}")
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                fitur = extract_features(image_path)
                if fitur is not None:
                    X.append(fitur)
                    y.append(folder_name)

    if len(X) == 0:
        print("Tidak ada gambar valid!")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    akurasi = accuracy_score(y_test, model.predict(X_test))
    print(f"✨ Training selesai! Akurasi Model dengan Bentuk+Warna: {akurasi * 100:.2f}%")
    
    joblib.dump(model, model_name)
    print(f"💾 Model baru berhasil disimpan sebagai {model_name}\n")

if __name__ == "__main__":
    train_and_save_model("dataset/buah", "model_buah.pkl")