import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image

# Setup Halaman
st.set_page_config(page_title="Deteksi Kematangan", page_icon="", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.write("### Menu Utama")
    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Buah", "Bunga", "Kematangan"],
        icons=["house", "apple", "flower1", "star-half"],
        default_index=3,
        styles={
            "container": {"background-color": "transparent", "padding": "0px"},
            "icon": {"color": "#fd7e14", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px 0px"},
            "nav-link-selected": {"background-color": "#262640", "border-left": "4px solid #fd7e14"},
        }
    )

if selected == "Beranda":
    st.switch_page("main.py")
elif selected == "Buah":
    st.switch_page("pages/buah.py")
elif selected == "Bunga":
    st.switch_page("pages/bunga.py")

# --- HEADER & DROPDOWN PILIHAN BUAH ---
st.title("🎯 Analisis Kematangan Buah")
st.write("Pilih jenis buah dan unggah foto untuk mengetahui fase kematangannya secara instan.")

pilihan_buah = st.selectbox(
    "Pilih Komoditas Buah:",
    ["Pisang", "Tomat"],
    index=0
)

# --- LOAD MODEL DINAMIS ---
@st.cache_resource
def load_model(nama_buah):
    if nama_buah == "Pisang":
        return joblib.load("model_kematangan.pkl")
    else:
        return joblib.load("model_tomat.pkl")

try:
    model_aktif = load_model(pilihan_buah)
except FileNotFoundError:
    st.error(f"⚠️ File model untuk '{pilihan_buah}' tidak ditemukan. Pastikan kamu sudah melakukan training!")
    st.stop()

# --- FUNGSI EKSTRAKSI FITUR (DIPISAH AGAR TIDAK ERROR) ---
def extract_features_pisang(img_bgr):
    """Hanya menggunakan HSV (Sesuai model pisang lamamu)"""
    img_resized = cv2.resize(img_bgr, (100, 100))
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features_tomat(img_bgr):
    """Menggunakan gabungan HSV + Hu Moments (Sesuai model tomat barumu)"""
    img_resized = cv2.resize(img_bgr, (100, 100))
    
    # Warna
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    color_features = hist.flatten()
    
    # Bentuk
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    cv2.normalize(hu_moments, hu_moments, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return np.hstack((color_features, hu_moments))

# --- TAB UNGGAH GAMBAR ---
tab1, tab2 = st.tabs(["Unggah Foto", "Ambil dari Kamera"])

with tab1:
    uploaded_file = st.file_uploader(f"Unggah Foto {pilihan_buah} (JPG/PNG)", type=["jpg", "jpeg", "png"])
with tab2:
    camera_file = st.camera_input("Ambil Foto Langsung via Webcam")

image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:
    # Membaca gambar
    image = Image.open(image_source)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_hsv_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Menampilkan Gambar
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Citra Asli (RGB)")
        st.image(image, use_container_width=True) 
    with col2:
        st.caption("Ekstraksi Fitur (HSV)")
        st.image(img_hsv_display, use_container_width=True, clamp=True)
        
    st.divider()

    # Proses Deteksi
    with st.spinner(f"Menganalisis kematangan {pilihan_buah}..."):
        
        # Eksekusi fitur sesuai buah yang dipilih
        if pilihan_buah == "Pisang":
            fitur = extract_features_pisang(img_bgr)
        else:
            fitur = extract_features_tomat(img_bgr)
            
        prediksi = model_aktif.predict([fitur])[0]

        # --- LOGIKA WARNA DINAMIS SESUAI BUAH ---
        if pilihan_buah == "Pisang":
            if prediksi == "Sangat Matang":
                color_hex, bg_rgba, border_rgba = "#22c55e", "rgba(34, 197, 94, 0.15)", "rgba(34, 197, 94, 0.3)" # Hijau
            elif prediksi == "Matang":
                color_hex, bg_rgba, border_rgba = "#facc15", "rgba(250, 204, 21, 0.15)", "rgba(250, 204, 21, 0.3)" # Kuning
            elif prediksi == "Belum Matang":
                color_hex, bg_rgba, border_rgba = "#ef4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.3)" # Merah
            else: # Busuk / Rusak
                color_hex, bg_rgba, border_rgba = "#78350f", "rgba(120, 53, 15, 0.15)", "rgba(120, 53, 15, 0.3)" # Coklat
        
        elif pilihan_buah == "Tomat":
            if prediksi == "Mentah":
                color_hex, bg_rgba, border_rgba = "#22c55e", "rgba(34, 197, 94, 0.15)", "rgba(34, 197, 94, 0.3)" # Hijau
            elif prediksi == "Setengah Matang":
                color_hex, bg_rgba, border_rgba = "#f97316", "rgba(249, 115, 22, 0.15)", "rgba(249, 115, 22, 0.3)" # Oranye
            elif prediksi == "Matang":
                color_hex, bg_rgba, border_rgba = "#ef4444", "rgba(239, 68, 68, 0.15)", "rgba(239, 68, 68, 0.3)" # Merah

        # Desain Card Custom
        html_result = f"""
        <div style="
            background: linear-gradient(135deg, {bg_rgba} 0%, rgba(255, 255, 255, 0) 100%);
            border: 1px solid {border_rgba};
            border-left: 8px solid {color_hex};
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        ">
            <p style="font-size: 1.1rem; color: #b0b0b0; margin-bottom: 5px; font-family: sans-serif; letter-spacing: 1px;">
                STATUS KEMATANGAN {pilihan_buah.upper()}
            </p>
            <h1 style="
                font-size: 4rem; 
                color: {color_hex}; 
                margin: 0px; 
                text-transform: uppercase; 
                letter-spacing: 3px; 
                font-weight: 900; 
                text-shadow: 2px 2px 8px {bg_rgba};
            ">
                {prediksi}
            </h1>
            <hr style="border: none; border-top: 1px solid {border_rgba}; margin: 20px 0 15px 0;">
            <p style="font-size: 0.95rem; color: #888888; margin: 0; font-family: sans-serif;">
                <i>Sistem mengevaluasi ekstraksi fitur untuk menentukan fase kematangan.</i>
            </p>
        </div>
        """
        st.markdown(html_result, unsafe_allow_html=True)
        
        # Visualisasi Area Chart HSV
        st.write("### 📊 Distribusi Intensitas Warna (HSV)")
        st.caption("Grafik ini menunjukkan sebaran piksel yang mendasari keputusan klasifikasi kematangan.")
        
        hist_h = cv2.calcHist([img_hsv_display], [0], None, [256], [0, 256]).flatten()
        hist_s = cv2.calcHist([img_hsv_display], [1], None, [256], [0, 256]).flatten()
        hist_v = cv2.calcHist([img_hsv_display], [2], None, [256], [0, 256]).flatten()
        
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        chart_data = pd.DataFrame({
            "Hue (Warna)": hist_h,
            "Saturation (Kepekatan)": hist_s,
            "Value (Kecerahan)": hist_v
        })

        st.area_chart(chart_data, color=["#b175c4", "#77c4a3", "#5c8cd6"])