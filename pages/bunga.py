import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Deteksi Bunga", page_icon="🌻")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.write("### Menu Utama")
    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Buah", "Bunga", "Kematangan"],
        icons=["house", "apple", "flower1", "star-half"],
        default_index=2, 
        styles={
            "container": {"background-color": "transparent", "padding": "0px"},
            "icon": {"color": "#facc15", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px 0px"},
            "nav-link-selected": {"background-color": "#262640", "border-left": "4px solid #facc15"},
        }
    )

if selected == "Beranda":
    st.switch_page("main.py")
elif selected == "Buah":
    st.switch_page("pages/buah.py")
elif selected == "Kematangan":
    st.switch_page("pages/kematangan.py")

# Load model secara dinamis
@st.cache_resource
def load_model():
    return joblib.load("model_bunga.pkl")

try:
    model_bunga = load_model()
except FileNotFoundError:
    st.error("File 'model_bunga.pkl' tidak ditemukan. Jalankan 'python train_model.py' terlebih dahulu.")
    st.stop()

def extract_features(img_bgr):
    img_resized = cv2.resize(img_bgr, (100, 100))
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

st.title("🌻 Identifikasi Jenis Bunga")
st.write("Sistem mendeteksi jenis bunga secara otomatis menggunakan ekstraksi fitur warna (HSV).")

tab1, tab2 = st.tabs(["Unggah Foto", "Ambil dari Kamera"])

with tab1:
    uploaded_file = st.file_uploader("Unggah Foto Bunga (JPG/PNG)", type=["jpg", "jpeg", "png"])
with tab2:
    camera_file = st.camera_input("Ambil Foto Langsung via Webcam")

image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:
    # Membaca gambar
    image = Image.open(image_source)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Visualisasi HSV
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

    # Proses Deteksi dan Visualisasi Hasil
    with st.spinner("Mengekstrak histogram warna dan menganalisis..."):
        fitur = extract_features(img_bgr)

        prediksi = model_bunga.predict([fitur])[0]
        
        # Desain Card Custom menggunakan HTML & CSS (TEMA KUNING)
        html_result = f"""
        <div style="
            background: linear-gradient(135deg, rgba(250, 204, 21, 0.15) 0%, rgba(255, 255, 255, 0) 100%);
            border: 1px solid rgba(250, 204, 21, 0.3);
            border-left: 8px solid #facc15;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        ">
            <p style="font-size: 1.1rem; color: #b0b0b0; margin-bottom: 5px; font-family: sans-serif; letter-spacing: 1px;">
                KESIMPULAN HASIL KLASIFIKASI
            </p>
            <h1 style="
                font-size: 4rem; 
                color: #facc15; 
                margin: 0px; 
                text-transform: uppercase; 
                letter-spacing: 3px; 
                font-weight: 900; 
                text-shadow: 2px 2px 8px rgba(250, 204, 21, 0.3);
            ">
                {prediksi}
            </h1>
            <hr style="border: none; border-top: 1px solid rgba(250, 204, 21, 0.2); margin: 20px 0 15px 0;">
            <p style="font-size: 0.95rem; color: #888888; margin: 0; font-family: sans-serif;">
                <i>Sistem mengidentifikasi pola histogram warna (HSV) yang paling identik dengan dataset referensi.</i>
            </p>
        </div>
        """
        # Render HTML ke dalam Streamlit
        st.markdown(html_result, unsafe_allow_html=True)
        
        st.write("### 📊 Distribusi Intensitas Warna (HSV)")
        st.caption("Grafik tumpang tindih ini menunjukkan sebaran piksel berdasarkan warna (Hue), kepekatan (Saturation), dan kecerahan (Value).")
        
        import pandas as pd
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

        # Menampilkan chart dengan warna kustom
        st.area_chart(chart_data, color=["#b175c4", "#77c4a3", "#5c8cd6"])