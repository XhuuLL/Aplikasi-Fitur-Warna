import streamlit as st
from streamlit_option_menu import option_menu
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Setup halaman dan menghilangkan sidebar bawaan
st.set_page_config(page_title="Aplikasi Fitur Warna", page_icon="", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Memasukkan Navbar Kustom ke dalam Sidebar
with st.sidebar:
    st.write("### Menu Utama")
    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Buah", "Bunga", "Kematangan"],
        icons=["house", "apple", "flower1", "star-half"],
        default_index=0, 
        styles={
            "container": {"background-color": "transparent", "padding": "0px"},
            "icon": {"color": "#ff4b4b", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px 0px"},
            "nav-link-selected": {"background-color": "#262640", "border-left": "4px solid #ff4b4b"},
        }
    )

# Logika Pindah Halaman
if selected == "Buah":
    st.switch_page("pages/buah.py")
elif selected == "Bunga":
    st.switch_page("pages/bunga.py")
elif selected == "Kematangan":
    st.switch_page("pages/kematangan.py")

col_text, col_image = st.columns([1.2, 1], gap="large")

with col_text:
    st.markdown("""
        <div style="padding-top: 40px;">
            <span style="background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 6px 16px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(255, 75, 75, 0.3);">
                Fruit & Flower Vision
            </span>
            <h1 style="font-size: 3.8rem; font-weight: 800; line-height: 1.1; margin-top: 25px; margin-bottom: 15px;">
                Sistem Identifikasi <br><span style="color: #ff4b4b;">Buah & Bunga</span><br> Otomatis.
            </h1>
            <p style="color: #a0aec0; font-size: 1.15rem; line-height: 1.6; margin-bottom: 30px;">
                Asisten pintar untuk mendeteksi dan menganalisis klasifikasi objek hanya dari sebuah foto. Mulai eksperimen ekstraksi fitur warna (HSV) dengan langkah yang mudah.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tombol interaktif
    if st.button("Mulai Deteksi Sekarang", type="primary"):
        st.switch_page("pages/buah.py")

with col_image:
    img_base64 = get_base64_image("zbungan.jpg")
    st.markdown(f"""
        <div style="margin-top: 85px; text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" 
                style="width: 100%; border-radius: 24px; box-shadow: 0 15px 35px rgba(0,0,0,0.4);">
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# Colom Arsitektur Sistem
st.markdown("""
    <h3 style="text-align: center; color: #ffffff; margin-bottom: 30px; font-weight: 600;">Arsitektur Sistem</h3>
""", unsafe_allow_html=True)
card1, card2, card3 = st.columns(3)

with card1:
    st.markdown("""
        <div style="background-color: #1e1e2f; padding: 25px; border-radius: 15px; border-top: 4px solid #ff4b4b; box-shadow: 0 8px 16px rgba(0,0,0,0.2); height: 100%;">
            <div style="font-size: 35px; margin-bottom: 15px;">📐</div>
            <h4 style="color: #ffffff; margin-top: 0; margin-bottom: 10px;">1. Preprocessing</h4>
            <p style="color: #888888; font-size: 14.5px; line-height: 1.5; margin: 0;">Standarisasi ukuran citra input dengan melakukan resize ke dimensi <strong>100x100 piksel</strong>.</p>
        </div>
    """, unsafe_allow_html=True)

with card2:
    st.markdown("""
        <div style="background-color: #1e1e2f; padding: 25px; border-radius: 15px; border-top: 4px solid #facc15; box-shadow: 0 8px 16px rgba(0,0,0,0.2); height: 100%;">
            <div style="font-size: 35px; margin-bottom: 15px;">🎨</div>
            <h4 style="color: #ffffff; margin-top: 0; margin-bottom: 10px;">2. Ekstraksi Fitur</h4>
            <p style="color: #888888; font-size: 14.5px; line-height: 1.5; margin: 0;">Transformasi ruang warna matriks citra dari BGR menjadi ruang warna <strong>HSV</strong>.</p>
        </div>
    """, unsafe_allow_html=True)

with card3:
    st.markdown("""
        <div style="background-color: #1e1e2f; padding: 25px; border-radius: 15px; border-top: 4px solid #3b82f6; box-shadow: 0 8px 16px rgba(0,0,0,0.2); height: 100%;">
            <div style="font-size: 35px; margin-bottom: 15px;">🧠</div>
            <h4 style="color: #ffffff; margin-top: 0; margin-bottom: 10px;">3. Model Klasifikasi</h4>
            <p style="color: #888888; font-size: 14.5px; line-height: 1.5; margin: 0;">Pencocokan pola histogram fitur warna menggunakan model <strong>K-Nearest Neighbors</strong>.</p>
        </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
    <div style="
        text-align: center; 
        padding: 20px 0 10px 0; 
        border-top: 1px solid rgba(255, 255, 255, 0.1); 
        margin-top: 60px;
    ">
        <p style="color: #a0aec0; font-size: 15px; margin: 0;">
            © 2026 <strong>Vision AI</strong>. Dibuat oleh Akhmad Fatkhul Arifin.
        </p>
        <p style="color: #666666; font-size: 13px; margin-top: 5px;">
            Program Studi Teknik Informatika | Universitas Muhadi Setiabudi
        </p>
    </div>
""", unsafe_allow_html=True)