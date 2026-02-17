import streamlit as st
import cv2
import numpy as np
from compressor import JPEGCompressor, ImageMetrics
from streamlit_image_comparison import image_comparison
import io

# Matrice de quantification standard
Q_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

st.set_page_config(page_title="DCT Image Compression", layout="wide")
st.title("Compression d'image par DCT")

# Sidebar pour les réglages
st.sidebar.header("Paramètres")
quality = st.sidebar.slider("Facteur de qualité (Q-multiplier)", 0.1, 5.0, 1.0)

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    img_rgb = img_rgb[:h-h%8, :w-w%8]
    
    # Traitement
    compressor = JPEGCompressor(Q_LUMINANCE * quality)
    img_centered = img_rgb.astype(np.float64) - 128
    
    img_quantized = compressor.process_image(img_centered, mode='compress')
    img_reconstructed = compressor.process_image(img_quantized, mode='decompress')
    
    img_final = np.clip(img_reconstructed + 128, 0, 255).astype(np.uint8)
    
    # Métriques
    err = ImageMetrics.l2_relative_error(img_centered, img_reconstructed)
    rate = ImageMetrics.compression_rate(img_quantized)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Compression", f"{rate*100:.2f}%")
    col2.metric("Erreur L2 relative", f"{err*100:.2f}%")
    
    # Préparation du fichier pour le téléchargement
    res_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", res_bgr)
    io_buf = io.BytesIO(buffer)

    col3.download_button(
        label="Télécharger l'image compressée",
        data=io_buf,
        file_name="compressed_image.jpg",
        mime="image/jpeg"
    )

    # Comparaison visuelle
    st.subheader("Comparaison Interactive")
    image_comparison(
        img1=img_rgb,
        img2=img_final,
        label1="Originale",
        label2="Compressée",
        width=1000,
        starting_position=50
    )