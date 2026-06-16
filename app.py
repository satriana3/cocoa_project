import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

st.set_page_config(
    page_title="Cocoa Disease Detector",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background: #0e1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }
    .hero {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #263244;
        padding: 24px 28px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }
    .hero h1 { color: #ffffff; margin-bottom: 6px; font-size: 2.1rem; }
    .hero p { color: #cbd5e1; margin: 0; font-size: 1rem; }
    .card {
        background: #161b22;
        border: 1px solid #263244;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }
    .metric-label { color: #9ca3af; font-size: 0.9rem; margin-bottom: 2px; }
    .metric-value { color: #ffffff; font-size: 1.6rem; font-weight: 700; }
    .small-note { color: #94a3b8; font-size: 0.9rem; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #334155;
        border-radius: 14px;
        padding: 10px;
        background: #111827;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #111827;
        border-radius: 12px 12px 0 0;
        padding: 10px 16px;
        border: 1px solid #263244;
    }
    .stTabs [aria-selected="true"] { background: #1f2937; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = (256, 256)
CLASS_NAMES = ["black_pod_rot", "healthy", "pod_borer"]

@st.cache_resource
def load_model():
    return joblib.load("cocoa_disease_model.pkl")

def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_colormap(arr, cmap_name="hsv"):
    arr = arr.astype(np.float32)
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    colored = plt.get_cmap(cmap_name)(norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def preprocess_image(img, size=IMG_SIZE):
    original = img.copy()
    resized = cv2.resize(img, size)
    blur = cv2.GaussianBlur(resized, (5, 5), 0)

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return original, resized, blur, enhanced, hsv, gray

def segment_disease(hsv):
    lower1 = np.array([0, 30, 20])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 30, 20])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def extract_features(img_bgr):
    original, resized, blur, enhanced, hsv, gray = preprocess_image(img_bgr)
    mask = segment_disease(hsv)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = circularity = solidity = extent = aspect_ratio = 0.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))

        if perimeter > 0:
            circularity = float(4 * np.pi * area / (perimeter ** 2))

        x, y, w, h = cv2.boundingRect(c)
        if w > 0 and h > 0:
            aspect_ratio = float(w / h)
            extent = float(area / (w * h))

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = float(area / hull_area)

    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    h_std = float(np.std(hsv[:, :, 0]))
    s_std = float(np.std(hsv[:, :, 1]))
    v_std = float(np.std(hsv[:, :, 2]))

    lbp = local_binary_pattern(masked_gray, P=8, R=1, method="uniform")
    lbp_mean = float(lbp.mean())
    lbp_std = float(lbp.std())

    features = np.array([
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        area, perimeter, circularity,
        solidity, extent, aspect_ratio,
        lbp_mean, lbp_std
    ], dtype=np.float32)

    return features, original, resized, blur, enhanced, hsv, mask

def show_preprocessing(image_path):
    img = cv2.imread(image_path)
    if img is None:
        st.error("Gambar tidak ditemukan.")
        return

    original, resized, blur, enhanced, hsv, gray = preprocess_image(img)
    mask = segment_disease(hsv)

    original_rgb = rgb(original)
    resized_rgb = rgb(resized)
    blur_rgb = rgb(blur)
    enhanced_rgb = rgb(enhanced)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h_color = apply_colormap(h, "hsv")
    masked = cv2.bitwise_and(enhanced_rgb, enhanced_rgb, mask=mask)

    c1, c2 = st.columns(2)

    with c1:
        st.image(original_rgb, caption="Original", use_container_width=True)
        st.image(resized_rgb, caption="Resized", use_container_width=True)
        st.image(blur_rgb, caption="Gaussian Blur", use_container_width=True)
        st.image(enhanced_rgb, caption="CLAHE Enhanced", use_container_width=True)

    with c2:
        st.image(h_color, caption="HSV - Hue", use_container_width=True)
        st.image(s, caption="HSV - Saturation", clamp=True, use_container_width=True)
        st.image(v, caption="HSV - Value", clamp=True, use_container_width=True)
        st.image(mask, caption="Disease Mask", clamp=True, use_container_width=True)
        st.image(masked, caption="Masked Result", use_container_width=True)

def display_metrics(pred, confidence, n_features):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'><div class='metric-label'>Prediksi Kelas</div><div class='metric-value'>{pred}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='metric-label'>Confidence</div><div class='metric-value'>{confidence*100:.2f}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='metric-label'>Jumlah Fitur</div><div class='metric-value'>{n_features}</div></div>", unsafe_allow_html=True)

st.sidebar.title("Informasi Aplikasi")
st.sidebar.markdown("""
**Model:** Random Forest  
**Fitur:** warna, bentuk, tekstur  
**Kelas:** black_pod_rot, healthy, pod_borer

**Alur kerja:**
1. Upload gambar.
2. Lihat preprocessing.
3. Cek prediksi dan confidence.
""")
st.sidebar.divider()
st.sidebar.caption("Aplikasi deteksi penyakit buah kakao berbasis pengolahan citra.")

st.markdown("""
<div class="hero">
    <h1>🍫 Cocoa Disease Detector</h1>
    <p>Klasifikasi penyakit buah kakao dengan preprocessing citra, ekstraksi fitur, dan Random Forest.</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
uploaded_file = st.file_uploader("Upload gambar buah kakao", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Gambar tidak dapat dibaca.")
        st.stop()

    features, original, resized, blur, enhanced, hsv, mask = extract_features(img_bgr)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h_color = apply_colormap(h, "hsv")
    masked_rgb = cv2.bitwise_and(rgb(enhanced), rgb(enhanced), mask=mask)

    pred = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    confidence = float(np.max(proba))

    display_metrics(pred, confidence, len(features))

    tab1, tab2, tab3 = st.tabs(["Preprocessing", "Probabilitas", "Fitur Numerik"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb(original), caption="Original", use_container_width=True)
            st.image(rgb(resized), caption="Resized", use_container_width=True)
            st.image(rgb(blur), caption="Gaussian Blur", use_container_width=True)
            st.image(rgb(enhanced), caption="CLAHE Enhanced", use_container_width=True)

        with col2:
            st.image(h_color, caption="HSV - Hue", use_container_width=True)
            st.image(s, caption="HSV - Saturation", clamp=True, use_container_width=True)
            st.image(v, caption="HSV - Value", clamp=True, use_container_width=True)
            st.image(mask, caption="Disease Mask", clamp=True, use_container_width=True)
            st.image(masked_rgb, caption="Masked Result", use_container_width=True)

    with tab2:
        proba_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)
        st.dataframe(proba_df, use_container_width=True)

    with tab3:
        feature_names = [
            "h_mean", "s_mean", "v_mean",
            "h_std", "s_std", "v_std",
            "area", "perimeter", "circularity",
            "solidity", "extent", "aspect_ratio",
            "lbp_mean", "lbp_std"
        ]
        feature_df = pd.DataFrame({"Fitur": feature_names, "Nilai": features})
        st.dataframe(feature_df, use_container_width=True)
else:
    st.info("Silakan upload gambar buah kakao terlebih dahulu.")