"""
TIPAT VISION - Aplikasi Deteksi Varietas Tipat Bali berbasis YOLOv8
Proyek Akademik Pengolahan Citra Digital (Digital Image Processing)

Varietas Tipat yang dideteksi:
1. Tipat Bantal
2. Tipat Dampulan
3. Tipat Gong
4. Tipat Nasi
"""

from collections import Counter
import csv
from datetime import datetime
import glob
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="TIPAT VISION - Deteksi Objek Tipat Bali",
    page_icon="🍙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. DESIGN SYSTEM & CUSTOM CSS (Modern Light Mode)
# ==============================================================================
CUSTOM_CSS = """
<style>
/* Palette Warna TIPAT VISION:
   - Warm Ivory: #F7F7F2 (Background Utama)
   - Emerald Green: #176B52 (Primary Brand)
   - Soft Sage: #EAF2EC (Secondary / Card Background)
   - Warm Gold: #E9A44A (Accent Highlight)
   - Slate Blue: #263B50 (Primary Text)
   - Muted Gray: #718096 (Secondary Text)
*/

:root {
    --bg-ivory: #F7F7F2;
    --primary-emerald: #176B52;
    --emerald-dark: #0f4c3a;
    --emerald-light: #228466;
    --card-sage: #EAF2EC;
    --accent-gold: #E9A44A;
    --gold-light: #fef7ed;
    --text-slate: #263B50;
    --text-muted: #718096;
    --border-soft: #D9E4DC;
    --white: #FFFFFF;
}

/* Global Container Tweaks */
.stApp {
    background-color: var(--bg-ivory);
    color: var(--text-slate);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Hide default streamlit header decoration */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #EAF2EC 0%, #F5F9F6 100%);
    border: 1px solid var(--border-soft);
    border-left: 6px solid var(--primary-emerald);
    border-radius: 14px;
    padding: 1.5rem 1.85rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 16px rgba(23, 107, 82, 0.05);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background-color: var(--primary-emerald);
    color: #FFFFFF !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.28rem 0.8rem;
    border-radius: 9999px;
    margin-bottom: 0.65rem;
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--primary-emerald);
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.025em;
}

.hero-tagline {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-gold);
    margin-top: 0.35rem;
    margin-bottom: 0.45rem;
    letter-spacing: -0.01em;
}

.hero-desc {
    font-size: 0.96rem;
    color: var(--text-slate);
    margin: 0;
    line-height: 1.55;
    max-width: 860px;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #F2F7F4;
    border-right: 1px solid var(--border-soft);
}

.sidebar-brand {
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 1.25rem;
}

.sidebar-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--primary-emerald);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
}

.sidebar-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}

.sidebar-section-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin: 1.2rem 0 0.5rem 0;
}

.info-card {
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 0.9rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.83rem;
    padding: 0.25rem 0;
}

.info-label {
    color: var(--text-muted);
    font-weight: 500;
}

.info-value {
    color: var(--text-slate);
    font-weight: 600;
    text-align: right;
    word-break: break-all;
}

.badge-status {
    background: #EAF7EE;
    color: #176B52;
    border: 1px solid #B8E4C4;
    padding: 0.15rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

/* Tipat Variety Mini Badges */
.class-chip {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.25rem 0.2rem 0;
    background-color: var(--white);
    border: 1px solid var(--border-soft);
    color: var(--text-slate);
}

/* Metric / Stat Cards */
.metric-card {
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.025);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(23, 107, 82, 0.08);
}

.metric-value {
    font-size: 1.95rem;
    font-weight: 800;
    color: var(--primary-emerald);
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.metric-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Image Card Box */
.image-panel {
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    margin-bottom: 1.25rem;
}

.panel-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-slate);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.panel-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    background: var(--card-sage);
    color: var(--primary-emerald);
}

/* Class Breakdown Pill */
.class-summary-pill {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
}

.class-summary-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-slate);
}

.class-summary-count {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--primary-emerald);
    background: var(--card-sage);
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
}

/* Empty State Card */
.empty-state-box {
    background: #FFFDF9;
    border: 1px solid #F2DFBC;
    border-left: 5px solid var(--accent-gold);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(233, 164, 74, 0.08);
}

.empty-state-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #925C18;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.empty-state-desc {
    font-size: 0.92rem;
    color: var(--text-slate);
    line-height: 1.5;
    margin: 0;
}

/* Initial Guide Card */
.guide-box {
    background: var(--white);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    margin-top: 1rem;
}

/* Footer */
.footer-box {
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-soft);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.84rem;
    line-height: 1.6;
}

.footer-brand {
    font-weight: 700;
    color: var(--primary-emerald);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 3. DYNAMIC MODEL DISCOVERY & ROBUST LOADING
# ==============================================================================
def discover_model_candidates(runs_root: str = "runs/detect") -> List[Dict[str, Any]]:
    """
    Memindai folder `runs/detect/` untuk menemukan kandidat model `best.pt` hasil training valid.
    
    Aturan Ketat:
    - Hanya mencari di dalam subfolder `runs/detect/`.
    - Tidak pernah menggunakan `best.pt` lama di root project atau folder dataset.
    - Tidak mengasumsikan nama folder statis seperti `yolov8_ketupat_v6i`.
    - Mengurutkan kandidat berdasarkan waktu modifikasi terbaru (paling mutakhir).
    - Mengekstrak metadata (ukuran file, epoch, mAP@50 jika results.csv tersedia).
    """
    runs_path = Path(runs_root).resolve()
    if not runs_path.exists() or not runs_path.is_dir():
        return []

    candidates: List[Dict[str, Any]] = []

    # Cari file 'best.pt' di dalam struktur folder runs/detect
    for pt_file in runs_path.rglob("best.pt"):
        # Pastikan file berada di dalam runs_path
        try:
            pt_file.relative_to(runs_path)
        except ValueError:
            continue

        # Validasi bahwa file ada dan ukurannya tidak nol
        if not pt_file.is_file() or pt_file.stat().st_size == 0:
            continue

        # Tentukan direktori training utama
        if pt_file.parent.name == "weights":
            run_dir = pt_file.parent.parent
        else:
            run_dir = pt_file.parent

        mtime = pt_file.stat().st_mtime
        size_mb = pt_file.stat().st_size / (1024 * 1024)

        # Coba ekstrak metadata dari results.csv jika ada
        results_csv = run_dir / "results.csv"
        epochs: Optional[int] = None
        map50: Optional[float] = None
        if results_csv.exists():
            try:
                with open(results_csv, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        epochs = len(rows)
                        for col, val in rows[-1].items():
                            if "mAP50(B)" in col or "metrics/mAP50(B)" in col:
                                map50 = float(val.strip())
                                break
            except Exception:
                pass

        # Format path relatif yang rapi untuk tampilan
        try:
            rel_display = str(pt_file.relative_to(Path.cwd())).replace("\\", "/")
        except Exception:
            rel_display = str(pt_file).replace("\\", "/")

        candidates.append({
            "path": str(pt_file.resolve()),
            "rel_path": rel_display,
            "run_name": run_dir.name,
            "mtime": mtime,
            "date_str": datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M"),
            "size_mb": size_mb,
            "epochs": epochs,
            "map50": map50,
        })

    # Urutkan berdasarkan waktu modifikasi terbaru terlebih dahulu
    candidates.sort(key=lambda c: c["mtime"], reverse=True)
    return candidates


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_file_path: str) -> YOLO:
    """
    Memuat model YOLOv8 menggunakan cache Streamlit agar efisien dan tidak reload berulang kali.
    """
    return YOLO(model_file_path)


# ==============================================================================
# 4. DISKUSI & VERIFIKASI KANDIDAT MODEL
# ==============================================================================
model_candidates = discover_model_candidates("runs/detect")

if not model_candidates:
    st.error("🚨 **KRITIS: File model valid `best.pt` tidak ditemukan di dalam direktori `runs/detect/`!**")
    st.markdown(
        """
        Sistem tidak menemukan file bobot pelatihan valid `runs/detect/.../weights/best.pt`.
        
        **Tindakan yang diperlukan:**
        1. Pastikan Anda telah menyelesaikan pelatihan YOLOv8.
        2. Simpan atau pastikan file `best.pt` berada di folder output pelatihan di dalam `runs/detect/<nama_training>/weights/best.pt`.
        3. Sistem **tidak akan** menggunakan file `best.pt` lama di root project untuk mencegah bias atau penggunaan model kadaluwarsa.
        """
    )
    st.stop()


# ==============================================================================
# 5. SIDEBAR: PENGATURAN DETEKSI & METADATA MODEL
# ==============================================================================
with st.sidebar:
    # Branding di Sidebar
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2 class="sidebar-title">🍙 TIPAT VISION</h2>
            <div class="sidebar-sub">Balinese Traditional Tipat Detection</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Bagian Informasi Model
    st.markdown('<div class="sidebar-section-title">Informasi Model AI</div>', unsafe_allow_html=True)

    # Pemilihan model jika ada beberapa kandidat training
    if len(model_candidates) > 1:
        model_options = [
            f"{c['run_name']} ({c['date_str']})" for c in model_candidates
        ]
        selected_idx = st.selectbox(
            "Pilih Bobot Model (runs/detect):",
            range(len(model_candidates)),
            format_func=lambda i: model_options[i],
            help="Sistem secara default memilih hasil training paling mutakhir."
        )
        active_candidate = model_candidates[selected_idx]
    else:
        active_candidate = model_candidates[0]

    # Muat model terpilih
    try:
        yolo_model = load_yolo_model(active_candidate["path"])
        model_load_error = None
    except Exception as e:
        yolo_model = None
        model_load_error = str(e)

    if model_load_error:
        st.error(f"Gagal memuat model: {model_load_error}")
        st.stop()

    # Model Info Card
    map_display = f"{active_candidate['map50']:.1%}" if active_candidate["map50"] is not None else "Tersedia"
    epoch_display = f"{active_candidate['epochs']} Epochs" if active_candidate["epochs"] is not None else "Selesai"

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-row">
                <span class="info-label">Status</span>
                <span class="badge-status">● Aktif & Siap</span>
            </div>
            <div class="info-row">
                <span class="info-label">Folder Run</span>
                <span class="info-value"><code>{active_candidate['run_name']}</code></span>
            </div>
            <div class="info-row">
                <span class="info-label">Ukuran Model</span>
                <span class="info-value">{active_candidate['size_mb']:.1f} MB</span>
            </div>
            <div class="info-row">
                <span class="info-label">Diperbarui</span>
                <span class="info-value">{active_candidate['date_str']}</span>
            </div>
            <div class="info-row">
                <span class="info-label">mAP@50</span>
                <span class="info-value" style="color: #176B52;">{map_display}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Bagian Parameter Deteksi
    st.markdown('<div class="sidebar-section-title">Parameter Inferensi</div>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Batas Kepercayaan (Confidence):",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Batas kepercayaan minimum agar objek diakui. Nilai standar 0.25 memberikan keseimbangan optimal antara sensitivitas dan ketepatan."
    )

    st.caption(
        "💡 *Turunkan nilai confidence jika objek belum terdeteksi, atau naikkan jika terdapat prediksi berlebih.*"
    )

    # Inference Augmentation (TTA) Toggle (Default: True untuk presisi tinggi)
    use_augmentation = st.checkbox(
        "Gunakan Test-Time Augmentation (TTA)",
        value=True,
        help="Mempertahankan setting augment=True untuk meningkatkan stabilitas dan presisi deteksi pada variasi orientasi tipat."
    )

    # 3. Bagian Informasi Kelas Varietas
    st.markdown('<div class="sidebar-section-title">Varietas Tipat Terdaftar</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 0.8rem;">
            <span class="class-chip">🍙 Tipat Bantal</span>
            <span class="class-chip">🍙 Tipat Dampulan</span>
            <span class="class-chip">🍙 Tipat Gong</span>
            <span class="class-chip">🍙 Tipat Nasi</span>
        </div>
        <div style="font-size: 0.78rem; color: #718096; line-height: 1.45;">
            Empat ragam tipat tradisional khas Bali yang dipelajari oleh model deep learning YOLOv8 berdasarkan bentuk anyaman janurnya.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# 6. HEADER / HERO SECTION
# ==============================================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-pill">
            <span>🌾</span> COMPUTER VISION • YOLOV8 DEEP LEARNING
        </div>
        <h1 class="hero-title">TIPAT VISION</h1>
        <div class="hero-tagline">"Discover the Beauty of Balinese Tipat"</div>
        <p class="hero-desc">
            Sistem deteksi objek berbasis kecerdasan buatan untuk mengenali dan mengklasifikasikan 
            aneka ragam varietas <strong>Tipat Tradisional Bali</strong> secara otomatis, cepat, dan presisi.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 7. WORKSPACE UNGGAH CITRA (UTAMA)
# ==============================================================================
col_upload, col_sample = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown("### 📤 Unggah Citra Tipat")
    st.caption("Pilih atau seret foto Tipat Bali yang ingin dianalisis. Format didukung: **JPG, JPEG, PNG**.")
    uploaded_file = st.file_uploader(
        "Pilih file foto:",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Unggah foto dengan pencahayaan yang memadai untuk hasil deteksi terbaik."
    )

with col_sample:
    st.markdown("### 💡 Uji Cepat dengan Contoh")
    st.caption("Gunakan citra sampel dari dataset pengujian untuk demonstrasi langsung:")

    # Cari beberapa sampel dari test dataset jika folder tersedia
    sample_dir = Path("Ketupat.v6i.yolov8/test/images")
    available_samples = []
    if sample_dir.exists():
        available_samples = sorted(list(sample_dir.glob("*.jpg")))[:6]

    if "selected_sample" not in st.session_state:
        st.session_state["selected_sample"] = None

    if available_samples:
        sample_cols = st.columns(2)
        sample_labels = [
            ("Sampel 1 (Bantal)", 0),
            ("Sampel 2 (Dampulan)", 2),
            ("Sampel 3 (Gong)", 3),
            ("Sampel 4 (Nasi)", 5),
        ]
        for idx, (label, s_idx) in enumerate(sample_labels):
            if s_idx < len(available_samples):
                with sample_cols[idx % 2]:
                    if st.button(label, use_container_width=True, key=f"btn_sample_{idx}"):
                        st.session_state["selected_sample"] = str(available_samples[s_idx])
    else:
        st.info("Silakan unggah foto tipat Anda melalui kotak di sebelah kiri.")


# Menentukan sumber citra yang aktif
active_image: Optional[Image.Image] = None
source_name = ""

if uploaded_file is not None:
    # Jika user mengunggah file baru, utamakan file unggahan
    try:
        active_image = Image.open(uploaded_file).convert("RGB")
        source_name = uploaded_file.name
        # Reset sample button state saat user upload
        st.session_state["selected_sample"] = None
    except Exception as e:
        st.error(f"Gagal membaca gambar yang diunggah: {e}")
elif st.session_state["selected_sample"] is not None:
    # Jika user menekan tombol sampel uji
    try:
        sample_path = st.session_state["selected_sample"]
        active_image = Image.open(sample_path).convert("RGB")
        source_name = Path(sample_path).name
    except Exception as e:
        st.error(f"Gagal membaca citra sampel: {e}")


# ==============================================================================
# 8. HASIL DETEKSI & RINGKASAN INFERENSI
# ==============================================================================
if active_image is not None:
    st.markdown("---")
    
    # 1. Jalankan Inferensi YOLOv8 dengan timing
    img_np = np.array(active_image)
    
    t_start = time.perf_counter()
    results = yolo_model(
        img_np, 
        conf=conf_threshold, 
        augment=use_augmentation, 
        verbose=False
    )
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000

    res = results[0]
    res_plotted = res.plot()  # Mengembalikan array numpy visualisasi bounding box resmi YOLOv8
    boxes = res.boxes

    # Ekstraksi hasil deteksi aktual
    detected_items: List[Dict[str, Any]] = []
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf_val = float(b.conf[0].item())
            cls_name = yolo_model.names.get(cls_id, f"Kelas {cls_id}")
            xyxy = [round(float(coord), 1) for coord in b.xyxy[0].tolist()]
            detected_items.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf_val,
                "box": xyxy,
            })

    total_detected = len(detected_items)
    class_counts = Counter([item["class_name"] for item in detected_items])
    unique_classes = len(class_counts)
    max_confidence = max([item["confidence"] for item in detected_items]) if detected_items else 0.0

    # 2. Ringkasan Statistik Deteksi (Metric Cards)
    st.markdown("### 📊 Ringkasan Hasil Deteksi")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_detected}</div>
                <div class="metric-label">Total Objek Terdeteksi</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{unique_classes}</div>
                <div class="metric-label">Varietas Teridentifikasi</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{max_confidence:.1%}</div>
                <div class="metric-label">Keyakinan Tertinggi</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{t_elapsed_ms:.0f} ms</div>
                <div class="metric-label">Kecepatan Inferensi</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")  # Spacing

    # 3. Tampilan Gambar Asli & Hasil Deteksi Bounding Box Side-by-Side
    col_orig, col_det = st.columns(2, gap="medium")

    with col_orig:
        st.markdown(
            f"""
            <div class="panel-title">
                <span>📷 Citra Asli (Input)</span>
                <span class="panel-badge">{active_image.width} × {active_image.height} px</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(
            active_image,
            caption=f"File: {source_name}",
            use_container_width=True
        )

    with col_det:
        st.markdown(
            f"""
            <div class="panel-title">
                <span>🎯 Hasil Bounding Box YOLOv8</span>
                <span class="panel-badge">Conf ≥ {conf_threshold:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(
            res_plotted,
            caption=f"Prediksi YOLOv8 ({total_detected} objek terdeteksi)",
            use_container_width=True
        )

    # 4. Detail Objek Terdeteksi / Status Jika Tidak Ada Objek
    if total_detected > 0:
        st.markdown("#### 📋 Rincian Objek per Varietas")
        
        detail_col1, detail_col2 = st.columns([1, 2], gap="large")

        with detail_col1:
            st.markdown("**Frekuensi per Varietas:**")
            for cls_name, count in class_counts.items():
                st.markdown(
                    f"""
                    <div class="class-summary-pill">
                        <span class="class-summary-name">🍙 {cls_name}</span>
                        <span class="class-summary-count">{count} buah</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with detail_col2:
            st.markdown("**Daftar Deteksi Bounding Box:**")
            table_rows = []
            for idx, item in enumerate(detected_items, 1):
                box_str = f"[{item['box'][0]}, {item['box'][1]}, {item['box'][2]}, {item['box'][3]}]"
                table_rows.append({
                    "No": idx,
                    "Varietas Tipat": item["class_name"],
                    "Tingkat Keyakinan": f"{item['confidence']:.2%}",
                    "Koordinat [x1, y1, x2, y2]": box_str
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

    else:
        # State jika tidak ada deteksi objek
        st.markdown(
            f"""
            <div class="empty-state-box">
                <div class="empty-state-title">
                    <span>🔍</span> Tidak Ada Objek Tipat yang Terdeteksi
                </div>
                <p class="empty-state-desc">
                    Model tidak menemukan objek tipat dengan ambang batas keyakinan saat ini (<strong>Confidence: {conf_threshold:.2f}</strong>).<br><br>
                    <strong>Saran untuk Anda:</strong><br>
                    1. <strong>Turunkan batas kepercayaan</strong> pada slider di sidebar (misalnya menjadi 0.15 - 0.20).<br>
                    2. Pastikan foto memiliki pencahayaan cukup dan anyaman tipat terlihat jelas.<br>
                    3. Unggah foto dengan jarak pengambilan yang lebih dekat atau objek tidak tertutup bayangan.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    # Initial State sebelum ada gambar yang diunggah
    st.markdown(
        """
        <div class="guide-box">
            <h3 style="color: #176B52; margin-top: 0; font-size: 1.15rem; font-weight: 700;">
                📖 Panduan Singkat Penggunaan TIPAT VISION
            </h3>
            <div style="font-size: 0.93rem; color: #263B50; line-height: 1.6;">
                <p>
                    Aplikasi ini dirancang untuk mendemonstrasikan kemampuan Computer Vision berbasis <strong>YOLOv8</strong> 
                    dalam mengenali empat ragam Tipat tradisional Bali secara real-time.
                </p>
                <ol style="margin-left: 1.2rem; padding-left: 0;">
                    <li><strong>Unggah Gambar:</strong> Pilih file foto dari perangkat Anda melalui kotak di atas, atau klik salah satu tombol sampel uji coba cepat.</li>
                    <li><strong>Atur Batas Keyakinan:</strong> Gunakan slider di sidebar untuk menyesuaikan ambang batas confidence (standar: 0.25).</li>
                    <li><strong>Amati Hasil:</strong> Bounding box dan skor keyakinan untuk setiap varietas tipat akan ditampilkan secara komparatif.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# 9. FOOTER
# ==============================================================================
st.markdown(
    f"""
    <div class="footer-box">
        <span class="footer-brand">TIPAT VISION</span> — Traditional Balinese Tipat Recognition System<br>
        Proyek Akademik Pengolahan Citra Digital (Digital Image Processing) • Berbasis YOLOv8 Object Detection<br>
        <span style="font-size: 0.78rem;">Model Terpilih: <code>{active_candidate['rel_path']}</code> (Diperbarui: {active_candidate['date_str']})</span>
    </div>
    """,
    unsafe_allow_html=True
)
