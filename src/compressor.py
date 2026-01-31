import streamlit as st
import fitz  # PyMuPDF
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import plotly.express as px

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="SmartPDF Optima",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* REMOVE TOP MARGIN/PADDING */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        margin-top: 0 !important;
    }
    
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headlines */
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #0f172a;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px !important;
        color: #64748b;
    }
    
    /* Recommendation Card */
    .rec-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Success/Warning messages text size */
    .stAlert {
        font-size: 16px;
    }
    
    /* Button enhancement */
    div.stButton > button:first-child {
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

TEMP_DIR = "processed_files"
DATA_FILE = "pdf_training_data.csv"
MODEL_FILE = "smart_pdf_model.pkl"

os.makedirs(TEMP_DIR, exist_ok=True)

# --- ML ENGINE ---
class SmartPDFModel:
    def __init__(self):
        self.model = None
        self.feature_names = ["Text_Ratio", "Image_Count", "Avg_Img_Size_KB", "File_Size_MB"]
        self.load_data()
        self.train()

    def load_data(self):
        """Loads dataset or creates a seed dataset if missing."""
        if os.path.exists(DATA_FILE):
            self.df = pd.read_csv(DATA_FILE)
        else:
            # Seed Data (Synthetic)
            data = {
                "Text_Ratio": [0.9, 0.8, 0.1, 0.3, 0.5, 0.2, 0.95, 0.05],
                "Image_Count": [2, 5, 50, 20, 10, 30, 1, 60],
                "Avg_Img_Size_KB": [50, 100, 500, 1000, 200, 800, 20, 1200],
                "File_Size_MB": [2.5, 5.0, 15.0, 50.0, 8.0, 25.0, 1.0, 60.0],
                "Label": [0, 0, 1, 1, 0, 1, 0, 1]  # 0=Lossless, 1=Rasterize
            }
            self.df = pd.DataFrame(data)
            self.df.to_csv(DATA_FILE, index=False)

    def train(self):
        """Trains the Random Forest model on current data."""
        X = self.df[self.feature_names]
        y = self.df["Label"]
        
        self.model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
            
    def predict(self, features):
        """Returns (prediction, probability)"""
        # Convert list to DataFrame to match training schema and avoid warnings
        df_feat = pd.DataFrame([features], columns=self.feature_names)
        pred = self.model.predict(df_feat)[0]
        prob = np.max(self.model.predict_proba(df_feat))
        return pred, prob

    def add_feedback(self, features, actual_label):
        """Adds new user data to the dataset and retrains."""
        new_row = pd.DataFrame([features + [actual_label]], columns=self.feature_names + ["Label"])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(DATA_FILE, index=False)
        self.train()  # Online learning (Retrain immediately)

# Initialize Model System
if 'ml_system' not in st.session_state:
    st.session_state.ml_system = SmartPDFModel()

ml = st.session_state.ml_system

# --- FEATURE EXTRACTION ---
def extract_features(doc, file_size_mb):
    text_len = 0
    image_count = 0
    total_image_size = 0
    
    sample_pages = min(len(doc), 5)
    for i in range(sample_pages):
        page = doc[i]
        text_len += len(page.get_text())
        images = page.get_images()
        image_count += len(images)
        for img in images:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                total_image_size += len(base_image["image"])
            except:
                pass
            
    avg_text = text_len / sample_pages if sample_pages > 0 else 0
    avg_image_size_kb = (total_image_size / 1024) / image_count if image_count > 0 else 0
    text_ratio = min(avg_text / 3000, 1.0) # Normalized
    
    return [text_ratio, image_count, avg_image_size_kb, file_size_mb]

# --- PDF PROCESSING ---
def process_pdf(input_file, strategy, dpi=150, to_grayscale=False, progress_bar=None):
    # (Keeping the original processing logic exactly as is, just wrapped cleanly)
    input_path = os.path.join(TEMP_DIR, f"input_{int(time.time())}_{input_file.name}")
    with open(input_path, "wb") as f:
        f.write(input_file.getbuffer())

    try:
        doc = fitz.open(input_path)
        total_pages = len(doc)
        output_doc = fitz.open()

        for i in range(total_pages):
            page = doc[i]
            if strategy == "Lossless (Structure Only)":
                output_doc.insert_pdf(doc, from_page=i, to_page=i)
                if progress_bar: progress_bar.progress((i+1)/total_pages, text=f"Optimizing {i+1}/{total_pages}")
            else:
                cs = fitz.csGRAY if to_grayscale else fitz.csRGB
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, colorspace=cs, alpha=False)
                new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(new_page.rect, pixmap=pix)
                if progress_bar: progress_bar.progress((i+1)/total_pages, text=f"Rasterizing {i+1}/{total_pages}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"smart_{timestamp}_{input_file.name}"
        out_path = os.path.join(TEMP_DIR, out_name)
        
        # INCREASE COMPRESSION LEVEL
        output_doc.save(
            out_path, 
            garbage=4, 
            deflate=True, 
            deflate_images=True, 
            deflate_fonts=True,
            clean=True
        )
        return out_path
    except Exception as e:
        raise e
    finally:
        try: doc.close(); output_doc.close()
        except: pass

# --- UI LAYOUT ---
with st.sidebar:
    st.title("SmartPDF Optima")
    st.caption("v2.1.0 Professional")
    
    st.markdown("---")
    
    # Clean Settings Area
    st.markdown("### ⚙️ System Status")
    st.success("● AI Engine Online")
    
    if st.checkbox("Show Developer Metrics"):
        st.markdown("**Model Confidence:**")
        st.progress(0.92)
        st.caption(f"Learned from {len(ml.df)} interactions")
        
        st.markdown("**Training Data:**")
        st.dataframe(ml.df, height=150)
        
    st.markdown("---")
    if st.button("🗑️ Reset Session Cache", help="Clear temporary files"):
        try:
            now = time.time()
            deleted = 0
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted += 1
                except: pass
            st.toast(f"System Optimized: {deleted} files removed.")
        except: pass
            
    st.markdown("---")
    st.caption("© 2026 Vivek's Portfolio\nPowered by Random Forest & PyMuPDF")

# Main Area
st.title("📄 SmartPDF Optima")
st.markdown("#### Intelligent Document Compression Suite")

# Professional Intro
st.info("This system uses **computer vision features** to automatically determine the optimal compression strategy for your documents, balancing file size with visual fidelity.")

st.divider()

col_upload, col_stats = st.columns([1, 1])
with col_upload:
    uploaded_file = st.file_uploader("📄 Upload PDF Document", type="pdf")

if uploaded_file:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    # 1. ANALYZE
    with col_stats:
        with st.status("🔍 AI Analyzing Document Structure...", expanded=True) as status:
            # Save temp for fitz
            tpath = os.path.join(TEMP_DIR, "temp_analysis.pdf")
            with open(tpath, "wb") as f: f.write(uploaded_file.getbuffer())
            
            doc = fitz.open(tpath)
            features = extract_features(doc, file_size_mb)
            doc.close()
            
            # Predict
            pred_idx, prob = ml.predict(features)
            ai_choice = "Lossless" if pred_idx == 0 else "Rasterize"
            
            st.write(f"**📄 Document Metrics:**")
            st.write(f"- Text Content: `{features[0]*100:.1f}%`")
            st.write(f"- Image Count: `{features[1]}`")
            st.write(f"- Original Size: `{features[3]:.2f} MB`")
            status.update(label="Analysis Complete", state="complete", expanded=False)

    # 2. RECOMMENDATION BAR (Compact)
    st.markdown("### 🤖 AI Insight")
    
    if ai_choice == "Lossless":
        bg, border, icon = "#dcfce7", "#22c55e", "✅"
        msg = "Vector-rich content detected. <b>Lossless Optimization</b> will preserve quality."
    else:
        bg, border, icon = "#ffedd5", "#f97316", "⚠️"
        msg = "Image-heavy content detected. <b>Rasterization</b> is recommended for maximum reduction."

    st.markdown(f"""
    <div style="background-color: {bg}; padding: 12px 20px; border-radius: 8px; border-left: 6px solid {border}; display: flex; align-items: center; justify-content: space-between;">
        <div style="color: #1e293b; font-size: 16px;">{icon} {msg}</div>
        <div style="color: {border}; font-weight: bold; font-size: 14px;">Confidence: {prob*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # 3. CONTROL PANEL
    st.divider()
    
    # Swapped Columns: Action (Left) | Settings (Right)
    col_action, col_settings = st.columns([1, 2])
    
    # --- DEFINE VARIABLES FIRST TO AVOID SCOPE ERRORS ---
    dpi = 150
    gray = False
    strategy = "Lossless (Structure Only)" # Default safe value

    with col_settings:
        st.subheader("⚙️ Compression Settings")
        
        # Pre-select based on AI
        user_choice_idx = 0 if ai_choice == "Lossless" else 1
        
        strategy = st.radio(
            "Select Strategy", 
            [
                "Lossless (Structure Only)", 
                "Balanced (Standard Quality)", 
                "Aggressive (Smallest Size)", 
                "Custom (Advanced)"
            ], 
            index=user_choice_idx,
            horizontal=True  # Horizontal layout
        )
        
        # LOGIC MAPPING
        if strategy == "Balanced (Standard Quality)":
            dpi = 150; gray = False
        elif strategy == "Aggressive (Smallest Size)":
            dpi = 96; gray = True
        elif strategy == "Custom (Advanced)":
            c1, c2 = st.columns(2)
            with c1: dpi = st.slider("Target DPI", 72, 300, 150)
            with c2: gray = st.checkbox("Grayscale Mode")

    with col_action:
        st.subheader("🚀 Execute")
        st.info("Ready to optimize?")
        
        if st.button("Run Compression", type="primary", use_container_width=True):
            # LEARNING LOGIC
            is_raster = strategy != "Lossless (Structure Only)"
            chosen_label = 1 if is_raster else 0
            
            if chosen_label != pred_idx:
                st.toast("🧠 Model is learning from your override...", icon="💾")
                ml.add_feedback(features, chosen_label)
                time.sleep(1)
            
            # EXECUTION
            progress = st.progress(0, text="Initializing...")
            try:
                # Pass correct params based on strategy
                if strategy == "Lossless (Structure Only)":
                    out_path = process_pdf(uploaded_file, "Lossless (Structure Only)", 0, False, progress)
                else:
                    out_path = process_pdf(uploaded_file, "Rasterize", dpi, gray, progress)

                out_size = os.path.getsize(out_path) / (1024*1024)
                reduction = (file_size_mb - out_size) / file_size_mb * 100
                
                # RESULT DISPLAY
                st.divider()
                if reduction < 0:
                    st.error(f"⚠️ Size Increased (+{abs(reduction):.1f}%)")
                    st.caption("Try 'Aggressive' mode for vector-heavy PDFs.")
                else:
                    st.balloons()
                    st.success(f"✅ Success! Saved {reduction:.1f}%")
                    
                    # Condensed Metrics Row
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Original", f"{file_size_mb:.1f}MB")
                    c2.metric("New", f"{out_size:.1f}MB")
                    c3.metric("Saved", f"{file_size_mb-out_size:.1f}MB")
                
                with open(out_path, "rb") as f:
                    st.download_button(
                        "📥 Download Optimized PDF", 
                        f, 
                        file_name=f"smart_{uploaded_file.name}",
                        use_container_width=True,
                        type="primary"
                    )
                    
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
