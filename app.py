import streamlit as st

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DocuMind AI")
st.markdown("### Intelligent Document Processing Suite")

st.markdown("""
Welcome to **DocuMind AI**, a professional-grade toolkit for automated document optimization.
Select a module from the sidebar to begin.

#### 🚀 Modules
- **📉 SmartPDF Compressor:** Uses ML (Random Forest) to classify documents and apply optimal compression (Lossless vs Rasterize).
- **✨ PDF Clarify:** Uses Computer Vision (OpenCV) to analyze brightness/contrast and auto-enhance scanned documents.

#### 🏗 Architecture
Built with `Streamlit`, `PyMuPDF`, `Scikit-Learn`, and `OpenCV`.
""")

st.info("👈 Select a tool from the sidebar!")
