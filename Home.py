import streamlit as st

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DocuMind AI")
st.subheader("Intelligent Document Processing Suite")

st.markdown("""
### Welcome to the Future of Document Management 🚀

**DocuMind AI** leverages advanced Machine Learning and Computer Vision to optimize your workflow.

---

### 📂 Select a Module from the Sidebar:

#### 1. 📉 [Smart Compressor](/Smart_Compressor)
> **"The Brain"**
> Uses a **Random Forest Classifier** to analyze document structure (text vs. images) and automatically selects the best compression strategy.
> *   **Best for:** Reducing file size while preserving quality.
> *   **Tech:** Scikit-Learn, PyMuPDF.

#### 2. ✨ [Image Enhancer](/Image_Enhancer)
> **"The Eye"**
> Uses **Computer Vision (OpenCV)** to detect brightness, contrast, and blur levels, then automatically restores scanned documents to perfection.
> *   **Best for:** Cleaning up dark, blurry, or old scans.
> *   **Tech:** OpenCV, PIL.

---
*Open Source Project for AI/ML Portfolio.*
""")
