import streamlit as st

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 DocuMind AI")
st.caption("Intelligent Document Processing Suite")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.image("https://img.icons8.com/color/96/compress.png", width=64)
    st.subheader("Smart Compressor")
    st.markdown("ML-powered reduction that preserves quality.")
    st.page_link("pages/1_📉_Smart_Compressor.py", label="Launch Compressor", icon="📉", use_container_width=True)

with col2:
    st.image("https://img.icons8.com/color/96/image.png", width=64)
    st.subheader("Image Enhancer")
    st.markdown("CV-powered restoration for scanned docs.")
    st.page_link("pages/2_✨_Image_Enhancer.py", label="Launch Enhancer", icon="✨", use_container_width=True)

st.markdown("---")
st.info("🔒 **Privacy First:** Files are processed locally in memory and auto-deleted after 1 hour.")
