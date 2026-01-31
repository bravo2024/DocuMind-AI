import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import io
import os
import time
import numpy as np
import cv2  # OpenCV for advanced analysis

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="PDF Clarify",
    page_icon="✨",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    h1 { color: #d946ef; } /* Pink title */
</style>
""", unsafe_allow_html=True)

TEMP_DIR = "enhanced_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- AI ANALYSIS FUNCTION ---
def analyze_image(pil_image):
    """
    Analyzes the image to suggest optimal settings.
    Returns: dict of suggested parameters
    """
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. Calculate Mean Brightness (0-255)
    mean_brightness = np.mean(gray)
    
    # 2. Calculate Contrast (Standard Deviation)
    contrast_score = np.std(gray)
    
    # 3. Calculate Blur (Laplacian Variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # --- SMART LOGIC ---
    suggestions = {
        "brightness": 1.0,
        "contrast": 1.0,
        "sharpness": 1.0,
        "threshold": False,
        "reason": []
    }
    
    # Brightness Logic
    if mean_brightness < 100: # Too Dark
        suggestions["brightness"] = 1.4
        suggestions["reason"].append("Image is dark (Boost Brightness)")
    elif mean_brightness > 200: # Too Bright
        suggestions["brightness"] = 0.9
        suggestions["reason"].append("Image is washed out (Reduce Brightness)")
        
    # Contrast Logic
    if contrast_score < 40: # Low Contrast
        suggestions["contrast"] = 1.5
        suggestions["reason"].append("Low contrast detected (Boost Contrast)")
        
    # Sharpness Logic
    if blur_score < 100: # Blurry
        suggestions["sharpness"] = 2.0
        suggestions["reason"].append("Blurriness detected (Apply Sharpening)")
    elif blur_score < 300: # Slightly Soft
        suggestions["sharpness"] = 1.5
        
    # Threshold Logic (for text docs)
    # If high contrast and bright background, assume text doc -> Threshold helps
    if contrast_score > 50 and mean_brightness > 150:
        suggestions["threshold"] = False # Optional: Keep False by default to be safe
    
    return suggestions

def enhance_page(page, brightness, contrast, sharpness, grayscale, threshold_mode):
    """
    Render PDF page to image, apply enhancements, return PIL image.
    """
    # 1. Render high-quality image from PDF page (2x zoom for clarity)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 2. Apply Filters
    if grayscale or threshold_mode:
        img = img.convert("L")  # Convert to Grayscale

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)

    # 3. Apply Thresholding (Scanner Effect) if requested
    if threshold_mode:
        # Simple binary threshold
        img = img.point(lambda p: 255 if p > 128 else 0)

    return img

def process_pdf(uploaded_file, brightness, contrast, sharpness, grayscale, threshold):
    input_path = os.path.join(TEMP_DIR, f"raw_{int(time.time())}.pdf")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        doc = fitz.open(input_path)
        output_images = []
        
        # Process each page
        progress_bar = st.progress(0)
        for i, page in enumerate(doc):
            enhanced_img = enhance_page(page, brightness, contrast, sharpness, grayscale, threshold)
            output_images.append(enhanced_img)
            progress_bar.progress((i + 1) / len(doc))

        # Save back to PDF
        output_filename = f"clarified_{uploaded_file.name}"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Save first image and append the rest
        if output_images:
            output_images[0].save(
                output_path, "PDF",
                resolution=150.0,
                save_all=True,
                append_images=output_images[1:]
            )
            
        return output_path

    except Exception as e:
        raise e
    finally:
        try: doc.close() 
        except: pass
        if os.path.exists(input_path): os.remove(input_path)

# --- SESSION STATE INITIALIZATION ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'brightness': 1.2,
        'contrast': 1.2,
        'sharpness': 1.5,
        'threshold': False
    }

# --- UI LAYOUT ---
st.title("✨ PDF Clarify")
st.markdown("### Make Scanned PDFs Bright, Sharp & Clear")

col_controls, col_preview = st.columns([1, 2])

# --- CONTROL PANEL ---
with col_controls:
    st.subheader("🎨 Enhancement Controls")
    
    uploaded_file = st.file_uploader("Upload Scan", type="pdf")
    
    # AI Auto-Adjust Button
    if uploaded_file:
        if st.button("✨ AI Auto-Adjust", type="secondary", use_container_width=True):
            with st.spinner("Analyzing image content..."):
                with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
                    # Analyze first page
                    sample_pix = doc[0].get_pixmap(matrix=fitz.Matrix(1, 1))
                    sample_img = Image.frombytes("RGB", [sample_pix.width, sample_pix.height], sample_pix.samples)
                    suggestions = analyze_image(sample_img)
                    
                    # Update State
                    st.session_state.settings['brightness'] = suggestions['brightness']
                    st.session_state.settings['contrast'] = suggestions['contrast']
                    st.session_state.settings['sharpness'] = suggestions['sharpness']
                    st.session_state.settings['threshold'] = suggestions['threshold']
                    
                    if suggestions['reason']:
                        st.success("Adjusted: " + ", ".join(suggestions['reason']))
                    else:
                        st.info("Image looks good! Keeping standard settings.")
                        
    st.divider()
    
    # Sliders (Controlled by Session State)
    brightness = st.slider("Brightness ☀️", 0.5, 2.0, st.session_state.settings['brightness'], 0.1, key="brightness_slider")
    contrast = st.slider("Contrast 🌓", 0.5, 2.0, st.session_state.settings['contrast'], 0.1, key="contrast_slider")
    sharpness = st.slider("Sharpness 🔪", 0.0, 3.0, st.session_state.settings['sharpness'], 0.1, key="sharpness_slider")
    
    st.markdown("---")
    st.markdown("**Modes**")
    grayscale = st.checkbox("Grayscale (B/W)", value=True)
    threshold = st.checkbox("Scanner Mode (Binary)", value=st.session_state.settings['threshold'], key="threshold_box", help="Forces pure black & white.")

    st.write("")
    if uploaded_file and st.button("🚀 Clarify PDF", type="primary", use_container_width=True):
        with st.spinner("Enhancing images..."):
            try:
                out_path = process_pdf(uploaded_file, brightness, contrast, sharpness, grayscale, threshold)
                st.success("Enhancement Complete!")
                
                with open(out_path, "rb") as f:
                    st.download_button(
                        "📥 Download Enhanced PDF",
                        f,
                        file_name=f"clarified_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# --- LIVE PREVIEW ---
with col_preview:
    st.subheader("👁️ Live Preview")
    
    if uploaded_file:
        try:
            # Load PDF
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            total_pages = len(doc)
            
            # Page Selector
            page_num = st.slider("Select Page to Preview", 1, total_pages, 1) - 1
            
            # Render & Enhance
            page = doc[page_num]
            preview_img = enhance_page(page, brightness, contrast, sharpness, grayscale, threshold)
            
            st.image(preview_img, caption=f"Preview of Page {page_num + 1}", use_column_width=True)
            doc.close()
            
        except Exception as e:
            st.warning(f"Could not load preview: {e}")
    else:
        st.info("Upload a PDF to see the magic happen!")
