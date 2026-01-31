# DocuMind AI 🧠
### Intelligent Document Processing & Optimization Suite

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV-green.svg)](https://opencv.org/)

**DocuMind AI** is a comprehensive solution for automated document management, leveraging Machine Learning and Computer Vision to intelligently compress and enhance PDF documents.

---

## 🚀 Key Features

### 1. 📉 Smart Compression Engine
*   **Problem:** Standard compression ruins text quality or fails to reduce file size on vector PDFs.
*   **Solution:** Uses a **Random Forest Classifier** to analyze document structure (Text/Image Ratio, Vector Count).
*   **Outcome:** Automatically selects **Lossless Optimization** for text docs and **Rasterization** for scans, achieving up to **70% size reduction** without quality loss.
*   **Active Learning:** The model retrains itself in real-time based on user feedback.

### 2. ✨ CV-Powered Enhancement
*   **Problem:** Scanned documents are often dark, blurry, or low-contrast.
*   **Solution:** Uses **OpenCV** to calculate pixel intensity histograms and Laplacian variance (blur detection).
*   **Outcome:** Auto-adjusts Brightness, Contrast, and Sharpness to restore document clarity.

---

## 🏗 System Architecture

```mermaid
graph TD
    A[User Upload] --> B{Router};
    B -->|Compression| C[Feature Extraction];
    C --> D[Random Forest Model];
    D -->|Text Heavy| E[Lossless Engine];
    D -->|Image Heavy| F[Rasterize Engine];
    B -->|Enhancement| G[OpenCV Analysis];
    G --> H[Auto-Adjustment Logic];
    H --> I[Image Processing Pipeline];
    E --> J[Optimized PDF];
    F --> J;
    I --> J;
```

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **PDF Core:** PyMuPDF (Fitz)
*   **Machine Learning:** Scikit-Learn (Random Forest)
*   **Computer Vision:** OpenCV, PIL
*   **Visualization:** Plotly

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bravo2024/DocuMind-AI.git
    cd DocuMind-AI
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## 👨‍💻 Developer Notes

This project demonstrates the application of **applied AI** to solve real-world administrative bottlenecks. The modular design allows for easy extension into OCR and extraction tasks.

---
*Developed by Vivek.*
