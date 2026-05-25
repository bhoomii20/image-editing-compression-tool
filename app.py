import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (compress_image, calculate_psnr, get_size_kb, 
                  create_difference_image, plot_histogram, 
                  resize_image, adjust_brightness_contrast)
import io
from zipfile import ZipFile
from streamlit_cropper import st_cropper
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Image Compression Visualizer", layout="wide")
st.title("🖼️ Advanced Image Processing & Compression Tool")
st.markdown("**Edit • Crop • Resize • Compress • Analyze**")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("🛠️ Editing Tools")
    enable_edit = st.checkbox("Enable Editing Tools", value=False)

    edit_option = None
    if enable_edit:
        edit_option = st.selectbox("Choose Editing Tool", 
            ["Resize", "Crop", "Brightness/Contrast", "Grayscale"])

    st.header("📦 Compression Settings")
    apply_compression = st.checkbox("Apply JPEG Compression", value=True)
    
    if apply_compression:
        mode = st.radio("Compression Mode", ["By Quality", "By Target Size"])
        if mode == "By Quality":
            quality = st.slider("JPEG Quality (%)", 10, 100, 75, step=5)
        else:
            target_size = st.number_input("Target File Size (KB)", 50, 5000, 500, step=50)

# ================== UPLOAD ==================
uploaded_files = st.file_uploader("Upload One or Multiple Images", 
                                 type=["jpg", "jpeg", "png"], 
                                 accept_multiple_files=True)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} image(s) uploaded")

    if len(uploaded_files) > 1:
        # ================== BATCH PROCESSING ==================
        st.subheader("Batch Processing")
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as zip_file:
            for uploaded_file in uploaded_files:
                with st.expander(f"📸 {uploaded_file.name}", expanded=False):
                    original_pil = Image.open(uploaded_file).convert("RGB")
                    original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                    orig_size_kb = get_size_kb(uploaded_file.getvalue())

                    working_pil = original_pil.copy()
                    working_cv = original_cv.copy()

                    # Apply Editing (Simple for batch)
                    if enable_edit and edit_option:
                        if edit_option == "Resize":
                            working_cv = resize_image(working_cv, 800, 600)
                            working_pil = Image.fromarray(cv2.cvtColor(working_cv, cv2.COLOR_BGR2RGB))

                    # Compression
                    if apply_compression:
                        if mode == "By Quality":
                            compressed_pil, compressed_bytes = compress_image(working_pil, quality)
                            used_q = quality
                        else:
                            compressed_pil, compressed_bytes = compress_image(working_pil, 70)
                            used_q = 70
                    else:
                        compressed_pil = working_pil
                        buf = io.BytesIO()
                        working_pil.save(buf, format="JPEG", quality=90)
                        compressed_bytes = buf.getvalue()
                        used_q = "N/A"

                    col1, col2 = st.columns(2)
                    col1.image(original_pil, caption=f"Original ({orig_size_kb:.1f} KB)", width=300)
                    col2.image(compressed_pil, caption=f"Processed (Q={used_q})", width=300)

                    zip_file.writestr(f"processed_{uploaded_file.name}", compressed_bytes)

        st.download_button(
            label="📥 Download All Processed Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="All_Processed_Images.zip",
            mime="application/zip",
            use_container_width=True
        )

    else:
        # ================== SINGLE IMAGE MODE ==================
        uploaded_file = uploaded_files[0]
        original_pil = Image.open(uploaded_file).convert("RGB")
        original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        orig_size_kb = get_size_kb(uploaded_file.getvalue())

        working_pil = original_pil.copy()
        working_cv = original_cv.copy()

        # ================== EDITING SECTION ==================
        if enable_edit and edit_option:
            with st.expander(f"✂️ {edit_option} Tool", expanded=True):
                if edit_option == "Resize":
                    col1, col2 = st.columns(2)
                    w = col1.number_input("Width", 10, 4000, original_pil.width)
                    h = col2.number_input("Height", 10, 4000, original_pil.height)
                  
                    working_cv = resize_image(original_cv, w, h)
                    working_pil = Image.fromarray(cv2.cvtColor(working_cv, cv2.COLOR_BGR2RGB))
                    st.success(f"Resized to {w}×{h}")
                    buf = io.BytesIO()
                    working_pil.save(buf, format="JPEG")
                    st.image(working_pil, caption=f"Resized Preview ({w}×{h})", width=400)
                    st.download_button(
                        label="📥 Download Resized Image",
                        data=buf.getvalue(),
                        file_name=f"resized_{uploaded_file.name}",
                        mime="image/jpeg",
                        use_container_width=True
                    )

                elif edit_option == "Crop":
                    st.info("Draw a rectangle to crop the image below and click 'Apply Crop'.")
                    cropped_img = st_cropper(working_pil, box_color='#FF0000', aspect_ratio=None)
                    st.image(cropped_img, caption="Crop Preview", width=450)
                    if st.button("Apply Crop"):
                        working_pil = cropped_img
                        working_cv = cv2.cvtColor(np.array(working_pil), cv2.COLOR_RGB2BGR)
                        st.success("✅ Crop applied successfully!")
                        buf = io.BytesIO()
                        working_pil.save(buf, format="JPEG")
                        st.download_button(
                            label="📥 Download Cropped Image",
                            data=buf.getvalue(),
                            file_name=f"cropped_{uploaded_file.name}",
                            mime="image/jpeg",
                            use_container_width=True
                        )


                elif edit_option == "Brightness/Contrast":
                    brightness = st.slider("Brightness", -50, 50, 0)
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                    working_cv = adjust_brightness_contrast(original_cv, brightness, contrast)
                    working_pil = Image.fromarray(cv2.cvtColor(working_cv, cv2.COLOR_BGR2RGB))
                    st.success("Adjusted!")
                    buf = io.BytesIO()
                    working_pil.save(buf, format="JPEG")
                    st.image(working_pil, caption="Brightness/Contrast Preview", width=400)
                    st.download_button(
                        label="📥 Download Adjusted Image",
                        data=buf.getvalue(),
                        file_name=f"adjusted_{uploaded_file.name}",
                        mime="image/jpeg",
                        use_container_width=True
                    )

                elif edit_option == "Grayscale":
                    intensity = st.slider("Grayscale Intensity", 0.5, 2.0, 1.0, 0.1)
                    
                    gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
                    gray = np.clip(gray * intensity, 0, 255).astype(np.uint8)
                    working_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    working_pil = Image.fromarray(cv2.cvtColor(working_cv, cv2.COLOR_BGR2RGB))
                    st.success("Grayscale applied!")
                    buf = io.BytesIO()
                    working_pil.save(buf, format="JPEG")
                    st.image(working_pil, caption="Grayscale Preview", width=400)
                    st.download_button(
                        label="📥 Download Grayscale Image",
                        data=buf.getvalue(),
                        file_name=f"grayscale_{uploaded_file.name}",
                        mime="image/jpeg",
                        use_container_width=True
                    )

        # ================== COMPRESSION ==================
        if apply_compression:
            if mode == "By Quality":
                compressed_pil, compressed_bytes = compress_image(working_pil, quality)
                used_quality = quality
            else:
                st.info("🔍 Finding best quality...")
                best_pil, best_bytes, best_q = None, None, 50
                for q in range(95, 9, -5):
                    temp_pil, temp_bytes = compress_image(working_pil, q)
                    if get_size_kb(temp_bytes) <= target_size:
                        best_pil, best_bytes, best_q = temp_pil, temp_bytes, q
                        break
                if best_pil is None:
                    best_pil, best_bytes = compress_image(working_pil, 10)
                    best_q = 10
                compressed_pil = best_pil
                compressed_bytes = best_bytes
                used_quality = best_q

            compressed_cv = cv2.cvtColor(np.array(compressed_pil), cv2.COLOR_RGB2BGR)

             # Calculate compressed size
            comp_size_kb = get_size_kb(compressed_bytes)

            col1, col2 = st.columns(2)
            with col1:
                st.image(working_pil, caption="**Edited / Original Image**", width=450)
            with col2:
                st.image(compressed_pil, 
                    caption=f"**Compressed (Q={used_quality}%) ({comp_size_kb:.1f} KB)**", 
                    width=450)
            
            comp_size = get_size_kb(compressed_bytes)
            reduction = ((orig_size_kb - comp_size) / orig_size_kb) * 100
            psnr_value = calculate_psnr(working_cv, compressed_cv)

            st.subheader("📊 Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Size Reduction", f"{reduction:.1f}%", f"{orig_size_kb:.1f} → {comp_size_kb:.1f} KB")
            m2.metric("PSNR", f"{psnr_value:.2f} dB")
            m3.metric("Quality", f"{used_quality}%")

            st.download_button(
                label="📥 Download Compressed Image",
                data=compressed_bytes,
                file_name=f"compressed_{uploaded_file.name}",
                mime="image/jpeg",
                use_container_width=True
            )

            tab1, tab2 = st.tabs(["🔍 Difference Map", "📈 Histograms"])
            with tab1:
                diff = create_difference_image(working_cv, compressed_cv)
                st.image(diff, caption="Red/Yellow = Quality Loss", channels="BGR", width=600)
            with tab2:
                c1, c2 = st.columns(2)
                c1.pyplot(plot_histogram(working_cv, "Input"))
                c2.pyplot(plot_histogram(compressed_cv, "Compressed"))
        
        else:
            # No Compression Case
            st.image(working_pil, caption="Edited Image (No Compression)", width=600)
