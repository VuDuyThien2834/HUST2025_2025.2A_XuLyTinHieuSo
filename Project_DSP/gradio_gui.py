import gradio as gr
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# --- IMPORT TỪ CÁC MODULE CỦA BẠN ---
# (Đảm bảo bạn đang đứng ở thư mục gốc của project để import được)
from modules.fourier import dft2d, idft2d, get_log_magnitude
from modules.filtering import (apply_2d_filter, ideal_lowpass, butterworth_lowpass, 
                               ideal_highpass, gaussian_highpass)
from modules.restoration import (motion_blur_psf, defocus_blur_psf, 
                                 wiener_deconvolution, find_best_k)
from modules.texture import extract_angular_features, TextureClassifier
from modules.motion_analysis import horn_schunck, flow_to_color

# =========================================================
# HÀM XỬ LÝ LOGIC (WRAPPERS)
# =========================================================

def process_fourier(input_img, filter_type, d0):
    if input_img is None: return None, None
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    M, N = img.shape
    
    # 1. Tính Phổ gốc
    spec_orig = dft2d(img)
    
    # 2. Tạo bộ lọc và áp dụng
    if "Ideal" in filter_type: H = ideal_lowpass(M, N, d0)
    elif "Butterworth" in filter_type: H = butterworth_lowpass(M, N, d0, n=2)
    else: H = gaussian_highpass(M, N, d0)
    
    img_proc = apply_2d_filter(img, H)
    
    # 3. FIX LỖI: Clip giá trị ảnh kết quả về [0, 1]
    img_proc = np.clip(img_proc, 0, 1)
    
    # 4. Tính phổ kết quả (Dùng cho hiển thị)
    spec_proc = dft2d(img_proc)
    log_mag_proc = get_log_magnitude(spec_proc)
    
    # Chuẩn hóa phổ về [0, 1] để Gradio không báo lỗi
    log_mag_proc = (log_mag_proc - log_mag_proc.min()) / (log_mag_proc.max() - log_mag_proc.min() + 1e-8)
    
    return img_proc, log_mag_proc

def process_restoration(input_img, degrade_type, k_value):
    if input_img is None: return None, None
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    M, N = img.shape
    
    if "Motion" in degrade_type: psf = motion_blur_psf(M, N, L=20)
    else: psf = defocus_blur_psf(M, N, r=10)
    
    blurred = np.real(idft2d(dft2d(img) * psf))
    # FIX LỖI: Clip ảnh mờ
    blurred = np.clip(blurred, 0, 1)
    
    restored = wiener_deconvolution(blurred, psf, k_value / 1000.0)
    # FIX LỖI: Clip ảnh phục hồi
    restored = np.clip(restored, 0, 1)
    
    return blurred, restored

def process_motion(frame1, frame2):
    if frame1 is None or frame2 is None: return None, None
    f1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    f2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    u, v = horn_schunck(f1, f2, alpha=1.0)
    color_flow = flow_to_color(u, v)
    
    # Tạo Quiver Plot bằng Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    step = 15
    y, x = np.mgrid[step//2:u.shape[0]:step, step//2:u.shape[1]:step].reshape(2, -1).astype(int)
    ax.imshow(f1, cmap='gray')
    ax.quiver(x, y, -u[y,x], v[y,x], color='red', scale=5)
    ax.axis('off')
    plt.tight_layout()
    
    return color_flow, fig

# =========================================================
# GIAO DIỆN GRADIO (LAYOUT TỔNG LỰC)
# =========================================================

with gr.Blocks(title="DSP Project 09 - Web Dashboard") as demo:
    gr.Markdown("# 🚀 DSP Project 09: 2D Image & Motion Analysis")
    gr.Markdown("Hệ thống xử lý tín hiệu số đa năng - Chuyển đổi từ giao diện Desktop sang Web.")

    with gr.Tabs():
        # --- TAB 1: FOURIER ---
        with gr.TabItem("1. Fourier & Filtering"):
            with gr.Row():
                with gr.Column(scale=1):
                    f_input = gr.Image(label="Input Image")
                    f_type = gr.Dropdown(["Ideal Lowpass", "Butterworth Lowpass", "Gaussian Highpass"], label="Bộ lọc", value="Ideal Lowpass")
                    f_d0 = gr.Slider(5, 150, value=30, label="Ngưỡng cắt (D0)")
                    f_btn = gr.Button("🚀 Apply Filter", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        f_out_img = gr.Image(label="Kết quả (Spatial)")
                        f_out_spec = gr.Image(label="Phổ tần số (Magnitude)")
            gr.HTML("""<div style='background: #e3f2fd; padding: 10px;'><b>Note:</b> Ideal gây Ringing, Gaussian trơn tru nhất.</div>""")

        # --- TAB 2: RESTORATION ---
        with gr.TabItem("2. Restoration"):
            with gr.Row():
                with gr.Column():
                    r_input = gr.Image(label="Original Image")
                    r_degrade = gr.Radio(["Motion Blur", "Defocus Blur"], label="Loại suy thoái", value="Motion Blur")
                    r_k = gr.Slider(1, 100, value=10, label="Tham số K (x10^-3)")
                    r_btn = gr.Button("🔍 Restore Image", variant="primary")
                with gr.Column():
                    r_out_blur = gr.Image(label="Ảnh bị mờ (Degraded)")
                    r_out_restored = gr.Image(label="Ảnh phục hồi (Wiener)")

        # --- TAB 3: MOTION ANALYSIS ---
        with gr.TabItem("3. Motion (Horn-Schunck)"):
            gr.Markdown("Tải lên 2 khung hình liên tiếp để tính toán luồng quang học.")
            with gr.Row():
                m_f1 = gr.Image(label="Frame t")
                m_f2 = gr.Image(label="Frame t+1")
            m_btn = gr.Button("🏃 Analyze Motion", variant="primary")
            with gr.Row():
                m_out_color = gr.Image(label="Color Encoded Flow")
                m_out_quiver = gr.Plot(label="Quiver Plot (Vector Field)")
            
            # Chèn sơ đồ hướng đã thiết kế
            gr.HTML("""
                <div align='center' style='background: #f5f5f5; padding: 10px; border-radius: 8px;'>
                    <table border='0' cellpadding='5'>
                        <tr><td></td><td align='center'><b style='color: #FF00FF;'>Purple</b>: Up ↑</td><td></td></tr>
                        <tr><td align='right'><b style='color: #0000FF;'>Blue</b>: Left ←</td><td align='center'>[Center]</td><td align='left'><b style='color: #FF0000;'>→ Red</b>: Right</td></tr>
                        <tr><td></td><td align='center'><b style='color: #00FF00;'>Green</b>: Down ↓</td><td></td></tr>
                    </table>
                </div>
            """)

    # --- KẾT NỐI EVENT ---
    f_btn.click(process_fourier, inputs=[f_input, f_type, f_d0], outputs=[f_out_img, f_out_spec])
    r_btn.click(process_restoration, inputs=[r_input, r_degrade, r_k], outputs=[r_out_blur, r_out_restored])
    m_btn.click(process_motion, inputs=[m_f1, m_f2], outputs=[m_out_color, m_out_quiver])

if __name__ == "__main__":
    demo.launch(share=True)