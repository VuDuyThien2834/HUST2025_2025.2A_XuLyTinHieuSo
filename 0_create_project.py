'''
Project_DSP/
├── environment.yml             # Quản lý môi trường Conda
├── main_gui.py                 # [Mục 8] File thực thi chính chạy GUI (Student B)
│
├── modules/                    # Thư mục chứa các lõi thuật toán
│   ├── __init__.py
│   ├── fourier.py              # [Mục 1] 2D DFT tự triển khai, kiểm tra sai số 1e-10
│   ├── filtering.py            # [Mục 1, 3] Các bộ lọc (Butterworth, Gaussian, Notch...)
│   ├── restoration.py          # [Mục 4] PSF Motion/Defocus blur & Wiener Deconvolution
│   ├── texture.py              # [Mục 5] Power Spectrum, Angular Bins, KNN classification
│   ├── motion_analysis.py      # [Mục 2, 6] Horn-Schunck Optical Flow core
│   └── video_utils.py          # [Mục 7] Pipeline xử lý 30 frames, tính Motion Magnitude
│
├── data/                       # Chứa dữ liệu đầu vào
│   ├── images/                 # Ảnh test cho Filtering/Wiener
│   ├── textures/               # [Mục 5] 4 lớp kết cấu cho KNN (mỗi lớp 5-10 ảnh)
│   └── video/                  # [Mục 7] Đoạn video 30 frames (mp4 hoặc chuỗi ảnh)
│
├── outputs/                    # [Mục 3, 4, 5, 6, 7] Lưu trữ kết quả xuất từ code/GUI
│   ├── figures/                # Các biểu đồ Gallery, Quiver plots, Magnitude curve
│   └── logs/                   # Lưu kết quả tính SSIM, Confusion Matrix
│
├── notebooks/                  # Dùng để dev nhanh và kiểm chứng (Verification)
│   ├── test_fft_precision.ipynb # Kiểm tra my_fft2 vs numpy.fft2
│   └── test_optical_flow.ipynb  # Test thử thuật toán Horn-Schunck
│
└── report/                     # [Mục 9] Tài liệu báo cáo
    ├── figures/                # Ảnh xuất riêng cho báo cáo
    └── report_joint.pdf        # Báo cáo chung > 15 trang
'''

import os

structure = [
    "Project_DSP/modules/__init__.py",
    "Project_DSP/modules/fourier.py",
    "Project_DSP/modules/filtering.py",
    "Project_DSP/modules/restoration.py",
    "Project_DSP/modules/texture.py",
    "Project_DSP/modules/motion_analysis.py",
    "Project_DSP/modules/video_utils.py",
    "Project_DSP/data/images/",
    "Project_DSP/data/textures/",
    "Project_DSP/data/video/",
    "Project_DSP/outputs/figures/",
    "Project_DSP/outputs/logs/",
    "Project_DSP/notebooks/test_fft_precision.ipynb",
    "Project_DSP/notebooks/test_optical_flow.ipynb",
    "Project_DSP/report/figures/",
    "Project_DSP/environment.yml",
    "Project_DSP/main_gui.py",
]

for path in structure:
    full_path = os.path.join(os.getcwd(), path)
    if path.endswith('/'):
        os.makedirs(full_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            pass # Tạo file trống
print("Cấu trúc project đã được khởi tạo thành công!")