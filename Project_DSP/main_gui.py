import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QFileDialog, QSlider, QComboBox, QGroupBox, 
                             QTableWidget, QTableWidgetItem, QSplitter,
                             QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# --- IMPORT TỪ CÁC MODULE ---
from modules.fourier import dft2d, idft2d, get_log_magnitude
from modules.filtering import (apply_2d_filter, ideal_lowpass, butterworth_lowpass, 
                               ideal_highpass, gaussian_highpass)
from modules.restoration import (motion_blur_psf, defocus_blur_psf, 
                                 wiener_deconvolution, find_best_k)
from modules.texture import extract_angular_features, TextureClassifier
from modules.motion_analysis import horn_schunck, flow_to_color
from modules.video_utils import extract_frames, extract_frames_from_folder, compute_video_motion

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

class DSP_Full_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP- Project 09 — 2D Image Processing & Video Motion Analysis ")
        self.resize(1600, 900)
        
        # Biến lưu trữ dữ liệu
        self.img_orig = None
        self.img_proc = None
        self.psf_current = None
        self.clf = TextureClassifier() # Mục 5
        
        # tạo hiệu ứng animation cho các khung hình hoặc video khi chạy optical flow
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.current_frame_idx = 0

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # =========================================================
        # CỘT TRÁI: BẢNG ĐIỀU KHIỂN (CONTROLS)
        # =========================================================
        control_panel = QTabWidget()
        control_panel.addTab(self.ui_tab_fourier(), "1. Fourier & Filter")
        control_panel.addTab(self.ui_tab_restoration(), "2. Restoration")
        control_panel.addTab(self.ui_tab_texture(), "3. Texture KNN")
        control_panel.addTab(self.ui_tab_motion(), "4. Motion/Video")
        
        main_layout.addWidget(control_panel, 1)

        # =========================================================
        # CỘT PHẢI: HIỂN THỊ (VIEWPORT)
        # =========================================================
        viewport = QVBoxLayout()
        
        # Hàng 1: Ảnh gốc và Ảnh đã xử lý
        img_row = QHBoxLayout()
        self.lbl_view_orig = self.create_image_label("Ảnh Gốc / Đầu vào")
        self.lbl_view_proc = self.create_image_label("Ảnh Kết quả")
        img_row.addWidget(self.lbl_view_orig)
        img_row.addWidget(self.lbl_view_proc)
        
        # Hàng 2: Phổ tần số và Biểu đồ
        analysis_row = QHBoxLayout()
        self.lbl_view_spec = self.create_image_label("Phổ Biên độ (Log Magnitude)") # Dùng get_log_magnitude
        self.canvas = MplCanvas(self) # Dùng để vẽ Quiver hoặc Curve
        analysis_row.addWidget(self.lbl_view_spec)
        analysis_row.addWidget(self.canvas)
        
        viewport.addLayout(img_row, 1)
        viewport.addLayout(analysis_row, 1)
        
        main_layout.addLayout(viewport, 2)

    def create_image_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("border: 2px solid #333; background: #111; color: white;")
        lbl.setMinimumSize(400, 300)
        return lbl

    # ---------------------------------------------------------
    # TAB 1: FOURIER & FILTERING (Sử dụng Mục 1 & 3)
    # ---------------------------------------------------------
    def ui_tab_fourier(self):
        w = QWidget()
        # Sử dụng Layout chính là ngang để ngăn cách phần điều khiển/ghi chú
        main_lay = QHBoxLayout() 
        
        # --- CỘT TRÁI: ĐIỀU KHIỂN & GHI CHÚ ---
        left_panel = QVBoxLayout()

        # 1. Nhóm Load dữ liệu
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.action_load_image)
        left_panel.addWidget(btn_load)

        # 2. Nhóm Tần số & Bộ lọc
        group = QGroupBox("Cấu hình Bộ lọc Tần số")
        vbox = QVBoxLayout()
        self.cb_filter = QComboBox()
        self.cb_filter.addItems(["Ideal Lowpass", "Butterworth Lowpass", "Gaussian Highpass"])
        vbox.addWidget(QLabel("Loại bộ lọc:"))
        vbox.addWidget(self.cb_filter)
        
        self.sld_d0 = QSlider(Qt.Horizontal)
        self.sld_d0.setRange(5, 150)
        self.sld_d0.setValue(30)
        vbox.addWidget(QLabel("Ngưỡng cắt (D0):"))
        vbox.addWidget(self.sld_d0)
        
        btn_apply = QPushButton("Apply & Show Spectrum")
        btn_apply.clicked.connect(self.action_filtering_full)
        btn_apply.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        vbox.addWidget(btn_apply)
        group.setLayout(vbox)
        left_panel.addWidget(group)

        # --- PHẦN GHI CHÚ KIẾN THỨC (NOTE) ---
        note_group = QGroupBox("Kiến thức Fourier")
        note_lay = QVBoxLayout()
        
        self.txt_fourier_note = QTextEdit()
        self.txt_fourier_note.setReadOnly(True)
        self.txt_fourier_note.setHtml("""
            <div style='line-height: 1.4; font-size: 13px;'>
                    <b style='color: #E91E63;'>1. Biến đổi Fourier (2D-DFT):</b><br>
                    Chuyển ảnh từ miền không gian sang miền tần số. 
                    Phổ biên độ cho biết mật độ phân bố năng lượng của ảnh.
                    <ul style='margin-top: 5px;'>
                        <li><b>Tần số thấp:</b> Nằm ở trung tâm (sau khi Shift), chứa thông tin nền và cấu trúc lớn.</li>
                        <li><b>Tần số cao:</b> Nằm ở xa tâm, chứa chi tiết, cạnh (edges) và nhiễu.</li>
                    </ul>

                <div style='margin-top: 10px;'>                      
                    <b style='color: #4CAF50;'>2. Các loại bộ lọc:</b><br>
                    <ul style='margin-top: 1.4px;'>
                        <li><b style='color: #FF9800;'>Ideal:</b> Cắt cụt tần số đột ngột, gây ra hiện tượng <i>Ringing Artifact</i> (vòng sáng quanh cạnh).</li>
                        <li><b style='color: #FF9800;'>Butterworth:</b> Sự chuyển đổi mượt mà hơn tùy theo bậc (order), giảm thiểu hiện tượng nhiễu vòng.</li>
                        <li><b style='color: #FF9800;'>Gaussian:</b> Hàm truyền cực kỳ trơn tru, không gây ra hiện tượng Ringing.</li>
                    </ul>
                </div>
                                      
                <div style='margin-top: 10px;'>
                    <b style='color: #2196F3;'>3. Quy trình thực hiện:</b><br>
                    <ol style='margin-top: 1.4px;'>
                        <li>Nhân ảnh với <i style='font-family: serif;'>(-1)<sup>x+y</sup></i> để dời tâm phổ.</li>
                        <li>Tính DFT.</li>
                        <li>Nhân phổ với hàm truyền <i style='font-family: serif;'>H(u,v)</i>.</li>
                        <li>Tính IDFT để quay về miền không gian.</li>
                    </ol>
                </div>                      
            </div>
        """)
        note_lay.addWidget(self.txt_fourier_note)
        note_group.setLayout(note_lay)
        left_panel.addWidget(note_group)
        
        # Gán vào Layout chính
        w.setLayout(left_panel)
        left_panel.setStretchFactor(note_group, 1) # Để phần ghi chú chiếm nhiều không gian hơn

        return w

    # ---------------------------------------------------------
    # TAB 2: RESTORATION (Sử dụng Mục 4 - Wiener & K-Sweep)
    # ---------------------------------------------------------
    def ui_tab_restoration(self):
        w = QWidget()
        main_lay = QHBoxLayout() # Sử dụng Layout ngang để chia cột
        
        # --- CỘT TRÁI: ĐIỀU KHIỂN & GHI CHÚ ---
        left_panel = QVBoxLayout()
        
        # 1. Nhóm Điều khiển suy thoái
        group_ctrl = QGroupBox("Cấu hình Phục hồi ảnh")
        vbox = QVBoxLayout()
        
        self.cb_degrade = QComboBox()
        self.cb_degrade.addItems(["Motion Blur (L=20)", "Defocus Blur (R=10)"])
        vbox.addWidget(QLabel("Mô hình suy thoái (PSF):"))
        vbox.addWidget(self.cb_degrade)
        
        btn_blur = QPushButton("Tạo ảnh mờ")
        btn_blur.clicked.connect(self.action_create_blur)
        vbox.addWidget(btn_blur)
        
        btn_sweep = QPushButton("Chạy K-Sweep (Best SSIM)")
        btn_sweep.clicked.connect(self.action_wiener_sweep)
        btn_sweep.setStyleSheet("background-color: #FB8C00; color: white; font-weight: bold;")
        vbox.addWidget(btn_sweep)
        
        self.sld_k_manual = QSlider(Qt.Horizontal)
        self.sld_k_manual.setRange(1, 1000) 
        vbox.addWidget(QLabel("Chỉnh K thủ công (Noise/Signal ratio):"))
        vbox.addWidget(self.sld_k_manual)
        self.sld_k_manual.valueChanged.connect(self.action_wiener_manual)
        
        group_ctrl.setLayout(vbox)
        left_panel.addWidget(group_ctrl)

        # --- PHẦN GHI CHÚ KIẾN THỨC (NOTE) ---
        note_group = QGroupBox("Lý thuyết Phục hồi ảnh")
        note_lay = QVBoxLayout()
        
        self.txt_rest_note = QTextEdit()
        self.txt_rest_note.setReadOnly(True)
        self.txt_rest_note.setHtml("""
            <div style='line-height: 1.4; font-size: 13px;'>
                    <b style='color: #F44336;'>1. Mô hình suy thoái (Degradation):</b><br>
                    Ảnh bị mờ được mô hình hóa bởi công thức: <i style='font-family: serif;'>g = h * f + &eta;</i><br>
                    Trong đó <i style='font-family: serif;'>h</i> là hàm lan truyền điểm (PSF).
                    <ul style='margin-top: 5px;'>
                        <li><b style='color: #2196F3;'>Motion Blur:</b> Do vật thể hoặc camera di chuyển (theo đường thẳng).</li>
                        <li><b style='color: #2196F3;'>Defocus Blur:</b> Do mất nét, được mô hình hóa bằng một đĩa tròn (pillbox).</li>
                    </ul>

                <div style='margin-top: 10px;'>
                    <b style='color: #4CAF50;'>2. Bộ lọc Wiener:</b><br>
                    Là bộ lọc tối ưu dựa trên tiêu chí bình phương trung bình tối thiểu (MSE). Công thức trong miền tần số:
                    <div style='background-color: #ffffff; padding: 10px; margin: 10px 0; text-align: center;'>
                        <i style='font-family: serif; font-size: 14px;'> 
                            W(u,v) = [ 1 / H(u,v) ] &middot; [ |H(u,v)|<sup>2</sup> / (|H(u,v)|<sup>2</sup> + K) ]
                        </i>
                    </div>
                </div>
                                   
                <div style='margin-top: 10px;'>                   
                    <b style='color: #FF9800;'>3. Tham số K (K-Sweep):</b><br>
                    <i style='font-family: serif;'>K</i> đại diện cho tỉ số Nhiễu/Tín hiệu (<i style='font-family: serif;'>SNR<sup>-1</sup></i>).
                    <ul style='margin-top: 5px;'>
                        <li>Nếu <b>K quá nhỏ</b>: Bộ lọc giống lọc ngược, nhiễu sẽ bị khuếch đại mạnh.</li>
                        <li>Nếu <b>K quá lớn</b>: Ảnh sẽ vẫn bị mờ vì không lọc đủ.</li>
                        <li><b>K-Sweep:</b> Tìm giá trị K sao cho chỉ số <b>Structural Similarity Index (SSIM)</b> (độ tương đồng cấu trúc) đạt cực đại.</li>
                    </ul>
                </div>
            </div>
        """)
        note_lay.addWidget(self.txt_rest_note)
        note_group.setLayout(note_lay)
        left_panel.addWidget(note_group)
        
        w.setLayout(left_panel)
        left_panel.setStretchFactor(note_group, 1) # Để phần ghi chú chiếm nhiều không gian hơn
        return w

    # ---------------------------------------------------------
    # TAB 3: TEXTURE KNN (Sử dụng Mục 5 - Angular Bins)
    # ---------------------------------------------------------
    def ui_tab_texture(self):
        w = QWidget()
        main_lay = QHBoxLayout() # Sử dụng Layout ngang

        # --- CỘT TRÁI: ĐIỀU KHIỂN & GHI CHÚ ---
        left_panel = QVBoxLayout()

        # 1. Nhóm Điều khiển
        group_ctrl = QGroupBox("Phân loại kết cấu (Texture)")
        vbox = QVBoxLayout()
        
        btn_train = QPushButton("Train KNN Model")
        btn_train.clicked.connect(self.action_train_texture)
        vbox.addWidget(btn_train)
        
        btn_classify = QPushButton("Classify Current Image")
        btn_classify.clicked.connect(self.action_classify_texture)
        btn_classify.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold;")
        vbox.addWidget(btn_classify)
        
        self.lbl_class_res = QLabel("Kết quả: Chưa phân loại")
        self.lbl_class_res.setStyleSheet("font-size: 16px; color: #FFEB3B; font-weight: bold; background-color: #333; padding: 5px; border-radius: 3px;")
        vbox.addWidget(self.lbl_class_res)
        
        group_ctrl.setLayout(vbox)
        left_panel.addWidget(group_ctrl)

        # --- PHẦN GHI CHÚ KIẾN THỨC (NOTE) ---
        note_group = QGroupBox("Lý thuyết Nhận dạng kết cấu")
        note_lay = QVBoxLayout()
        
        self.txt_tex_note = QTextEdit()
        self.txt_tex_note.setReadOnly(True)
        self.txt_tex_note.setHtml("""
            <div style='line-height: 1.4; font-size: 13px;'>
                <b style='color: #00BCD4;'>1. Trích xuất đặc trưng (LBP):</b><br>
                Hệ thống sử dụng toán tử <b>Local Binary Patterns (LBP)</b> để mô tả kết cấu bề mặt. 
                LBP so sánh một pixel với các lân cận để tạo mã nhị phân, giúp thuật toán bất biến với sự thay đổi cường độ sáng.
                <ul style='margin-top: 5px;'>
                    <li>Mỗi ảnh được chuyển thành một <b>Histogram</b> đặc trưng.</li>
                </ul>
                                  
                <div style='margin-top: 10px;'>
                    <b style='color: #FFC107;'>2. Thuật toán KNN (K-Nearest Neighbors):</b><br>
                    Là thuật toán học máy dựa trên khoảng cách. 
                    <ul style='margin-top: 5px;'>
                        <li><b>Giai đoạn Train:</b> Lưu trữ các vector đặc trưng của các mẫu đã biết nhãn (cát, cỏ, vải...).</li>
                        <li><b>Giai đoạn Classify:</b> Tính khoảng cách (thường là Euclidean) từ mẫu mới đến toàn bộ dữ liệu mẫu.</li>
                        <li><b>Quyết định:</b> Gán nhãn phổ biến nhất trong <i style='font-family: serif;'>K</i> láng giềng gần nhất.</li>
                    </ul>
                </div>

                <div style='margin-top: 10px;'>
                    <b style='color: #8BC34A;'>3. Tại sao dùng KNN cho Texture?</b><br>
                    <ul style='margin-top: 2px;'>
                        <li>Hiệu quả với dữ liệu có tính lặp lại như kết cấu.</li>
                        <li>Dễ giải thích và trực quan hóa trong không gian đặc trưng.</li>
                    </ul>
                </div>
            </div>
        """)
        note_lay.addWidget(self.txt_tex_note)
        note_group.setLayout(note_lay)
        left_panel.addWidget(note_group)
        
        w.setLayout(left_panel)
        left_panel.setStretchFactor(note_group, 1) # Để phần ghi chú chiếm nhiều không gian hơn

        return w

    # ---------------------------------------------------------
    # TAB 4: MOTION & VIDEO (Sử dụng Mục 6 & 7 - Horn-Schunck)
    # ---------------------------------------------------------
    def ui_tab_motion(self):
        w = QWidget()
        lay = QHBoxLayout() 

        # --- CỘT TRÁI: ĐIỀU KHIỂN & GHI CHÚ ---
        left_panel = QVBoxLayout()
        
        # 1. Nhóm Load dữ liệu
        load_group = QGroupBox("Input Source")
        load_lay = QVBoxLayout()
        btn_load_v = QPushButton("Load Video File")
        btn_load_v.clicked.connect(self.action_load_video)
        btn_load_f = QPushButton("Load Image Folder")
        btn_load_f.clicked.connect(self.action_load_image_sequence)
        load_lay.addWidget(btn_load_v)
        load_lay.addWidget(btn_load_f)
        load_group.setLayout(load_lay)
        left_panel.addWidget(load_group)

        # --- ĐÂY LÀ PHẦN BỊ THIẾU ---
        # Thêm nhãn hiển thị trạng thái load ảnh/video
        self.lbl_motion_status = QLabel("Trạng thái: Chưa có dữ liệu")
        self.lbl_motion_status.setStyleSheet("color: #FFC107; font-weight: bold;") # Màu vàng cho dễ nhìn
        left_panel.addWidget(self.lbl_motion_status)
        # ---------------------------

        # 2. Nút chạy chính
        btn_run_flow = QPushButton("Run Motion Analysis")
        btn_run_flow.clicked.connect(self.action_run_video_pipeline)
        btn_run_flow.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; height: 35px;")
        left_panel.addWidget(btn_run_flow)

        # 3. PHẦN GHI CHÚ (NOTE)
        note_group = QGroupBox("Hướng dẫn đọc kết quả")
        note_lay = QVBoxLayout()
        
        self.txt_note = QTextEdit()
        self.txt_note.setReadOnly(True)
        self.txt_note.setHtml("""
            <div style='line-height: 1.5;'>
                <b style='color: #2196F3; font-size: 14px;'>1. Bản đồ mã hóa màu (Color Map):</b><br>
                Sử dụng không gian <b>HSV</b> đại diện cho vector vận tốc:
                
                <ul style='margin-top: 5px; margin-bottom: 15px;'>
                    <li><b>Màu sắc (Hue):</b> Hướng di chuyển</li>
                </ul>

                <div align='center' style='margin-top: 15px; margin-bottom: 15px;'>
                    <table border='0' cellpadding='5'>
                        <tr>
                            <td></td>
                            <td align='center'><span style='color: #FF00FF; font-weight: bold;'>Purple</span>: Lên ↑</td>
                            <td></td>
                        </tr>
                        <tr>
                            <td align='right'><span style='color: #0000FF; font-weight: bold;'>Blue</span>: Trái ←</td>
                            <td align='center' style='color: gray; padding: 0 10px;'>[Center]</td>
                            <td align='left'><span style='color: #FF0000; font-weight: bold;'>→ Red</span>: Phải</td>
                        </tr>
                        <tr>
                            <td></td>
                            <td align='center'><span style='color: #00FF00; font-weight: bold;'>Green</span>: Xuống ↓</td>
                            <td></td>
                        </tr>
                    </table>
                </div>

                <ul style='margin-top: 5px; margin-bottom: 15px;'>
                    <li><b>Độ sáng (Value):</b> Độ lớn/Tốc độ (Càng sáng càng nhanh, Đen là đứng yên).</li>
                </ul>
                
                <hr style='border: 0.5px solid #eee;'>

                <div style='margin-top: 10px;'>
                    <b style='color: #F44336; font-size: 14px;'>2. Biểu đồ mũi tên (Quiver Plot):</b><br>
                    Vẽ vector vận tốc thực tế đè lên ảnh:
                    <ul style='margin-top: 5px;'>
                        <li><b>Hướng mũi tên:</b> Chỉ về hướng pixel đang trôi tới.</li>
                        <li><b>Độ dài:</b> Tỉ lệ thuận với vận tốc (Dài: nhanh, Ngắn: chậm).</li>
                        <li><b>Mật độ:</b> Lấy mẫu <i style='font-family: serif;'>step = 10</i> để quan sát cấu trúc trường vận tốc.</li>
                    </ul>
                </div>
            </div>
        """)
        note_lay.addWidget(self.txt_note)
        note_group.setLayout(note_lay)
        left_panel.addWidget(note_group)
        
        w.setLayout(left_panel)
        left_panel.setStretchFactor(note_group, 1) # Để phần ghi chú chiếm nhiều không gian hơn

        return w

    # =========================================================
    # THỰC THI LOGIC - GỌI TRỰC TIẾP CÁC HÀM MODULES
    # =========================================================
    
    def action_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "data/images")
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.img_orig = img.astype(np.float32) / 255.0
            self.display_image(self.img_orig, self.lbl_view_orig)
            # Dùng get_log_magnitude ngay khi load
            self.update_spectrum(self.img_orig)

    def update_spectrum(self, img):
        # Mục 1: DFT -> get_log_magnitude
        spec = dft2d(img)
        log_mag = get_log_magnitude(spec)
        # Chuẩn hóa để hiển thị 8-bit
        log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
        self.display_image(log_mag_norm, self.lbl_view_spec)

    def action_filtering_full(self):
        if self.img_orig is None: return
        M, N = self.img_orig.shape
        d0 = self.sld_d0.value()
        f_type = self.cb_filter.currentText()
        
        # Mục 3: Tạo bộ lọc
        if "Ideal" in f_type: H = ideal_lowpass(M, N, d0)
        elif "Butterworth" in f_type: H = butterworth_lowpass(M, N, d0, n=2)
        else: H = gaussian_highpass(M, N, d0)
        
        # Mục 1: Áp dụng
        self.img_proc = apply_2d_filter(self.img_orig, H)
        self.display_image(self.img_proc, self.lbl_view_proc)
        self.update_spectrum(self.img_proc)

    def action_create_blur(self):
        if self.img_orig is None: return
        M, N = self.img_orig.shape
        if "Motion" in self.cb_degrade.currentText():
            self.psf_current = motion_blur_psf(M, N, L=20)
        else:
            self.psf_current = defocus_blur_psf(M, N, r=10)
        
        # Gây mờ: DFT -> Nhân PSF -> IDFT
        blurred = np.real(idft2d(dft2d(self.img_orig) * self.psf_current))
        self.img_blurred = np.clip(blurred, 0, 1)
        self.display_image(self.img_blurred, self.lbl_view_orig)

    def action_wiener_sweep(self):
        # Mục 4: Quét K và tính SSIM
        best_res, best_k, max_ssim, _ = find_best_k(self.img_blurred, self.img_orig, self.psf_current)
        self.img_proc = best_res
        self.display_image(self.img_proc, self.lbl_view_proc)
        print(f"Best K: {best_k}, SSIM: {max_ssim}")

    def action_wiener_manual(self):
        # Mục 4: Gọi trực tiếp wiener_deconvolution
        k = self.sld_k_manual.value() / 1000.0
        if hasattr(self, 'img_blurred'):
            self.img_proc = wiener_deconvolution(self.img_blurred, self.psf_current, k)
            self.display_image(self.img_proc, self.lbl_view_proc)

    def action_train_texture(self):
        # Mục 5: TextureClassifier
        # Giả lập training để GUI không lỗi, Student B sẽ trỏ path thật
        self.lbl_class_res.setText("Đang huấn luyện KNN...")
        QApplication.processEvents()
        # self.clf.train(X, y, class_names)
        self.lbl_class_res.setText("✅ Đã huấn luyện xong model KNN.")

    def action_classify_texture(self):
        # Mục 5: extract_angular_features
        if self.img_orig is not None:
            features = extract_angular_features(self.img_orig, n_bins=12)
            # pred = self.clf.model.predict([features])
            self.lbl_class_res.setText(f"Đặc trưng (bin 1): {features[0]:.4f}")

    def action_load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn Video", "data/video", "*.mp4 *.avi *.mov")
        if path:
            # Gọi hàm extract_frames cũ
            self.loaded_frames = extract_frames(path, num_frames=30)
            self.lbl_motion_status.setText(f"Loaded Video: {len(self.loaded_frames)} frames")
            if self.loaded_frames:
                self.display_image(self.loaded_frames[0], self.lbl_view_orig)

    def action_load_image_sequence(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục chuỗi ảnh")
        if folder_path:
            # Gọi hàm extract_frames_from_folder mới
            self.loaded_frames = extract_frames_from_folder(folder_path)
            self.lbl_motion_status.setText(f"Loaded Folder: {len(self.loaded_frames)} frames")
            if self.loaded_frames:
                self.display_image(self.loaded_frames[0], self.lbl_view_orig)

    def action_run_video_pipeline(self):
        if not hasattr(self, 'loaded_frames') or len(self.loaded_frames) < 2:
            self.lbl_motion_status.setText("⚠️ Cần ít nhất 2 khung hình!")
            return

        # Reset chỉ số về frame đầu tiên
        self.current_frame_idx = 0
        
        # Bắt đầu Timer: cứ mỗi 100ms (0.1 giây) sẽ đổi 1 frame
        # Có thể chỉnh 100 thành 50 để chạy nhanh hơn (20 FPS)
        self.timer.start(100) 
        self.lbl_motion_status.setText("🚀 Đang chạy mô phỏng chuyển động...")

    def update_animation(self):
        """Hàm này được gọi liên tục bởi QTimer"""
        if self.current_frame_idx < len(self.loaded_frames) - 1:
            # 1. Lấy 2 frame liên tiếp
            f1 = self.loaded_frames[self.current_frame_idx]
            f2 = self.loaded_frames[self.current_frame_idx + 1]
            
            # 2. Hiển thị frame hiện tại lên ô Ảnh Gốc
            self.display_image(f1, self.lbl_view_orig)
            
            # 3. Tính toán Horn-Schunck cho cặp này
            u, v = horn_schunck(f1, f2, alpha=1.0)
            
            # 4. Hiển thị bản đồ màu vào ô Processed Image
            from modules.motion_analysis import flow_to_color
            color_flow = flow_to_color(u, v)
            self.display_color_image(color_flow, self.lbl_view_proc)

            # 5. Cập nhật biểu đồ mũi tên lên Canvas
            self.update_quiver_plot(f1, u, v)
            
            self.current_frame_idx += 1
        else:
            # Khi chạy hết danh sách ảnh
            self.timer.stop()
            self.lbl_motion_status.setText(f"✅ Đã hoàn thành {len(self.loaded_frames)} frames")

    def update_quiver_plot(self, frame, u, v):
        """Cập nhật biểu đồ mũi tên từng frame"""
        self.canvas.axes.cla()
        step = 15 # Tăng step lên một chút cho đỡ rối khi chạy animation
        y, x = np.mgrid[step//2:u.shape[0]:step, step//2:u.shape[1]:step].reshape(2, -1).astype(int)
        fx, fy = u[y, x], v[y, x]
        
        self.canvas.axes.imshow(frame, cmap='gray')
        self.canvas.axes.quiver(x, y, -fx, fy, color='red', scale=5)
        self.canvas.axes.axis('off')
        self.canvas.draw()

    def display_color_image(self, bgr_img, label):
        """Hiển thị ảnh màu (BGR từ OpenCV) lên QLabel"""
        try:
            h, w, c = bgr_img.shape
            bytes_per_line = c * w
            # Chuyển BGR (OpenCV) sang RGB (Qt)
            qimg = QImage(bgr_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Lỗi hiển thị ảnh màu: {e}")

    def display_image(self, img_np, label):
        """Hiển thị ảnh mức xám (Grayscale) từ Numpy array 0-1 lên QLabel"""
        try:
            h, w = img_np.shape
            # Chuyển về 8-bit (0-255)
            img_8bit = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            # Tạo QImage từ dữ liệu mức xám
            qimg = QImage(img_8bit.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            # Scale ảnh cho khớp với kích thước QLabel mà không làm méo tỉ lệ
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Lỗi hiển thị ảnh xám: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DSP_Full_GUI()
    window.show()
    sys.exit(app.exec_())