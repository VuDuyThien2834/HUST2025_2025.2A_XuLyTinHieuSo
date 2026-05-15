# [Mục 7] Pipeline xử lý 30 frames, tính Motion Magnitude

import numpy as np
import cv2
import matplotlib.pyplot as plt
from modules.motion_analysis import horn_schunck
import os

def extract_frames(video_path, num_frames=30, target_size=(256, 256)):
    """
    Case 1: Trích xuất từ video.
    Đọc video và trích xuất số lượng khung hình chỉ định.
    Chuyển về grayscale và chuẩn hóa để sẵn sàng tính toán.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tiền xử lý: Xám -> Resize (để Horn-Schunck chạy nhanh) -> Chuẩn hóa
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        frames.append(resized.astype(np.float32) / 255.0)
        count += 1
        
    cap.release()
    return frames

def extract_frames_from_folder(folder_path, target_size=(256, 256)):
    """
    Case 2: Trích xuất từ thư mục ảnh.
    Đọc chuỗi ảnh từ một thư mục thay vì file video.
    """
    # Lấy danh sách file ảnh và sắp xếp theo tên để đúng thứ tự thời gian
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    files.sort() 
    
    frames = []
    for f in files:
        img_path = os.path.join(folder_path, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, target_size)
            frames.append(resized.astype(np.float32) / 255.0)
            
    return frames

def compute_video_motion(frames, alpha=1.0, n_iter=50):
    """
    Chạy Horn-Schunck qua chuỗi khung hình để tính toán Motion Magnitude.
    """
    magnitudes = []
    
    # Tính toán giữa các cặp khung hình liên tiếp (t và t+1)
    for i in range(len(frames) - 1):
        u, v = horn_schunck(frames[i], frames[i+1], alpha=alpha, n_iter=n_iter)
        
        # Magnitude tại mỗi pixel: sqrt(u^2 + v^2)
        mag_field = np.sqrt(u**2 + v**2)
        
        # Lấy giá trị trung bình của toàn khung hình làm đại diện cho frame đó
        avg_mag = np.mean(mag_field)
        magnitudes.append(avg_mag)
        
        print(f"Đang xử lý cặp frame {i+1}/{len(frames)-1} - Magnitude: {avg_mag:.6f}")
        
    return magnitudes

def save_motion_plot(magnitudes, output_path):
    """
    [Deliverable 7] Vẽ và lưu biểu đồ biến thiên độ lớn chuyển động.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(magnitudes) + 1), magnitudes, 'b-o', linewidth=2, markersize=4)
    plt.title("Biểu đồ Motion Magnitude (30 Frames)")
    plt.xlabel("Thứ tự cặp Frame")
    plt.ylabel("Độ lớn chuyển động trung bình")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu vào thư mục outputs/figures/
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu biểu đồ tại: {output_path}")

if __name__ == "__main__":
    # Test nhanh logic
    print("--- Kiểm tra module video_utils.py ---")
    
    # Tạo 30 khung hình giả lập (hình vuông di chuyển chéo)
    fake_frames = []
    for i in range(30):
        f = np.zeros((256, 256))
        f[50+i:100+i, 50+i:100+i] = 1.0
        fake_frames.append(f)
        
    mags = compute_video_motion(fake_frames)
    
    if len(mags) == 29:
        print("✅ SUCCESS: Đã tính toán xong Magnitude cho chuỗi frame.")
        # save_motion_plot(mags, "test_magnitude_curve.png")