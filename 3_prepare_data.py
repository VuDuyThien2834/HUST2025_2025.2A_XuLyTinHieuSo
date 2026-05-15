import cv2
import os
import numpy as np

# 1. Tự động lấy đường dẫn của thư mục chứa file script này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def split_textures():
    # 1. Cấu hình đường dẫn
    input_dir = os.path.join(BASE_DIR, 'Project_DSP', 'data', 'textures')  # Nơi chứa các file ảnh .tiff
    output_base = os.path.join(BASE_DIR, 'Project_DSP', 'data', 'textures')
    patch_size = 128
    max_patches = 20 # Số lượng ảnh con tối đa mỗi loại

    # Ánh xạ tên file vào tên thư mục tương ứng
    mapping = {
        "Brodatz-BrickWall-D94.tiff": "brick",
        "Brodatz-Grass-D9.tiff": "grass",
        "Brodatz-BeachSand-D29.tiff": "sand"
    }

    for filename, folder_name in mapping.items():
        img_path = os.path.join(input_dir, filename)
        
        # Kiểm tra file tồn tại
        if not os.path.exists(img_path):
            print(f"⚠️ Không tìm thấy file: {filename}")
            continue

        # Đọc ảnh (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Không thể đọc file: {filename}")
            continue

        # Tạo thư mục đích
        target_dir = os.path.join(output_base, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        h, w = img.shape
        count = 0
        
        # Tách ảnh thành các patch 128x128
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                if count >= max_patches:
                    break
                
                # Cắt ảnh
                patch = img[i:i+patch_size, j:j+patch_size]
                
                # Lưu ảnh con
                save_path = os.path.join(target_dir, f"{folder_name}_{count:02d}.png")
                cv2.imwrite(save_path, patch)
                count += 1
            
            if count >= max_patches:
                break
        
        print(f"✅ Đã tách {count} ảnh nhỏ vào thư mục: {folder_name}/")

if __name__ == "__main__":
    split_textures()