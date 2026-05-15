import os
import cv2
from skimage import data

# 1. Tự động lấy đường dẫn của thư mục chứa file script này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Xây dựng đường dẫn tương đối
# os.path.join sẽ tự xử lý dấu "/" hoặc "\" tùy theo Windows hay Linux
IMAGE_DIR = os.path.join(BASE_DIR, 'Project_DSP', 'data', 'images')

# Tạo thư mục
os.makedirs(IMAGE_DIR, exist_ok=True)

# 3. Lưu ảnh từ thư viện chuẩn của Python
cv2.imwrite(os.path.join(IMAGE_DIR, 'cameraman.jpg'), data.camera())
cv2.imwrite(os.path.join(IMAGE_DIR, 'astronaut.png'), data.astronaut()[:,:,0])

print(f"Done! Ảnh mẫu được lưu tại: {IMAGE_DIR}")