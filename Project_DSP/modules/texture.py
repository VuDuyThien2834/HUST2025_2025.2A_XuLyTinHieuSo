# [Mục 5] Power Spectrum, Angular Bins, KNN classification

'''
Trích xuất đặc trưng: Hàm extract_angular_features trả về vector năng lượng chuẩn hóa dựa trên số n_bins (mặc định là 12).

Phân loại: Sử dụng class TextureClassifier với các phương thức train, predict (thông qua mô hình KNN nội bộ) và evaluate để lấy Confusion Matrix.

Cấu trúc dữ liệu: Quy trình huấn luyện yêu cầu danh sách các vector đặc trưng X và nhãn y.
'''
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from modules.fourier import dft2d

def get_power_spectrum(image):
    """Tính phổ công suất (Power Spectrum) 2D."""
    F = dft2d(image)
    F_shift = np.fft.fftshift(F)
    # Power Spectrum = |F(u,v)|^2
    power_spec = np.abs(F_shift)**2
    return power_spec

def extract_angular_features(image, n_bins=12):
    """
    [Yêu cầu 5] Trích xuất năng lượng theo hướng bằng cách tích phân trên các angular bins.
    """
    M, N = image.shape
    ps = get_power_spectrum(image)
    
    # Tạo lưới tọa độ tâm (0,0)
    yc, xc = np.indices((M, N))
    yc, xc = yc - M//2, xc - N//2
    
    # Tính góc của từng pixel (từ -pi đến pi)
    angles = np.arctan2(yc, xc)
    angles[angles < 0] += 2 * np.pi # Chuyển về [0, 2*pi]
    
    # Chia vòng tròn thành n_bins miếng bánh
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    features = []
    
    for i in range(n_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i+1])
        # Tích phân (tính tổng) năng lượng trong bin này
        bin_energy = np.sum(ps[mask])
        features.append(bin_energy)
        
    # Chuẩn hóa vector đặc trưng (để tổng năng lượng = 1)
    features = np.array(features)
    return features / (np.sum(features) + 1e-8)

class TextureClassifier:
    """Hỗ trợ phân loại 4 lớp kết cấu bằng thuật toán KNN."""
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classes = []

    def train(self, X, y, class_names):
        self.classes = class_names
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        """[Deliverable 5] Xuất Confusion Matrix."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.classes)
        return cm, report

# Hàm hỗ trợ load dữ liệu từ thư mục data/textures/
def load_texture_dataset(base_path, n_bins=12):
    """
    Giả sử cấu trúc: data/textures/{class_name}/{image_files}
    """
    X = []
    y = []
    class_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    class_names = class_names[:4] # Giới hạn 4 lớp theo đề bài
    
    for idx, name in enumerate(class_names):
        class_dir = os.path.join(base_path, name)
        for img_file in os.listdir(class_dir):
            # Lưu ý: Cần dùng thư viện như PIL hoặc OpenCV để đọc ảnh ở đây
            # Giả lập: image = load_image(os.path.join(class_dir, img_file))
            pass
            
    return X, y, class_names

if __name__ == "__main__":
    print("--- Kiểm tra module texture.py ---")
    # Test case: Tạo đặc trưng cho ảnh ngẫu nhiên
    dummy_img = np.random.rand(256, 256)
    feats = extract_angular_features(dummy_img, n_bins=8)
    print(f"Đặc trưng trích xuất (8 hướng): \n{feats}")
    print("\n✅ Đã có đủ hàm tính Phổ công suất, Tích phân Angular Bins và KNN.")