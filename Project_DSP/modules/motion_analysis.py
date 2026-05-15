# [Mục 6] Horn-Schunck Optical Flow

'''
compute_derivatives(img1, img2): Tính $I_x, I_y, I_t$.
horn_schunck(img1, img2, alpha, n_iter): Hàm cốt lõi trả về $(u, v)$, có chuẩn hóa ảnh về [0, 1].
visualize_flow(img, u, v, step): Dùng quiver plot để hiển thị trường vector.
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def compute_derivatives(img1, img2):
    """
    Tính toán các đạo hàm không gian và thời gian (Ix, Iy, It).
    Sử dụng kernel trung bình để tính đạo hàm giữa hai khung hình.
    """
    # Kernel đạo hàm theo Horn-Schunck
    kernel_x = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernel_y = np.array([[-1, -1], [1, 1]]) * 0.25
    kernel_t = np.array([[1, 1], [1, 1]]) * 0.25

    Ix = convolve(img1, kernel_x) + convolve(img2, kernel_x)
    Iy = convolve(img1, kernel_y) + convolve(img2, kernel_y)
    It = convolve(img2, kernel_t) - convolve(img1, kernel_t)

    return Ix, Iy, It

def horn_schunck(img1, img2, alpha=1.0, n_iter=100):
    """
    Triển khai thuật toán Horn-Schunck.
    alpha: Hệ số mượt (smoothness weight). alpha càng lớn, trường vector càng mượt.
    n_iter: Số vòng lặp Gauss-Seidel.
    """
    # Chuyển ảnh về float [0, 1] để tính toán chính xác
    img1 = img1.astype(np.float32) / 255.0 if img1.max() > 1 else img1
    img2 = img2.astype(np.float32) / 255.0 if img2.max() > 1 else img2

    # Khởi tạo trường vector vận tốc ban đầu (u, v) bằng 0
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)

    # Tính các đạo hàm Ix, Iy, It
    Ix, Iy, It = compute_derivatives(img1, img2)

    # Kernel để tính giá trị trung bình cục bộ (Laplacian approximation)
    # Đây là kernel 3x3 chuẩn cho Horn-Schunck
    kernel_avg = np.array([[1/12, 1/6, 1/12],
                           [1/6,    0, 1/6],
                           [1/12, 1/6, 1/12]])

    for i in range(n_iter):
        # Tính giá trị trung bình của u và v tại lân cận
        u_avg = convolve(u, kernel_avg)
        v_avg = convolve(v, kernel_avg)

        # Công thức cập nhật lặp Gauss-Seidel:
        # u = u_avg - [Ix * (Ix*u_avg + Iy*v_avg + It)] / (alpha^2 + Ix^2 + Iy^2)
        
        numerator = Ix * u_avg + Iy * v_avg + It
        denominator = alpha**2 + Ix**2 + Iy**2
        
        u = u_avg - Ix * (numerator / denominator)
        v = v_avg - Iy * (numerator / denominator)

    return u, v

def visualize_flow(img, u, v, step=8):
    """
    Hàm hỗ trợ vẽ các vector vận tốc lên trên ảnh nền.
    """
    
    plt.imshow(img, cmap='gray')
    
    # Tạo lưới tọa độ để vẽ mũi tên (Quiver plot)
    y, x = np.mgrid[step/2:img.shape[0]:step, step/2:img.shape[1]:step].reshape(2, -1).astype(int)
    fx, fy = u[y, x], v[y, x]
    
    plt.quiver(x, y, fx, -fy, color='red', scale=5) # -fy vì trục y trong ảnh ngược với đồ thị
    plt.title("Optical Flow Field (Horn-Schunck)")
    plt.show()

# def flow_to_color(u, v):
#     """
#     Chuyển đổi u, v sang ảnh màu HSV để trực quan hóa.
#     Màu sắc (Hue) = Hướng di chuyển.
#     Độ sáng (Value) = Độ lớn di chuyển.
#     """
#     h, w = u.shape
#     hsv = np.zeros((h, w, 3), dtype=np.uint8)
#     hsv[..., 1] = 255  # Độ bão hòa tối đa

#     # Tính độ lớn và góc (hướng)
#     mag, ang = cv2.cartToPolar(u, v)
    
#     # Chuẩn hóa góc về [0, 180] cho OpenCV HSV
#     hsv[..., 0] = ang * 180 / np.pi / 2
#     # Chuẩn hóa độ lớn về [0, 255]
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
#     # Chuyển từ HSV sang BGR để hiển thị
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return bgr

def flow_to_color(u, v):
    h, w = u.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255 

    # Đổi dấu u và v nếu hướng bị ngược so với thực tế
    # Thử đảo dấu u để Đỏ quay về hướng Phải:
    mag, ang = cv2.cartToPolar(-u, v) # Thêm dấu trừ trước u
    
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

if __name__ == "__main__":
    print("--- Đang kiểm tra Horn-Schunck Optical Flow ---")
    # Tạo 2 khung hình giả lập: 1 hình vuông di chuyển
    img1 = np.zeros((100, 100))
    img1[40:60, 40:60] = 1.0
    
    img2 = np.zeros((100, 100))
    img2[42:62, 42:62] = 1.0 # Di chuyển xuống dưới và sang phải 2px
    
    u, v = horn_schunck(img1, img2, alpha=1.0, n_iter=50)
    
    # Tính vận tốc trung bình để kiểm tra
    print(f"Vận tốc trung bình u (ngang): {np.mean(u[42:62, 42:62]):.4f}")
    print(f"Vận tốc trung bình v (dọc): {np.mean(v[42:62, 42:62]):.4f}")