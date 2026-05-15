# [Mục 1, 3] Các bộ lọc (Butterworth, Gaussian, Notch...)

'''
các hàm Bộ lọc: ideal_lowpass, butterworth_lowpass, ideal_highpass, gaussian_highpass, và notch_reject_filter.

Thực thi: Hàm apply_2d_filter(image, H) sử dụng chính xác quy trình nhân miền tần số.

Cơ chế: Ma trận khoảng cách get_dist_matrix khớp với định dạng không shift của dft2d.
'''
import numpy as np
from modules.fourier import dft2d, idft2d

def get_dist_matrix(M, N):
    """Tính ma trận khoảng cách D(u,v) phục vụ thiết kế bộ lọc."""
    u = np.arange(M)
    v = np.arange(N)
    # Dịch chuyển để tần số 0 nằm ở góc (khớp với dft2d không shift)
    u[u > M//2] -= M
    v[v > N//2] -= N
    uu, vv = np.meshgrid(v, u)
    return np.sqrt(uu**2 + vv**2)

# (1) Bộ lọc thông thấp lý tưởng (Ideal Lowpass)
def ideal_lowpass(M, N, D0):
    D = get_dist_matrix(M, N)
    H = np.zeros((M, N))
    H[D <= D0] = 1
    return H

# (2) Bộ lọc thông thấp Butterworth (Butterworth Lowpass) bậc 1-10
def butterworth_lowpass(M, N, D0, n=2):
    D = get_dist_matrix(M, N)
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

# (3) Bộ lọc thông cao lý tưởng (Ideal Highpass)
def ideal_highpass(M, N, D0):
    D = get_dist_matrix(M, N)
    H = np.ones((M, N))
    H[D <= D0] = 0
    return H

# (4) Bộ lọc thông cao Gaussian (Gaussian Highpass)
def gaussian_highpass(M, N, D0):
    D = get_dist_matrix(M, N)
    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    return H

# (5) Bộ lọc Notch 2D (Đặt notch thủ công)
def notch_reject_filter(M, N, D0, centers, n=2):
    """
    centers: danh sách tọa độ nhiễu [(u1, v1), (u2, v2)...] 
    được xác định thủ công từ phổ tần số.
    """
    H = np.ones((M, N))
    D_mat = get_dist_matrix(M, N)
    
    for u_c, v_c in centers:
        # Tính khoảng cách tới điểm nhiễu và điểm đối xứng qua tâm
        D1 = np.roll(np.roll(D_mat, u_c, axis=0), v_c, axis=1)
        D2 = np.roll(np.roll(D_mat, -u_c, axis=0), -v_c, axis=1)
        
        # Công thức Notch Butterworth Reject
        H_k = (1 / (1 + (D0 / (D1 + 1e-6))**n)) * (1 / (1 + (D0 / (D2 + 1e-6))**n))
        H *= H_k
    return H

def apply_2d_filter(image, H):
    """Quy trình: DFT -> Nhân phổ -> IDFT"""
    F = dft2d(image)
    G = F * H
    filtered_image = idft2d(G)
    return np.real(filtered_image)

if __name__ == "__main__":
    print("--- Kiểm tra 5 bộ lọc hệ thống ---")
    img = np.random.rand(256, 256)
    M, N = img.shape
    
    # Test nhanh sự tồn tại của 5 bộ lọc
    filters = [
        ideal_lowpass(M, N, 30),
        butterworth_lowpass(M, N, 30, n=2),
        ideal_highpass(M, N, 30),
        gaussian_highpass(M, N, 30),
        notch_reject_filter(M, N, 5, [(50, 50)])
    ]
    
    for i, H in enumerate(filters):
        res = apply_2d_filter(img, H)
        print(f"Bộ lọc {i+1}: Hoàn thành (Output shape: {res.shape})")