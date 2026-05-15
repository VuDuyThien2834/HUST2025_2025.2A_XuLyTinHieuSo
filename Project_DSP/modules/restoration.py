# [Mục 4] PSF Motion/Defocus blur & Wiener Deconvolution
'''
Hai loại PSF: motion_blur_psf (ngang) và defocus_blur_psf (hình đĩa).
Hàm khôi phục chính: wiener_deconvolution.
Cơ chế quét K: Tự động tính SSIM để tìm $K$ tối ưu.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from modules.fourier import dft2d, idft2d

def motion_blur_psf(M, N, L):
    """PSF làm mờ chuyển động ngang 1D độ dài L (L = 10, 30, 50)."""
    psf = np.zeros((M, N))
    psf[0, :L] = 1.0 / L
    H = dft2d(psf)
    return H

def defocus_blur_psf(M, N, r):
    """PSF làm mờ mất nét (đĩa bán kính r = 5, 15, 25)."""
    psf_centered = np.zeros((M, N))
    y, x = np.ogrid[-M//2:M//2, -N//2:N//2]
    mask = x**2 + y**2 <= r**2
    psf_centered[mask] = 1
    psf_centered /= np.sum(psf_centered)
    
    # Đưa về dạng không shift để nhân miền tần số
    psf = np.fft.ifftshift(psf_centered)
    H = dft2d(psf)
    return H

def wiener_deconvolution(blurred_img, H, K):
    """Triển khai bộ lọc Wiener: H*(f) / (|H(f)|^2 + K)"""
    F = dft2d(blurred_img)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H)**2
    
    W = H_conj / (H_mag_sq + K)
    G = W * F
    
    result = np.real(idft2d(G))
    return np.clip(result, 0, 1)

def find_best_k(blurred_img, original_img, H, k_values=[0.0001, 0.001, 0.01, 0.1, 1]):
    """
    [Yêu cầu 4] Quét K trên 5 giá trị; chọn K bằng tiêu chí SSIM tối đa.
    Trả về: ảnh tốt nhất, giá trị K tốt nhất, và danh sách tất cả kết quả để vẽ biểu đồ.
    """
    best_k = k_values[0]
    max_ssim = -1
    best_res = None
    sweep_results = []

    for k in k_values:
        res = wiener_deconvolution(blurred_img, H, k)
        # Tính SSIM để chọn K tốt nhất
        score = ssim(original_img, res, data_range=1.0)
        sweep_results.append((k, score, res))
        
        if score > max_ssim:
            max_ssim = score
            best_k = k
            best_res = res
            
    return best_res, best_k, max_ssim, sweep_results

def plot_wiener_results(original, blurred, sweep_results, best_k, max_ssim):
    """
    [Deliverable 4] Tạo hình ảnh minh họa cho báo cáo: 
    Ảnh mờ, kết quả K-sweep, và kết quả tốt nhất.
    """
    n = len(sweep_results)
    plt.figure(figsize=(15, 8))
    
    # Hiển thị ảnh mờ đầu vào
    plt.subplot(2, n, 1)
    plt.title("Blurred Input")
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')

    # Hiển thị kết quả quét K (K-sweep)
    for i, (k, score, res) in enumerate(sweep_results):
        plt.subplot(2, n, n + i + 1)
        title = f"K={k}\nSSIM={score:.4f}"
        if k == best_k:
            title += " (BEST)"
        plt.title(title)
        plt.imshow(res, cmap='gray')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test case giả lập đúng yêu cầu đề bài
    img = np.random.rand(256, 256) # Thay bằng ảnh thực tế khi chạy project
    M, N = img.shape
    
    # 1. Tạo ảnh mờ (ví dụ Motion Blur L=30)
    H = motion_blur_psf(M, N, L=30)
    blurred = np.real(idft2d(dft2d(img) * H))
    
    # 2. Thực hiện quét K (K-sweep) - 5 giá trị
    best_img, b_k, b_ssim, sweep_data = find_best_k(blurred, img, H)
    
    print(f"K tốt nhất tìm được: {b_k} với SSIM: {b_ssim:.4f}")
    
    # 3. Xuất đồ thị phục vụ Deliverable
    # plot_wiener_results(img, blurred, sweep_data, b_k, b_ssim)