# [Mục 1] 2D DFT tự triển khai, kiểm tra sai số 1e-10

'''
3 hàm cốt lõi:

    dft2d(image): Biến đổi Fourier 2D tự triển khai.

    idft2d(spectrum): Biến đổi ngược 2D tự triển khai.

    get_log_magnitude(spectrum): Tính phổ biên độ logarit (đã căn giữa).
'''
import numpy as np
import matplotlib.pyplot as plt

def dft2d(image): 
    """
    Triển khai 2D DFT bằng phương pháp Row-Column Decomposition. 
    Sử dụng np.fft.fft (1D) để tối ưu tốc độ nhưng vẫn giữ đúng thuật toán tách lớp. 
    """
    # Bước 1: FFT 1D theo từng hàng 
    rows_fft = np.fft.fft(image, axis=1) 

    # Bước 2: FFT 1D theo từng cột của kết quả bước 1 
    dft_result = np.fft.fft(rows_fft, axis=0) 
    
    return dft_result 

def idft2d(spectrum): 
    """ 
    Triển khai 2D Inverse DFT bằng phương pháp Row-Column Decomposition. 
    """ 
    # Bước 1: IFFT 1D theo từng hàng 
    rows_ifft = np.fft.ifft(spectrum, axis=1) 

    # Bước 2: IFFT 1D theo từng cột 
    idft_result = np.fft.ifft(rows_ifft, axis=0) 
    
    return idft_result 

def get_log_magnitude(spectrum): 
    """ 
    Tính phổ biên độ logarit đã được căn giữa (shifted). 
    Dùng để hiển thị hình ảnh phổ. 
    """ 
    # Căn giữa phổ (đưa tần số 0 vào giữa) 
    shifted = np.fft.fftshift(spectrum) 

    # Tính Magnitude và áp dụng Log scale: log(1 + |F|) 
    magnitude = np.abs(shifted) 
    log_magnitude = np.log(1 + magnitude) 
    
    return log_magnitude 

if __name__ == "__main__": 
    print("--- Đang kiểm tra module fourier.py ---") 

    # 1. Tạo ảnh giả lập (Synthetic Image) để test 
    # Tạo một hình vuông trắng giữa nền đen 
    img = np.zeros((256, 256)) 
    img[100:156, 100:156] = 1 
    
    # 2. Thực hiện DFT bằng hàm tự viết 
    my_dft = dft2d(img) 
    
    # 3. Thực hiện DFT bằng NumPy để đối chứng 
    np_dft = np.fft.fft2(img) 
    
    # 4. Kiểm tra sai số (Yêu cầu đề bài < 1e-10) 
    diff = np.abs(my_dft - np_dft) 
    max_diff = np.max(diff) 
    print(f"Sai số lớn nhất so với NumPy: {max_diff:.2e}") 
    
    if max_diff < 1e-10: 
        print("SUCCESS: Độ chính xác đạt yêu cầu (< 1e-10)!") 
    else: 
        print("FAILED: Sai số quá lớn!") 
    
    # 5. Kiểm tra tính thuận nghịch (DFT -> IDFT) 
    img_back = np.abs(idft2d(my_dft)) 
    reconstruction_error = np.max(np.abs(img - img_back)) 
    print(f"Sai số khôi phục (Original vs IDFT): {reconstruction_error:.2e}") 
    
    # 6. Trực quan hóa kết quả 
    log_spec = get_log_magnitude(my_dft)
    
    plt.figure(figsize=(12, 4)) 
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("Log Magnitude Spectrum")
    plt.imshow(log_spec, cmap='viridis')
    
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Image")
    plt.imshow(img_back, cmap='gray')
    
    plt.tight_layout()
    plt.show()