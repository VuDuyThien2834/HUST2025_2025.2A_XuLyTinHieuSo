# HUST2025_2025.2A_XuLyTinHieuSo

## 0. Yêu cầu:
- Có cài conda

## 1. Chuẩn bị môi trường conda:
    $ source 1_create_environment.sh
    $ conda activate dsp_project
    $ pip install -r  1_requirements.txt

## 2. Chuẩn bị data:
    Nếu chưa có ảnh trong các thư mục con tại Project_DSP/data/, run:
    $ python 2_download_images.py
    $ python 3_prepare_data.py

## 3. Chạy code:
    $ python Project_DSP/main_gui.py