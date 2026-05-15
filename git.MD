1. Khởi tạo Local Repository
git init

2. Thêm các tệp vào khu vực chờ (Staging Area)
git add .

3. Xác nhận thay đổi (Commit)
git commit -m "Initial commit - Add code project"

4. Tạo Repository trên GitHub
URL:
https://github.com/VuDuyThien2834/HUST2025_2025.2A_XuLyTinHieuSo.git

5. Kết nối Local Repo với GitHub
git remote add origin https://github.com/VuDuyThien2834/HUST2025_2025.2A_XuLyTinHieuSo.git

6. Kéo các file được tạo trên Github Repo về
git pull origin main --rebase

6. Đẩy code lên GitHub (Push)
git branch -M main
git push -u origin main