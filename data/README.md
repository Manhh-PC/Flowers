# Dữ liệu cục bộ (không đưa lên GitHub)

Đặt tệp đã tải vào `data/raw/`:

```
data/
  raw/
    102flowers.tgz
    imagelabels.mat
    setid.mat
    102segmentations.tgz  # tùy chọn, chưa dùng trong thí nghiệm chính
  jpg/                    # notebook giải nén tự động
```

Nguồn: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

- `102flowers.tgz`: 8.189 ảnh gốc; MD5 `52808999861908f626f3c1f4e79d11fa`.
- `imagelabels.mat`: nhãn 1..102 của từng ảnh; MD5 `e0620be6f572b9609742df49c70aed4d`.
- `setid.mat`: train/validation/test chính thức; MD5 `a5357ecc9cb78c4bef273ce3793fc85c`.
- `102segmentations.tgz`: ảnh đã phân đoạn; không chứa nhãn phân loại thay cho imagelabels.mat.
- `distancematrices102.mat`: không cần cho pipeline tự trích đặc trưng này.

Nếu thiếu hai tệp `.mat`, Bước 3 tải tự động từ Oxford và kiểm tra MD5. Nếu thiếu archive ảnh, chép vào đúng vị trí rồi chạy lại. `.gitignore` loại trừ các tệp dữ liệu lớn, chỉ giữ tài liệu này.
