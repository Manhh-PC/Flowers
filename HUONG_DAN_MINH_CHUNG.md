# Lưu minh chứng theo từng buổi

Chụp màn hình để thể hiện đã thao tác trên VS Code. Dùng PNG/PDF do notebook xuất khi đưa biểu đồ vào báo cáo để tránh ảnh chữ mờ. Giữ cả notebook có outputs và ZIP kết quả, không chỉ giữ screenshot.

| Mốc | Chạy đến | Cần xem/chụp | Cần giải thích |
|---|---|---|---|
| Buổi 1: môi trường và dữ liệu | Bước 4 | Tên notebook trên VS Code; thư viện; số ảnh 8.189/102 lớp; split; audit | Ảnh, nhãn và split là 3 tệp khác nhau; ghép theo image_id |
| Buổi 2: EDA/tiền xử lý | Bước 7 | Phân phối lớp, kích thước/kênh, ảnh mẫu, HOG/HSV, số chiều | Vì sao giữ màu; tác động resize; HOG khác pixel |
| Buổi 3: huấn luyện/chọn | Bước 10 | Baseline; thử k/metric/depth/criterion; validation; mô hình chốt | Vì sao dùng validation; vì sao chọn F1-macro |
| Buổi 4: đánh giá/demo | Bước 14 | Test table; confusion; lớp/cặp nhầm; ảnh lỗi; demo; ZIP | Điểm mạnh/yếu có căn cứ; không chọn lại bằng test |

## Danh mục hình tự xuất

| Tên đầu tệp | Nội dung | Phần báo cáo |
|---|---|---|
| 00_exact_duplicates | Nhóm ảnh trùng hoàn toàn, gồm trùng khác split | 1: EDA; 2: xử lý dữ liệu |
| 01_class_distribution | Nhãn và chia tập chính thức | 1: EDA |
| 02_image_properties | Kích thước/chế độ màu | 1: EDA |
| 03_samples_page_1..6 | Ảnh mẫu đủ 102 lớp | 1 hoặc phụ lục |
| 04_preprocessing_hog_hsv | Ảnh gốc, resize, xám, HOG, HSV | 2: thiết kế |
| 05_knn_tuning | k và metric trên validation | 4: thử nghiệm |
| 06_tree_tuning | Độ sâu/criterion của cây | 4: thử nghiệm |
| 07_pca_variance | Phương sai PCA nếu bật | 4: mở rộng |
| 08_validation_comparison | Bảng/biểu đồ trước khi mở test | 4: chọn mô hình |
| 09_test_comparison | Chỉ số test và thời gian predict | 4: kết quả |
| 10_confusion_full | Confusion matrix đầy đủ | 4 hoặc phụ lục |
| 11_confusion_detail | Nhóm lớp thường nhầm | 4-5 |
| 12_hardest_classes | 15 lớp F1 thấp nhất | 5: phân tích lỗi |
| 13_wrong_predictions | Ảnh sai lấy ngẫu nhiên | 5: phân tích lỗi |
| 14_correct_predictions | Ảnh đúng lấy ngẫu nhiên | 5: đối chiếu |
| 15_demo_test_<image_id> | Ảnh test, nhãn thật/dự đoán, ĐÚNG/SAI và top 5 điểm | 3 hoặc phụ lục |
| 16_demo_uploaded | Demo ảnh tự tải nếu thực hiện | Phụ lục |

PNG 220 dpi và PDF được xuất cùng tên. Tổng số thay đổi nếu tắt PCA hoặc bật demo ảnh cá nhân; đọc `figure_index.csv`, không ghi số hình cố định trước khi chạy.

## Cách ghi nhận xét có căn cứ

1. Ghi giá trị từ CSV, giữ thang 0-1 hoặc chuyển tất cả sang phần trăm nhất quán.
2. Với một so sánh, nêu yếu tố giữ nguyên, yếu tố thay đổi và chênh lệch đo được.
3. Nêu cả kết quả không cải thiện. Không giả định mọi bước phức tạp hơn đều tốt hơn.
4. Khi nói lớp A dễ nhầm B, ghi số lần nhầm và tỷ lệ trên số ảnh lớp A, kèm một ảnh minh họa.
5. Với ảnh sai, ghi quan sát như “hoa nhỏ, nền nhiều lá”. Câu “mô hình chú ý nền” chỉ là giả thuyết nếu chưa kiểm chứng thêm.
6. Đọc F1 từng lớp cùng support. Confusion matrix con bỏ bớt cột nên các hàng có thể không cộng thành 1.
7. Thời gian predict ở bảng không bao gồm trích đặc trưng; xem log thời gian feature riêng. Thời gian phụ thuộc CPU và việc cache đã có hay chưa.
8. Đối chiếu test Oxford gốc (6.149 ảnh) và test bổ sung không trùng (6.145 ảnh). Đưa hình 00 và bảng loại trùng vào phần EDA/thiết kế; nêu rõ dataset gốc vẫn giữ nguyên. Không sử dụng khác biệt hai bảng để chọn lại mô hình.

## Cần gửi lại để viết báo cáo

- Notebook `.ipynb` đã chạy, giữ outputs.
- ZIP `..._ket_qua.zip` của đúng lần chạy muốn báo cáo.
- Template LaTeX được giảng viên cung cấp khi có.
- Thông tin nhóm/lớp và phân công thực tế; địa chỉ repository thống nhất.
- 5-10 nhận xét của bạn về các hình, những bước chưa hiểu hoặc lỗi đã gặp.

Không cần gửi lại toàn bộ 8.189 ảnh. Không dùng số liệu kiểm thử mã bằng dữ liệu giả để viết kết quả thực nghiệm.
