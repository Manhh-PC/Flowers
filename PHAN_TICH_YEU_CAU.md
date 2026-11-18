# Phân tích yêu cầu đề tài 7

Đã đọc toàn bộ 11 trang của tài liệu người học cung cấp. Đề tài 7 ở mục 4.2, trang 9; phải áp dụng đồng thời yêu cầu chung ở mục 2 và quy trình thị giác máy tính ở mục 4.1, trang 8. Yêu cầu riêng của đề tài văn bản/dữ liệu bảng không áp dụng cho bài phân loại hoa.

## 1. Bài toán và dữ liệu

- Đầu vào: một ảnh hoa. Đầu ra: một trong 102 lớp Oxford Flowers 102.
- Dùng đúng dataset được chỉ định, không thay bằng Iris, Flowers-5 hoặc tự chọn một vài lớp.
- Oxford mô tả 40-258 ảnh/lớp và biến đổi về kích thước, tư thế, ánh sáng; nguồn: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.
- Dùng 3 tệp ảnh/nhãn/split. Không cần tải segmentation hoặc khoảng cách tính sẵn.
- Kiểm chứng số lượng từ metadata: 8.189 ảnh, 102 lớp; train 1.020, validation 1.020, test 6.149.
- Giữ split chính thức; không mặc định chia 80/20 hoặc đảo tập test thành train. Nếu làm thêm split khác, phải báo rõ đó là thí nghiệm khác, không so trực tiếp với split Oxford.
- Không có thông tin chắc chắn về từng bệnh lý dữ liệu trước khi chạy: ảnh hỏng, trùng, kích thước và kênh phải được kiểm tra thực tế.

**Cập nhật từ audit dữ liệu thật khi chuyển sang VS Code:** đủ 8.189 ảnh, không ảnh hỏng/nhãn thiếu; có 8 ảnh thuộc 4 nhóm trùng RGB. Một nhóm train-test (7307/7309), một nhóm validation-test (8077/8067), hai nhóm nội bộ test (7323/7328 và 7885/7886). Dữ liệu/chia tập gốc được giữ nguyên. Báo cáo thêm đánh giá trên 6.145 ảnh test không trùng, loại các ID 7309, 7328, 7886, 8067 theo quy tắc nội dung ảnh xác định trước khi fit. Không chọn lại mô hình theo bảng bổ sung.

## 2. Đối chiếu yêu cầu bắt buộc

| Yêu cầu từ tài liệu | Cách thực hiện | Bước | Minh chứng |
|---|---|---|---|
| Đúng dataset, rõ đầu vào/đầu ra | 102 lớp, URL chính thức, metadata, MD5 | 3 | nguồn, checksums, split manifest |
| EDA: số ảnh/lớp, phân phối | Bảng toàn bộ và từng split | 3-5 | class_counts.csv, hình 01 |
| Ảnh thiếu/hỏng/trùng, bất thường | Đọc tất cả ảnh, hash RGB, thống kê kích thước/kênh | 4 | audit_summary.json, image_audit.csv |
| Ảnh mẫu từng lớp | Một ảnh train cho mỗi lớp, 6 trang | 5 | hình 03_samples_page_1..6 |
| Resize, normalize | RGB 64×64, pixel/255; cùng kích thước mọi nhánh | 6 | hình 04, cấu hình |
| Cân nhắc grayscale | Chỉ HOG dùng xám; pixel/HSV giữ màu | 6 | giải thích và thí nghiệm HOG+HSV |
| Baseline vector pixel | K-NN trên RGB pixel, cấu hình cố định | 8 | M1_Pixel_KNN_baseline |
| HOG/đặc trưng cổ điển | HOG và histogram HSV | 6-9 | vector, hình HOG/HSV, bảng so sánh |
| K-NN và/hoặc cây | Chọn cả hai thuật toán đã học | 8-9 | M1-M8 |
| Ít nhất hai phương án có ý nghĩa | So đặc trưng giữ thuật toán; so thuật toán giữ đặc trưng | 8-11 | bảng validation/test |
| Thử siêu tham số | k, metric, max_depth, criterion | 9 | hyperparameter_trials.csv, hình 05-06 |
| Tách tập, chống leakage | Fit trên train; chọn trên val; test sau khi chốt | 3, 7-11 | frozen_selection.json |
| Seed/cấu hình tái tạo | seed 42, phiên bản môi trường và cache theo cấu hình | 2,7 | run_config.json, requirements-lock.txt |
| Accuracy, Precision, Recall, F1 | Macro cho so sánh, weighted để bổ sung; metric từng lớp | 8-12 | validation/test_results, classification_report |
| Confusion Matrix | Lưu 102×102 mọi mô hình; vẽ toàn bộ và trích chi tiết mô hình đã chọn | 11-12 | CSV, hình 10-11 |
| Phân tích lỗi, lớp nhầm | Cặp nhầm, lớp F1 thấp, ảnh đúng/sai | 12 | hình 12-14, bảng phân tích để người học điền |
| So sánh và hạn chế | Đọc kết quả, giải thích từ hình/bảng thật | Sau chạy | báo cáo phần 4-6 |
| Repo/hướng dẫn/thư viện | Bộ tệp đi kèm, README, khóa phiên bản của lần chạy | Toàn bộ | repository được thống nhất |
| Báo cáo LaTeX/PDF | Đúng 6 phần, cần đối chiếu template chính thức khi có | Sau thực hành | report/latex và report/pdf, dùng kết quả thật; người học điền thông tin |
| Nhóm 2-4 người, đóng góp và hiểu mã | Phân công thật, commit thật, trả lời câu hỏi | Toàn bộ | TEAM.md, lịch sử commit |
| Khai báo dùng AI | AI_USAGE.md và kiểm chứng của nhóm | Toàn bộ | khai báo được cập nhật trước nộp |

## 3. Các thí nghiệm và câu hỏi cần trả lời

| Mã | Biểu diễn / thuật toán | Vai trò |
|---|---|---|
| M0 | DummyClassifier most_frequent | Mốc phụ không dùng đặc trưng, không thay M1 |
| M1 | Pixel RGB + K-NN k=5 Euclidean | Baseline bắt buộc |
| M2 | Pixel RGB + Tree depth=20 gini | Đổi thuật toán trên cùng pixel |
| M3 | HOG + K-NN như M1 | Đổi biểu diễn, giữ cấu hình K-NN |
| M4 | HOG + Tree như M2 | Đổi biểu diễn, giữ cấu hình cây |
| M5 | HOG+HSV + K-NN như M3 | Thêm màu với cùng cấu hình |
| M6 | HOG + K-NN chọn k/metric trên val | Ảnh hưởng siêu tham số K-NN |
| M7 | HOG + Tree chọn depth/criterion trên val | Ảnh hưởng siêu tham số cây |
| M8 | HOG+HSV + K-NN chọn trên val | Đánh giá biểu diễn kết hợp |
| M9 | Scale+PCA128 trên HOG+HSV, K-NN giống M8 | Bổ sung giảm chiều, đánh đổi tốc độ/hiệu năng |

M1/M3, M2/M4 và M3/M5 là các cặp so sánh có kiểm soát tốt hơn việc chỉ đối chiếu hai pipeline thay đổi nhiều thứ. M9 thêm cả scale và PCA, vì vậy không quy mọi khác biệt cho PCA riêng lẻ. Không hứa trước phương án cải tiến sẽ có điểm cao hơn.

Mô hình cuối là phương án có validation F1-macro cao nhất, trừ M0. Test lớn và lệch số ảnh giữa các lớp; F1-macro tránh để các lớp đông ảnh chi phối toàn bộ kết luận. Recall macro cũng tương ứng trung bình recall từng lớp. Tất cả mô hình dùng cùng train, không gộp val vào train sau khi chọn trong bản thực hành này.

## 4. Phần bổ sung và điểm thưởng

- PCA là lựa chọn bổ sung, không bắt buộc bật nếu nguồn lực hạn chế; quyết định trước test.
- GUI/ứng dụng: tài liệu nêu cộng 0,5-1,0 điểm tùy chất lượng. Demo notebook là bước khởi đầu, chưa tự coi là GUI hoàn chỉnh.
- Tái tạo tốt: tối đa 0,5 điểm, cần repo thực sự chạy lại được.
- Kỹ thuật mở rộng: tối đa 0,5 điểm nếu có lý do, thí nghiệm so sánh và giải thích. Học sâu không thay phần cổ điển bắt buộc.
- Tổng điểm sau thưởng không quá 10. Không có cam kết đạt điểm thưởng chỉ vì đã gọi thư viện.

## 5. Rubric để ưu tiên công việc

| Tiêu chí | Trọng số |
|---|---:|
| Xác định bài toán, dữ liệu, nguồn | 10% |
| Tiền xử lý và biểu diễn | 15% |
| Pipeline mô hình | 20% |
| Đánh giá và so sánh | 20% |
| Diễn giải và khuyến nghị | 15% |
| Tái lập, repo, log, quản lý nhóm | 10% |
| Báo cáo, slide, demo/vấn đáp | 10% |

Liêm chính học thuật và sử dụng AI là điều kiện/điểm trừ riêng. Tài liệu cho phép AI hỗ trợ, nhưng người học phải kiểm chứng và giải thích được. Không chèn toàn bộ mã vào báo cáo. Trình bày các đoạn trọng yếu: ghép nhãn theo ID, trích HOG/HSV, fit/train/val/test, tính metric và chọn mô hình.

## 6. Những phần cần hoàn tất sau khi chạy VS Code

1. Xác nhận số lượng và audit; xử lý vấn đề có thật nếu xuất hiện.
2. Đọc bảng validation/test, viết nhận xét dựa trên số liệu và hình thật.
3. Chọn 6-12 ảnh lỗi để mô tả quan sát và giả thuyết; tránh kết luận nhân quả chưa kiểm chứng.
4. Hoàn thiện báo cáo theo template LaTeX của học phần, điền đúng thành viên/lớp.
5. Đưa mã và báo cáo lên repository được thống nhất; lưu đóng góp thực tế, khai báo AI.
6. Chuẩn bị slide/demo và trả lời câu hỏi kỹ thuật; mở rộng GUI hoặc transfer learning nếu còn thời gian.

Chưa có template LaTeX, địa chỉ GitHub, tên/thông tin thành viên hoặc hạn nộp trong nội dung người học đã cung cấp. Không tự đặt những thông tin này.
