# Dàn ý báo cáo theo FIT4013

Đây là dàn ý đối chiếu yêu cầu. Báo cáo đã được soạn trong `latex/bao_cao.tex` và xuất tại `pdf/bao_cao.pdf`, dùng kết quả thật run `20261117_160946_514913`. Điền thông tin nhóm và đối chiếu template chính thức trước khi nộp; hướng dẫn tại `CHUP_ANH_VA_HOAN_THIEN.md`.

## 1. Đặt vấn đề và phân tích yêu cầu

Bài toán phân loại ảnh thành 102 lớp hoa, đầu vào/đầu ra, ý nghĩa và phạm vi. Nguồn Oxford, trích dẫn công trình Nilsback & Zisserman 2008. Trình bày số liệu audit thật, phân phối nhãn, kích thước/kênh, ảnh mẫu. Phân biệt dữ liệu công khai với quyền tái phân phối; không tự gán giấy phép.

Minh chứng: audit_summary.json, split_manifest.csv, class_counts.csv, hình 01-03.

## 2. Thiết kế giải pháp và quy trình học máy

Mô tả luồng: tải/kiểm tra dữ liệu → split Oxford → tiền xử lý → đặc trưng → fit train → chọn validation → test → phân tích lỗi. Giải thích resize/RGB/normalize/HOG/HSV, cấu hình, và lợi ích/hạn chế giả thuyết. Với PCA, giải thích fit scaler/PCA trong train. Thuật toán K-NN và cây, cách chọn tham số, seed 42, chống leakage.

Minh chứng: run_config.json, hình 04, frozen_selection.json.

## 3. Phân tích mã nguồn và chức năng

Trình bày cấu trúc notebook và vai trò nhóm hàm: tải/kiểm tra, extract_features, feature_matrix, fit_validate/tune, score_metrics, export_checkpoint và demo_image. Chọn vài đoạn mã ngắn cần thiết để giải thích. Giải thích lưu/nạp model và demo cùng pipeline đặc trưng.

## 4. Thử nghiệm và kết quả

Thông tin môi trường VS Code và phiên bản thật. Số ảnh train/val/test; M1 là baseline, M0 là mốc bổ sung. Bảng tham số thử và tiêu chí validation F1-macro. Bảng Accuracy/Precision macro/Recall macro/F1 macro/F1 weighted; thời gian fit/predict và số chiều. Hình tuning, PCA nếu có, so sánh test, confusion matrix. Chỉ ghi số từ lần chạy được lưu.

Minh chứng: requirements-lock.txt, hyperparameter_trials.csv, validation_results.csv, test_results.csv, logs/, hình 05-11.

Với dữ liệu đã kiểm tra: phải báo cáo ảnh trùng đã phát hiện và đối chiếu cả `test_results.csv` (6.149 ảnh Oxford gốc) với `test_results_unique_unseen.csv` (6.145 ảnh test không trùng). Dùng `evaluation_policy.json` và `test_duplicate_exclusions.csv` để giải thích quy tắc; không coi test gốc hoàn toàn độc lập với train/validation.

## 5. Thảo luận và so sánh

So M1/M3, M2/M4, M3/M5, thuật toán và tuning; M8/M9 nếu có. Không chỉ nói phương án nào cao nhất. Phân tích ảnh lỗi với ID, lớp thật/dự đoán, số lượng/tỷ lệ cặp nhầm, support từng lớp. Nêu hạn chế: ít ảnh train/lớp, thay đổi góc/nền/ánh sáng, resize méo tỷ lệ, mất thông tin màu của HOG, histogram bỏ vị trí, phát hiện trùng chưa bao gồm mọi ảnh gần giống, validation chỉ có một split, điểm mô hình chưa hiệu chỉnh và chỉ phân loại trong 102 lớp.

Minh chứng: top_confused_pairs.csv, selected_model_per_class.csv, error_analysis_to_complete.csv sau khi nhóm điền, hình 12-14.

## 6. Tổng kết và hướng phát triển

Tóm tắt kết quả đo thật, mức đáp ứng mục tiêu, phương án chọn trên validation. Nêu cải tiến có thể kiểm chứng: resize giữ tỷ lệ, đặc trưng màu theo vùng, tăng cường dữ liệu chỉ trên train, đánh giá nhiều seed/split độc lập nếu được phép, transfer learning như mở rộng có so sánh, giao diện demo. Không viết các cải tiến dự kiến như đã thực hiện.

## Tài liệu tham khảo và phụ lục

Trích nguồn dataset, API, mã tham khảo; khai báo AI và đóng góp thật. Đưa ma trận 102 lớp đầy đủ, các trang ảnh mẫu, bảng chi tiết vào phụ lục nếu cần. Nộp PDF xuất từ LaTeX cùng mã nguồn theo yêu cầu. Slide được chuẩn bị sau khi kết quả và báo cáo đã ổn định.
