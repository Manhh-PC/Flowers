# Kiểm tra bản VS Code ngày 01/11/2026

## Đã kiểm tra bằng dữ liệu Oxford thật

- Archive `102flowers.tgz` do người học tải có MD5 đúng, chứa 8.189 ảnh JPEG.
- Archive phân đoạn chứa 8.189 ảnh; đã đặt trong `data/raw/`, chưa dùng làm đặc trưng chính.
- Hai tệp nhãn/chia tập tải từ Oxford, MD5 đúng; 102 lớp, train=1.020, validation=1.020, test=6.149.
- Chạy Bước 1-6 bằng kernel Jupyter thực trong `.venv` của dự án: thành công; đây là kiểm tra thực thi cục bộ, không phải ảnh chụp thao tác trong giao diện VS Code.
- Đọc toàn bộ ảnh: 0 ảnh hỏng, 0 nhãn thiếu, 0 xung đột nhãn trong các nhóm trùng.
- Có 8 ảnh trong 4 nhóm trùng RGB, gồm 2 nhóm khác split. Không có trùng train-validation.
- Giữ dataset/split gốc; tạo tập chỉ số test bổ sung 6.145 ảnh, loại ID 7309, 7328, 7886, 8067 theo quy tắc nội dung ảnh trước huấn luyện.
- Đã tạo và kiểm tra hình minh họa trùng lặp và tiền xử lý. Hình/bảng ban đầu được lưu trong `report/evidence/setup/`.
- Notebook kiểm tra có outputs: `outputs/checked_through_step_06.ipynb`. Khi bổ sung demo, giữ nguyên kết quả Bước 1-2 người học đã lưu trong notebook chính.

## Đã kiểm tra kỹ thuật mô hình

- Schema notebook và cú pháp tất cả 16 ô Python hợp lệ sau khi thêm ô chọn ảnh test.
- Trong môi trường `.venv` mới, chạy các nhánh xử lý/huấn luyện/đánh giá/xuất tệp bằng 180 ảnh texture giả chỉ dành cho kiểm thử.
- Kiểm tra pixel/HOG/HOG+HSV đúng số chiều; 10 cấu hình; 18 thử tham số; scaler/PCA fit train; chọn trên validation; lưu/nạp model; CSV/ma trận/ZIP.
- Bộ kiểm thử chủ động có trùng train-test và trùng nội bộ test; kiểm tra bảng đánh giá bổ sung loại đúng các mẫu, giữ đúng số mẫu cho cả 10 mô hình.
- `pip check` không báo xung đột; `requirements-lock.txt` lưu phiên bản đã cài.
- `.gitignore` đã được kiểm tra: dataset, `.venv`, cache, outputs, ZIP không vào danh sách Git. Chưa có commit/remote/push.
- Demo chọn ảnh test đã kiểm tra riêng: chọn đúng tập ID không trùng, đối chiếu đúng/sai, chặn chỉ số ngoài phạm vi, lưu hình/bảng riêng theo ID; thứ tự top 5 khi bằng điểm khớp quy tắc của K-NN/cây. Kiểm tra này dùng dữ liệu giả, không tạo số đo phân loại hoa.

## Thực nghiệm thật phục vụ báo cáo

- Đã chạy đủ 16 ô mã / 14 bước bằng kernel Jupyter trên Oxford thật, run `20261117_160946_514913`, không thay notebook đang thực hành của người học.
- 10 mô hình, 18 thử tham số, 21 hình, model lưu/nạp nhất quán; đủ bảng test gốc và test loại trùng.
- M8 chọn trên validation (F1-macro 0,1696355794); test Accuracy 0,1583997398, F1-macro 0,1473182851; test loại trùng Accuracy 0,1583401139, F1-macro 0,1471678984.
- Đối chiếu số ảnh dự đoán, trace/tổng confusion matrix và 6.145 mẫu cho từng ma trận loại trùng: đạt.
- `Ket_qua_thuc_nghiem_FIT4013.ipynb` giữ outputs thực tế. `Demo_mo_hinh_da_luu.ipynb` đã chạy riêng với model lưu, kiểm tra ví dụ đúng ID 34/chỉ số 30 và sai ID 1/chỉ số 0; không fit lại.
- Số liệu giả từ kiểm thử kỹ thuật không nằm trong kết quả báo cáo. Các mốc kiểm tra giả phía trên chỉ là lịch sử kiểm tra mã.

## Phần người học cần hoàn tất

Điền trang bìa/đóng góp thực tế, tự kiểm chứng và hiểu mã/nhận xét, chụp thao tác VS Code nếu dùng minh chứng giao diện, đối chiếu template học phần và push GitHub khi sẵn sàng. Hướng dẫn cụ thể: `report/CHUP_ANH_VA_HOAN_THIEN.md`.

Chưa kiểm chứng các thao tác chọn kernel/chạy ô trong giao diện VS Code của người học; môi trường và file workspace đã được chuẩn bị. Sau khi mở notebook, chọn kernel `.venv` và chạy từ Bước 1 để tạo lại biến trong bộ nhớ.
