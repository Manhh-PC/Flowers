# Ảnh chụp và các mục tự điền trước khi nộp

Báo cáo dùng số liệu thật của lần chạy `20261117_160946_514913`. Toàn bộ 14 bước đã chạy thành công. Các biểu đồ và bảng đã được chèn trực tiếp từ kết quả; không cần chụp lại mọi biểu đồ hoặc toàn bộ mã nguồn.

## 1. Điền thông tin cá nhân

- Mở `report/latex/thong_tin.tex`, thay các phần `(Điền...)` bằng trường, khoa, lớp, nhóm, họ tên và mã sinh viên thật. Trong mã có chú thích `%` hướng dẫn.
- Nếu nhóm có 3-4 người, thêm dòng trong bảng trang bìa ở `report/latex/bao_cao.tex`.
- Điền công việc thực tế và minh chứng trong `report/latex/sections/phu_luc.tex`; cập nhật `TEAM.md`. Không tự xác nhận đã làm hoặc đã kiểm chứng nếu chưa thực hiện.
- Điền địa chỉ GitHub sau khi push. Hiện tại chưa push, chưa có remote.
- Chưa có template LaTeX chính thức của học phần trong các tệp đã cung cấp. Bản hiện tại dùng bố cục A4, tiếng Việt, đúng 6 phần bắt buộc. Nếu giảng viên yêu cầu template riêng, chuyển nội dung vào template đó trước khi nộp.

## 2. Năm ảnh chụp đề nghị

Đây là minh chứng thao tác bổ sung do bạn chụp, không phải yêu cầu số lượng ảnh được trích nguyên văn từ tài liệu học phần. Bảng/hình thực nghiệm trong báo cáo đã có sẵn.

Mở `Ket_qua_thuc_nghiem_FIT4013.ipynb` trong VS Code để xem notebook đã lưu toàn bộ outputs. Không cần chạy lại toàn bộ chỉ để chụp các bảng. Nếu muốn tự tái tạo, dùng notebook thực hành chính, chạy từ đầu; lần chạy mới có mã khác và có thể khác thời gian đo.

| Ảnh | Mở ở đâu | Nội dung cần nhìn thấy | Lưu với tên |
|---|---|---|---|
| 1 | Notebook kết quả, Bước 2 | Tên notebook, kernel `.venv`, phiên bản Python/thư viện và cấu hình | `01_moi_truong.png` |
| 2 | Notebook kết quả, cuối Bước 10 | Bảng validation và dòng `Mô hình đã chốt: M8_HOG_HSV_KNN_tuned` | `02_chon_mo_hinh.png` |
| 3 | Notebook kết quả, Bước 11 | Bảng test Oxford và dòng của M8. Nếu hai bảng không vừa, thêm ảnh phụ bảng test không trùng | `03_ket_qua_test.png` |
| 4 | `Demo_mo_hinh_da_luu.ipynb`, ví dụ đúng | Ô `TEST_INDEX = 30`, ảnh ID 34, nhãn thật/dự đoán và chữ ĐÚNG | `04_demo_dung.png` |
| 5 | Cùng notebook demo, ví dụ sai | Ô `TEST_INDEX = 0`, ảnh ID 1, nhãn thật/dự đoán và chữ SAI | `05_demo_sai.png` |

Với demo: chọn kernel `.venv`, chạy ô khởi tạo một lần rồi chạy ô ví dụ. Notebook nạp model đã lưu, không huấn luyện lại. Chỉ số 30/0 được chọn từ kết quả hiện tại để minh họa hai tình huống; không dùng hai ảnh để suy ra Accuracy.

Chụp bằng **Win+Shift+S** hoặc công cụ chụp màn hình của Windows. Giữ đủ tiêu đề bước và kết quả để đối chiếu; không thu nhỏ chữ đến mức khó đọc. Không cần ảnh toàn màn hình nếu có thông tin cá nhân ngoài bài học. Không sửa nhãn hoặc số liệu trên ảnh.

Chép năm ảnh vào `report/latex/screenshots/`. Mã LaTeX tự chèn ảnh tồn tại khi biên dịch lại; nếu chưa có ảnh thì không tạo trang trống. Bạn cũng có thể gửi các ảnh trong cuộc trò chuyện để được kiểm tra trước khi chèn.

## 3. Biên dịch và nộp

- Toàn bộ nguồn nằm trong `report/latex/`. Trên Overleaf, tải ZIP nguồn, đặt file chính `bao_cao.tex`, chọn compiler **XeLaTeX** rồi Recompile.
- Trên máy có XeLaTeX: chạy `xelatex bao_cao.tex` hai lần từ thư mục `report/latex/` để cập nhật mục lục/tham chiếu. Có thể dùng `tectonic bao_cao.tex` nếu đã cài Tectonic.
- Sau khi thêm thông tin và ảnh, kiểm tra lại trang bìa, mục lục, bảng, hình và phụ lục trong PDF mới.
- Đọc kỹ phần thảo luận: M8 được chọn trên validation dù M5 cao hơn trên test; Accuracy khoảng 15,84% còn thấp, không mô tả sản phẩm là nhận dạng tin cậy.
- Nộp PDF, mã nguồn/notebook, README, yêu cầu thư viện và khai báo AI theo hướng dẫn. Không đưa `.venv`, cache hoặc archive dataset lên GitHub.

Không có yêu cầu phải chụp từng dòng mã. Các đoạn mã trọng yếu về đặc trưng, chọn mô hình và đánh giá đã được trình bày dưới dạng văn bản trong báo cáo, dễ đọc và kiểm tra hơn ảnh chụp dài.
