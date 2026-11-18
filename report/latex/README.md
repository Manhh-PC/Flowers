# Nguồn LaTeX báo cáo FIT4013

- File chính: `bao_cao.tex`.
- Thông tin tự điền: `thong_tin.tex`; đóng góp thực tế trong `sections/phu_luc.tex`.
- Sáu phần nội dung: `sections/01_...` đến `sections/06_...`.
- Hình có thật: `figures/`; bảng/số liệu sinh từ CSV: `generated/`.
- Ảnh chụp bổ sung: `screenshots/`, tên và hướng dẫn trong `CHUP_ANH_VA_HOAN_THIEN.md` đi kèm gói.

Biên dịch với **XeLaTeX** (hai lần) hoặc Tectonic. Không dùng pdfLaTeX vì báo cáo sử dụng fontspec/Unicode. Phông ưu tiên Times New Roman; tự chuyển TeX Gyre Termes nếu máy không có. Mọi đường dẫn trong nguồn LaTeX đều tương đối.

Lần chạy nguồn: `20261117_160946_514913`, Oxford Flowers 102 thật. Mô hình M8 chọn bằng validation, test Accuracy 15,84%, F1-macro 14,73%. Không thay các số trong văn bản bằng số của lần chạy khác mà chưa kiểm tra lại toàn bộ bảng và nhận xét.

Nguồn này tuân theo thứ tự sáu phần của tài liệu học phần, nhưng **chưa phải template chính thức** vì chưa được cung cấp template. Cần đối chiếu/chuyển sang template nếu giảng viên yêu cầu.

Để sinh lại bảng/hình trong dự án đầy đủ, chạy `report/prepare_report.py` với mã lần chạy theo hướng dẫn đầu tệp. Script này không fit mô hình. Nó tái tạo notebook demo và bảng/hình, nên không chạy khi đã chỉnh thủ công các đầu ra đó mà chưa sao lưu. Nội dung nhận xét do biên soạn riêng, cần đọc lại nếu đổi run ID.
