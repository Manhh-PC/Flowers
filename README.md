# Đề tài 7: Phân loại hoa

Thực hành **Jupyter Notebook (.ipynb) trong VS Code**, dùng Python và Oxford Flowers 102. Chạy từng ô như Colab; dữ liệu và kết quả lưu trên máy. Đường dẫn dữ liệu trong mã tính tương đối từ thư mục dự án.

## Mở bài thực hành trên máy hiện tại

1. Trong VS Code chọn **File → Open Folder**, mở thư mục `flower_project` chứa README này.
2. Mở `De_tai_07_Phan_loai_hoa_FIT4013.ipynb`.
3. Góc trên bên phải, chọn **Select Kernel → Python Environments → .venv** của dự án. Nếu chưa hiện, dùng **Python: Select Interpreter → Enter interpreter path**, chọn `.venv/Scripts/python.exe`, rồi chọn lại kernel.
4. Chạy từng ô bằng nút ▶ hoặc Shift+Enter, bắt đầu Bước 1 đến Bước 4. Không cần GPU.
5. Chụp bảng số ảnh/chia tập và bảng kiểm tra chất lượng; sau đó tiếp tục EDA, đặc trưng và mô hình khi đã hiểu kết quả.
6. Nhấn **Ctrl+S** để lưu notebook cùng outputs. Khi Restart Kernel hoặc mở lại phiên, chạy lại các ô trước để tạo biến; outputs cũ không khôi phục bộ nhớ Python.

VS Code cần tiện ích **Python** và **Jupyter** của Microsoft. Workspace đã có tệp gợi ý hai tiện ích và cấu hình dùng `.venv`.

## Cài lại trên máy khác sau khi clone

Cài Python 3.12 và hai tiện ích trên. Tại Terminal của **thư mục dự án**, chạy trên Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu máy dùng lệnh `python` thay cho launcher `py`, dùng `python -m venv .venv` sau khi xác nhận Python 3.12. Không cần đổi execution policy hoặc chạy Activate.ps1; gọi trực tiếp Python trong `.venv` được.

Để dùng đúng phiên bản đã kiểm tra, thay `requirements.txt` bằng `requirements-lock.txt` nếu tệp khóa đã được tạo. Trên macOS/Linux, tạo môi trường bằng `python3 -m venv .venv`, dùng `.venv/bin/python` cho các lệnh tiếp theo. Không đưa `.venv` lên GitHub.

## Cấu trúc dữ liệu và kết quả

```
flower_project/
├── De_tai_07_Phan_loai_hoa_FIT4013.ipynb
├── flowers102_steps.py
├── requirements.txt
├── .vscode/
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── 102flowers.tgz
│   │   ├── imagelabels.mat
│   │   ├── setid.mat
│   │   └── 102segmentations.tgz
│   └── jpg/                  # giải nén tự động
├── cache/                    # đặc trưng đã tính
├── outputs/<ma-lan-chay>/     # hình, bảng, model, cấu hình
├── exports/                  # ZIP minh chứng
├── demo_images/              # ảnh cá nhân thử mô hình
└── report/                   # báo cáo và minh chứng chọn lọc
```

Chép archive ảnh đã tải vào `data/raw/102flowers.tgz`. Bước 3 kiểm tra MD5, giải nén, tự tải hai tệp nhỏ `imagelabels.mat` và `setid.mat` nếu thiếu. Archive có sẵn không tải lại. Tệp sai MD5 sẽ báo lỗi để kiểm tra, không âm thầm ghi đè.

`102segmentations.tgz` chứa ảnh đã phân đoạn, giữ làm phần mở rộng và **không thay thế nhãn phân loại**. Thí nghiệm chính dùng ảnh gốc, pixel/HOG/HSV. `distancematrices102.mat` không cần dùng.

Nguồn: [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). Xem `data/README.md` để biết MD5 và vai trò từng tệp.

## Lộ trình 14 bước

| Bước | Nội dung |
|---|---|
| 1-4 | Môi trường, dữ liệu, nhãn/split, ảnh hỏng/trùng |
| 5-7 | EDA, mẫu đủ 102 lớp, RGB/HOG/HSV, trích đặc trưng |
| 8-10 | Baseline, so sánh thuật toán, tham số, PCA tùy chọn; chốt bằng validation |
| 11-12 | Test, confusion matrix, lớp/cặp nhầm, phân tích ảnh lỗi |
| 13-14 | Lưu/nạp model, demo, ZIP minh chứng |

Train=1.020, validation=1.020, test=6.149; tổng 8.189 ảnh/102 lớp. Giữ split chính thức và nhãn 1..102. Không đảo test thành train hoặc ghép nhãn theo thứ tự duyệt thư mục.

M1 là baseline pixel RGB 64×64 / 255 + K-NN(k=5, Euclidean). M0 Dummy là mốc phụ. M1-M5 so sánh biểu diễn/thuật toán với cấu hình cố định; M6-M8 chọn k/metric hoặc depth/criterion trên validation; M9 thêm scaler+PCA128. Chọn bằng F1-macro trên validation, không fit trên test, không chọn lại sau khi xem test.

Audit trên dữ liệu thật phát hiện 4 cặp trùng RGB: một cặp train-test, một cặp validation-test và hai cặp nội bộ test. Giữ nguyên split Oxford, đồng thời đánh giá bổ sung trên **6.145 ảnh test không trùng**. Bảng `test_results.csv` ghi test gốc; `test_results_unique_unseen.csv` ghi đánh giá bổ sung cùng mô hình, cùng dự đoán. Bốn ID bỏ khỏi bảng bổ sung là 7309, 7328, 7886, 8067; không xóa ảnh khỏi dataset. Chi tiết được tính và ghi lại tự động ở Bước 4.

Notebook lưu PNG 220 dpi, PDF biểu đồ, bảng CSV, dự đoán từng ảnh, classification report, confusion matrix và model. ZIP trong `exports/` không kèm dataset/cache. Xem `figure_index.csv` trong từng lần chạy để biết chú thích.

## Test một ảnh trong bộ dữ liệu

1. Chạy lần lượt Bước 1–13 trong notebook để có mô hình đã huấn luyện. Chỉ chạy Bước 1–6 chưa đủ để dự đoán.
2. Tại mục **Thực hành · chọn một ảnh trong tập test**, chạy ô:

```python
TEST_INDEX = 0
demo_result = demo_test_image(TEST_INDEX)
```

3. Đổi `0` thành `1`, `2`, `100`... rồi chạy lại riêng ô này. Với bộ dữ liệu hiện tại, chỉ số hợp lệ từ `0` đến `6144`; đây là vị trí trong danh sách test không trùng, không phải image_id hoặc nhãn hoa.
4. Đọc nhãn thật, nhãn dự đoán, kết quả ĐÚNG/SAI và bảng 5 lớp có điểm cao nhất. Điểm mô hình không phải xác suất chắc chắn dự đoán đúng. Nhãn thật chỉ dùng để đối chiếu, không đưa vào bộ phân loại.
5. Lấy hình tại `outputs/<mã-lần-chạy>/figures/15_demo_test_<image_id>.png` (có cả PDF); bảng đối chiếu và top 5 nằm trong `tables/`. Mỗi ảnh lưu theo ID riêng. Chọn cả ví dụ đúng và sai cho báo cáo.

Đổi ảnh trong cùng phiên không cần huấn luyện lại. Sau Restart Kernel, chạy lại các ô trước để khôi phục biến. Dùng Bước 11 để xem chất lượng trên toàn bộ tập test; một ảnh demo chỉ là minh họa. Không dùng ảnh train để báo cáo khả năng phân loại ảnh mới. Ô **Tùy chọn · thử ảnh của bạn** dành cho ảnh ngoài dataset và không tự có nhãn thật để đối chiếu.

## Chuẩn bị GitHub

`.gitignore` loại trừ `.venv`, dataset, cache, outputs từng lần chạy, ZIP và model lớn. Giữ notebook, mã nguồn, tài liệu, yêu cầu thư viện và báo cáo. Sau thực hành, chép hình/bảng cần nộp vào `report/evidence/`. Dataset được tái tạo bằng hướng dẫn và nguồn tải, không đưa archive vài trăm MB vào Git.

Chưa cần URL repository để thực hành. Khi bạn đã tạo repository và sẵn sàng push, thao tác tại thư mục dự án:

```powershell
git init -b main
git status --short
git add .
git diff --cached --stat
git commit -m "Add Flowers102 classification practice"
git remote add origin <URL_REPOSITORY_CUA_BAN>
git push -u origin main
```

Thay placeholder bằng URL thật. Nếu đã khởi tạo Git hoặc đã có origin, bỏ qua bước tương ứng; không ghi đè remote sẵn có. Kiểm tra staged files trước commit. Không tự gán tên/email thành viên để tạo lịch sử đóng góp.

## Tệp hướng dẫn

Báo cáo LaTeX/PDF dùng kết quả thật run `20261117_160946_514913` được chuẩn bị trong `report/`. Xem `report/CHUP_ANH_VA_HOAN_THIEN.md` để điền thông tin và chụp đúng 5 màn hình. `Ket_qua_thuc_nghiem_FIT4013.ipynb` có toàn bộ outputs; `Demo_mo_hinh_da_luu.ipynb` nạp model đã lưu để demo nhanh, không huấn luyện lại. Giữ mô hình tại đường dẫn `outputs/<run_id>/models/selected_model.joblib` khi dùng notebook demo.

- `PHAN_TICH_YEU_CAU.md`: yêu cầu, thí nghiệm và rubric.
- `HUONG_DAN_MINH_CHUNG.md`: hình/bảng cần lưu theo từng buổi.
- `report/DE_CUONG.md`: đúng 6 phần báo cáo LaTeX, chưa phải báo cáo nộp.
- `TEAM.md`, `AI_USAGE.md`: điền đóng góp thực tế và ghi nhận AI.
- `VALIDATION.md`: phạm vi kiểm tra và giới hạn.
- `build_notebook.py`: tạo lại notebook từ script; chỉ chạy khi chủ ý đồng bộ vì xóa outputs của notebook đích.
- `scripts/check_setup.py`: kiểm tra bằng kernel thật, mặc định Bước 1-4; không thay cách thực hành từng ô.

## Kiểm tra kỹ thuật tùy chọn

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/check_setup.py --through-step 4
```

Bản notebook kiểm tra có outputs lưu riêng trong `outputs/`; notebook chính không bị ghi đè. Dùng `--through-step 6` để kiểm tra thêm EDA/tiền xử lý, hoặc 14 để chạy toàn bộ khi cần.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không thấy kernel | Chọn `.venv/Scripts/python.exe`; kiểm tra Python/Jupyter extension và ipykernel |
| Thiếu thư viện | Cài requirements bằng đúng Python `.venv` |
| Không thấy dữ liệu | Mở đúng thư mục dự án, kiểm tra `data/raw/102flowers.tgz` |
| MD5 sai | Kiểm tra/tải lại đúng tệp Oxford, không bỏ kiểm tra |
| NameError sau khi mở lại | Chạy lại từ Bước 1 |
| Ảnh hỏng hoặc trùng train-validation | Xem audit và xử lý trước huấn luyện, không tự xóa mẫu |
| K-NN chạy lâu | Đợi ô kết thúc, không chạy trùng; phần này dùng CPU |
| Điểm thấp | Phân tích lỗi thật, không sửa số hoặc đánh giá bằng train |

## Nguồn

- Nilsback, M.-E. và Zisserman, A. (2008), *Automated Flower Classification over a Large Number of Classes*, ICVGIP; [dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
- [Tên lớp, MD5, split](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/flowers102.html).
- [HOG](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog), [K-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- [Jupyter Notebook trong VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
- Tài liệu FIT4013 năm học 2026-2027 của ThS. Lê Thị Thùy Trang do người học cung cấp.
