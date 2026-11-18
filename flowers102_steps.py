# %% [markdown]
# # Đề tài 7 · Phân loại hoa Oxford Flowers 102
# **FIT4013 · Thực hành Python trong VS Code bằng Jupyter Notebook**
#
# Chạy từng ô từ trên xuống bằng nút ▶ hoặc Shift+Enter. Sau mỗi bước, đọc phần
# **Cần hiểu / Minh chứng** trước khi tiếp tục. Lần đầu nên dừng ở cuối Bước 4
# để kiểm tra dữ liệu cùng người hướng dẫn. Không dùng Run all ngay khi mới học.
#
# Notebook này chưa chứa kết quả thực nghiệm. Mọi số đo, hình, bảng và dự đoán
# bên dưới được tạo từ lần chạy thật của bạn. CPU đủ cho phần bắt buộc.
#
# **Đúng yêu cầu:** pixel baseline, HOG, K-NN, Cây quyết định, thay đổi tham số,
# Accuracy/Precision/Recall/F1/Confusion Matrix, phân tích lỗi, khả năng tái tạo.
# HOG + màu HSV và PCA là thí nghiệm bổ sung; không cần học sâu để hoàn thành.
#
# **Quy tắc:** dùng toàn bộ 102 lớp; giữ chia train/validation/test chính thức.
# Chỉ dùng train để fit, validation để chọn cấu hình. Đóng băng lựa chọn trước
# khi đánh giá test. Không đổi tham số sau khi xem kết quả test.
# Audit dữ liệu thật tìm thấy ảnh trùng: giữ kết quả trên test Oxford gốc và
# báo cáo thêm test loại ảnh trùng train/validation hoặc trùng nội bộ (Bước 4, 11).
#
# Nguồn: [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/),
# [quy ước split và tên lớp của Torchvision](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/flowers102.html),
# [HOG](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog),
# [scikit-learn](https://scikit-learn.org/stable/).
# Yêu cầu học phần: tài liệu FIT4013 do người học cung cấp, mục 2 và 4.

# %% [markdown]
# ## Bước 1 · Chuẩn bị thư viện
# Mở thư mục dự án trong VS Code, mở file .ipynb, chọn **Select Kernel → Python
# Environments → .venv**. Môi trường riêng được chuẩn bị theo README.
# Chạy bằng CPU. Ô này kiểm tra đúng môi trường; không tự cài thư viện vào Python khác.
# **Minh chứng:** chụp phiên bản Python/thư viện xuất hiện ở Bước 2.

# %%
import importlib.util
import subprocess
import sys

PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "skimage": "scikit-image",
    "matplotlib": "matplotlib", "PIL": "Pillow", "requests": "requests",
    "joblib": "joblib", "tqdm": "tqdm", "IPython": "ipython",
    "ipykernel": "ipykernel", "tzdata": "tzdata",
}
missing = [pkg for module, pkg in PACKAGES.items()
           if importlib.util.find_spec(module) is None]
if missing:
    raise RuntimeError("Thiếu thư viện: " + ", ".join(missing)
        + ". Chọn kernel .venv của dự án hoặc cài requirements.txt theo README.")
print("Thư viện sẵn sàng. Python:", sys.version.split()[0])
print("Đang dùng môi trường:", sys.prefix.split("/")[-1].split("\\")[-1])

# %% [markdown]
# ## Bước 2 · Cấu hình và thư mục minh chứng
# `SEED=42` giúp tái tạo các thao tác có yếu tố ngẫu nhiên. Split Oxford đã cố định,
# không phải được tạo bằng seed này. Đường dẫn tính từ thư mục dự án:
# `data/raw/` chứa file tải về, `data/jpg/` chứa ảnh đã giải nén, `cache/` chứa
# đặc trưng, `outputs/<mã-lần-chạy>/` chứa minh chứng, `exports/` chứa ZIP.
# Các tệp gốc trong Downloads không bị sửa. Ctrl+S lưu notebook và outputs hiển thị.

# %%
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import tarfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import display
from PIL import Image, ImageOps
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from skimage.color import rgb2gray, rgb2hsv
from skimage.exposure import rescale_intensity
from skimage.feature import hog
from tqdm.auto import tqdm

SEED = 42
IMAGE_SIZE = 64
RUN_PCA = True  # Phần bổ sung; có thể False trước khi bắt đầu thí nghiệm.
np.random.seed(SEED)
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (PROJECT_ROOT / "flowers102_steps.py").exists():
    raise RuntimeError("Hãy mở đúng thư mục flower_project trong VS Code rồi chọn lại kernel .venv.")
os.chdir(PROJECT_ROOT)
BASE = Path(".")
DATA = Path("data")
RAW = DATA / "raw"
CACHE = Path("cache")
EXPORTS = Path("exports")
RUN_ID = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S_%f")
OUT = Path("outputs") / RUN_ID
FIG = OUT / "figures"
TABLE = OUT / "tables"
for folder in [RAW, CACHE, EXPORTS, FIG, TABLE, OUT / "models", OUT / "logs"]:
    folder.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

figure_records = []
def save_figure(fig, name, caption):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    figure_records[:] = [r for r in figure_records if r["file"] != name + ".png"]
    figure_records.append({"file": name + ".png", "caption": caption})
    pd.DataFrame(figure_records).to_csv(OUT / "figure_index.csv", index=False, encoding="utf-8-sig")
    plt.show()
    plt.close(fig)

def export_checkpoint(label="checkpoint"):
    """Nén minh chứng hiện có; không kèm dữ liệu ảnh/cache lớn."""
    archive = shutil.make_archive(str(EXPORTS / f"{RUN_ID}_{label}"), "zip", OUT)
    print("Đã tạo:", archive)
    return archive

versions = {pkg: metadata.version(pkg) for pkg in PACKAGES.values()}
save_json({"python": sys.version, "platform": platform.platform(), "versions": versions,
           "seed": SEED, "image_size": IMAGE_SIZE, "run_pca": RUN_PCA,
           "run_id": RUN_ID, "split_policy": "Oxford official; fit train only; select on val"}, OUT / "run_config.json")
(OUT / "requirements-lock.txt").write_text("\n".join(f"{p}=={v}" for p, v in versions.items()) + "\n", encoding="utf-8")
print("Thư mục kết quả:", OUT)
display(pd.DataFrame(versions.items(), columns=["Thư viện", "Phiên bản"]))

# %% [markdown]
# ## Bước 3 · Dùng dữ liệu đã tải, bổ sung nhãn và chia tập
# Cần 3 tệp: `102flowers.tgz`, `imagelabels.mat`, `setid.mat`.
# Đặt các tệp trong `data/raw/`. Ảnh đã có sẽ được dùng lại, không tải lại 329 MiB.
# `102segmentations.tgz` là ảnh đã phân đoạn, không thay thế file nhãn lớp;
# giữ làm dữ liệu mở rộng, chưa dùng trong thí nghiệm chính. Không cần khoảng cách tính sẵn.
# Downloader kiểm tra mã MD5 đối chiếu thông tin Torchvision và chỉ giải nén JPEG.
# Chỉ `imagelabels.mat` và `setid.mat` được tự tải nếu thiếu. Nếu chưa có archive ảnh,
# notebook dừng và hướng dẫn đặt đúng tệp, không âm thầm tải bộ ảnh lớn.
#
# `trnid` = train, `valid` = validation, `tstid` = test. Nhãn Oxford là 1..102;
# notebook giữ nguyên cách đánh số này. Không tự đảo train/test để tăng dữ liệu train.
# **Kết quả cần thấy:** 8.189 ảnh; train 1.020, validation 1.020, test 6.149.

# %%
FILES = {
    "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
    "setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
    "102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
}
DOWNLOAD_BASES = ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
                  "https://thor.robots.ox.ac.uk/flowers/102/"]

def file_hash(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def download_verified(filename, expected_md5):
    target = RAW / filename
    if target.exists():
        if file_hash(target) != expected_md5:
            raise ValueError(f"MD5 không khớp: {target}. Kiểm tra/tải lại đúng tệp trước khi tiếp tục.")
        print("Đã có, MD5 đúng:", target)
        return target
    if filename == "102flowers.tgz":
        raise FileNotFoundError("Chép 102flowers.tgz đã tải vào data/raw/ rồi chạy lại ô này.")
    errors = []
    for base_url in DOWNLOAD_BASES:
        url = base_url + filename
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) or None
                with open(partial, "wb") as stream, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
                            bar.update(len(block))
            if file_hash(partial) != expected_md5:
                raise ValueError("MD5 không khớp; tệp tải chưa đúng.")
            partial.replace(target)
            return target
        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Chưa tải được dữ liệu. Kiểm tra mạng rồi chạy lại ô này.\n" + "\n".join(errors))

for filename, md5 in FILES.items():
    download_verified(filename, md5)

labels = loadmat(RAW / "imagelabels.mat")["labels"].ravel().astype(int)
split_mat = loadmat(RAW / "setid.mat")
split_ids = {name: split_mat[key].ravel().astype(int)
             for name, key in {"train": "trnid", "val": "valid", "test": "tstid"}.items()}
assert len(labels) == 8189 and set(labels) == set(range(1, 103))
assert {k: len(v) for k, v in split_ids.items()} == {"train": 1020, "val": 1020, "test": 6149}
combined_ids = np.concatenate(list(split_ids.values()))
assert len(np.unique(combined_ids)) == len(combined_ids) == len(labels)
assert set(combined_ids) == set(range(1, len(labels) + 1))
expected_names = {f"image_{i:05d}.jpg" for i in combined_ids}
jpg = DATA / "jpg"
if not expected_names.issubset({p.name for p in jpg.glob("*.jpg")}):
    jpg.mkdir(exist_ok=True)
    with tarfile.open(RAW / "102flowers.tgz", "r:gz") as archive:
        members = [m for m in archive.getmembers()
                   if m.isfile() and re.fullmatch(r"(?:\./)?jpg/image_\d{5}\.jpg", m.name)]
        assert len(members) == len(labels), "Số JPEG trong archive không đúng."
        for member in tqdm(members, desc="Giải nén ảnh"):
            with archive.extractfile(member) as src, open(jpg / Path(member.name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)

df = pd.DataFrame({"image_id": np.arange(1, len(labels) + 1), "label": labels})
df["path"] = df.image_id.map(lambda i: str(jpg / f"image_{i:05d}.jpg"))
split_by_id = {int(i): name for name, ids in split_ids.items() for i in ids}
df["split"] = df.image_id.map(split_by_id)
assert df.path.map(lambda p: Path(p).exists()).all()
assert df.image_id.is_unique and not df.isna().any().any()
ALL_LABELS = np.arange(1, 103)
# Tên thông dụng theo thứ tự nhãn trong metadata Flowers102 của Torchvision.
CLASS_NAMES = """pink primrose|hard-leaved pocket orchid|canterbury bells|sweet pea|english marigold|tiger lily|moon orchid|bird of paradise|monkshood|globe thistle|snapdragon|colt's foot|king protea|spear thistle|yellow iris|globe-flower|purple coneflower|peruvian lily|balloon flower|giant white arum lily|fire lily|pincushion flower|fritillary|red ginger|grape hyacinth|corn poppy|prince of wales feathers|stemless gentian|artichoke|sweet william|carnation|garden phlox|love in the mist|mexican aster|alpine sea holly|ruby-lipped cattleya|cape flower|great masterwort|siam tulip|lenten rose|barbeton daisy|daffodil|sword lily|poinsettia|bolero deep blue|wallflower|marigold|buttercup|oxeye daisy|common dandelion|petunia|wild pansy|primula|sunflower|pelargonium|bishop of llandaff|gaura|geranium|orange dahlia|pink-yellow dahlia?|cautleya spicata|japanese anemone|black-eyed susan|silverbush|californian poppy|osteospermum|spring crocus|bearded iris|windflower|tree poppy|gazania|azalea|water lily|rose|thorn apple|morning glory|passion flower|lotus|toad lily|anthurium|frangipani|clematis|hibiscus|columbine|desert-rose|tree mallow|magnolia|cyclamen|watercress|canna lily|hippeastrum|bee balm|ball moss|foxglove|bougainvillea|camellia|mallow|mexican petunia|bromelia|blanket flower|trumpet creeper|blackberry lily""".split("|")
assert len(CLASS_NAMES) == 102
name_of = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
df["class_name"] = df.label.map(name_of)
df.drop(columns="path").to_csv(TABLE / "split_manifest.csv", index=False)
save_json(name_of, OUT / "class_names.json")
save_json(FILES, OUT / "source_checksums.json")
print(f"Số ảnh: {len(df):,}; số lớp: {df.label.nunique()}")
display(df.groupby("split").agg(so_anh=("image_id", "count"), so_lop=("label", "nunique")))
display(df.head())

# %% [markdown]
# ## Bước 4 · Kiểm tra chất lượng dữ liệu
# Đọc tất cả ảnh để kiểm tra ảnh hỏng, kích thước, số kênh; kiểm tra SHA-256 của
# nội dung RGB đã giải mã để phát hiện ảnh trùng hoàn toàn. Cách này không phát
# hiện mọi ảnh gần giống (khác crop/góc chụp); ghi rõ hạn chế đó trong báo cáo.
# Ảnh hỏng, nhãn xung đột hoặc trùng train-validation làm notebook dừng để kiểm tra.
# Với trùng train/validation-test và trùng nội bộ test: giữ nguyên dữ liệu/chia tập gốc,
# đồng thời tạo danh sách test bổ sung không trùng để đo lại cùng dự đoán ở Bước 11.
# Quy tắc này chỉ dựa vào nội dung ảnh, không dựa vào nhãn hay kết quả mô hình.
# **Minh chứng:** bảng kiểm tra và bảng số ảnh mỗi split. Đến đây hãy kiểm tra cùng người hướng dẫn.

# %%
audit_rows = []
for row in tqdm(df.itertuples(index=False), total=len(df), desc="Kiểm tra ảnh"):
    rec = {"image_id": row.image_id, "error": "", "width": None, "height": None,
           "channels": None, "mode": None, "rgb_sha256": None}
    try:
        with Image.open(row.path) as im:
            im.load()
            rgb = np.asarray(im.convert("RGB"))
            digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            rec.update(width=im.width, height=im.height, channels=len(im.getbands()),
                       mode=im.mode, rgb_sha256=digest)
    except Exception as exc:
        rec["error"] = str(exc)
    audit_rows.append(rec)
audit = pd.DataFrame(audit_rows).merge(df[["image_id", "split", "label"]], on="image_id")
audit.to_csv(TABLE / "image_audit.csv", index=False)
bad = audit[audit.error != ""]
duplicates = audit[audit.rgb_sha256.duplicated(keep=False) & audit.rgb_sha256.notna()]
duplicates.to_csv(TABLE / "exact_duplicate_images.csv", index=False)
cross_split = duplicates.groupby("rgb_sha256").split.nunique()
summary = {"total_images": len(df), "classes": df.label.nunique(),
           "missing_labels": int(df.label.isna().sum()), "broken_images": len(bad),
           "exact_duplicate_rows": len(duplicates),
           "cross_split_duplicate_groups": int((cross_split > 1).sum()),
           "conflicting_duplicate_groups": int((duplicates.groupby("rgb_sha256").label.nunique() > 1).sum())}
save_json(summary, OUT / "audit_summary.json")
display(pd.DataFrame(summary.items(), columns=["Kiểm tra", "Kết quả"]))
display(audit[["width", "height", "channels"]].describe())
assert bad.empty, "Có ảnh hỏng: kiểm tra image_audit.csv và tải/giải nén lại trước khi tiếp tục."
assert summary["conflicting_duplicate_groups"] == 0, "Có ảnh trùng nhưng khác nhãn: cần kiểm tra trước khi chạy."
train_hashes = set(audit.loc[audit.split == "train", "rgb_sha256"])
val_hashes = set(audit.loc[audit.split == "val", "rgb_sha256"])
assert not train_hashes.intersection(val_hashes), "Có ảnh trùng train-validation: cần xử lý trước khi chọn tham số."
test_audit = audit[audit.split == "test"].sort_values("image_id").copy()
seen_during_development = train_hashes | val_hashes
test_audit["exclude_reason"] = ""
overlap = test_audit.rgb_sha256.isin(seen_during_development)
test_audit.loc[overlap, "exclude_reason"] = "duplicate_of_train_or_validation"
repeat_in_test = (~overlap) & test_audit.rgb_sha256.duplicated(keep="first")
test_audit.loc[repeat_in_test, "exclude_reason"] = "duplicate_within_test"
clean_test_ids = test_audit.loc[test_audit.exclude_reason == "", "image_id"].to_numpy()
excluded_test = test_audit[test_audit.exclude_reason != ""]
excluded_test.to_csv(TABLE / "test_duplicate_exclusions.csv", index=False)
assert set(test_audit.loc[test_audit.exclude_reason == "", "label"]) == set(ALL_LABELS)
evaluation_policy = {"official_test_images": len(test_audit), "unique_unseen_test_images": len(clean_test_ids),
    "excluded_test_ids": excluded_test.image_id.tolist(), "train_val_overlap_groups": 0,
    "policy": "Keep Oxford split unchanged; additionally score test after removing decoded-RGB duplicates of train/val and retaining the lowest image_id within each remaining test duplicate group.",
    "chosen_using_model_predictions": False}
save_json(evaluation_policy, OUT / "evaluation_policy.json")
print("Test Oxford gốc:", len(test_audit), "| Test không trùng (đánh giá bổ sung):", len(clean_test_ids))
if not excluded_test.empty:
    display(excluded_test[["image_id", "split", "label", "exclude_reason"]])
if not duplicates.empty:
    groups = list(duplicates.groupby("rgb_sha256"))
    fig, axes = plt.subplots(len(groups), 2, figsize=(9, 3.5 * len(groups)), squeeze=False,
                             constrained_layout=True)
    for axes_row, (_, group) in zip(axes, groups):
        for ax, duplicate_row in zip(axes_row, group.sort_values("image_id").head(2).itertuples()):
            path = df.loc[df.image_id == duplicate_row.image_id, "path"].iloc[0]
            with Image.open(path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"ID {duplicate_row.image_id} | {duplicate_row.split} | lớp {duplicate_row.label}")
            ax.axis("off")
    save_figure(fig, "00_exact_duplicates", "Các nhóm ảnh có RGB trùng hoàn toàn; chia tập gốc được giữ nguyên và có đánh giá test bổ sung loại trùng.")
print("ĐIỂM DỪNG 1: đã kiểm tra nguồn, nhãn, split và chất lượng ảnh.")
# Muốn kết thúc buổi tại đây: bỏ dấu # ở dòng kế tiếp để tạo ZIP minh chứng hiện có.
# export_checkpoint("buoi_1")

# %% [markdown]
# ## Bước 5 · EDA và ảnh mẫu của đủ 102 lớp
# Phân phối nhãn toàn bộ dữ liệu dùng để mô tả dataset, không dùng chọn siêu tham số.
# Mọi ảnh mẫu và minh họa tiền xử lý lấy từ train. Bộ train chính thức có 10 ảnh/lớp,
# đây là hạn chế đáng kể. Không kỳ vọng K-NN/HOG đạt mức của mô hình học sâu.
# **Minh chứng:** phân phối nhãn, kích thước ảnh, chia tập và 6 trang ảnh mẫu.

# %%
counts = pd.crosstab(df.label, df.split).reindex(ALL_LABELS, fill_value=0)
counts["total"] = counts.sum(axis=1)
counts.to_csv(TABLE / "class_counts.csv")
assert (counts["train"] == 10).all() and (counts["val"] == 10).all()
fig, axes = plt.subplots(2, 1, figsize=(16, 7), constrained_layout=True)
axes[0].bar(ALL_LABELS, counts.total, color="#317d9c")
axes[0].set(title=f"Phân phối {len(df):,} ảnh theo {len(ALL_LABELS)} lớp", xlabel="Mã lớp", ylabel="Số ảnh")
axes[1].bar(ALL_LABELS, counts.train, label="Train", color="#25786b")
axes[1].bar(ALL_LABELS, counts.val, bottom=counts.train, label="Validation", color="#e6ae45")
axes[1].bar(ALL_LABELS, counts.test, bottom=counts.train + counts.val, label="Test", color="#859db9")
axes[1].set(xlabel="Mã lớp", ylabel="Số ảnh", title="Giữ nguyên chia tập Oxford")
axes[1].legend(ncols=3)
save_figure(fig, "01_class_distribution", "Phân phối nhãn toàn bộ dataset và theo chia tập chính thức.")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].scatter(audit.width, audit.height, s=5, alpha=0.2)
axes[0].set(xlabel="Chiều rộng (pixel)", ylabel="Chiều cao (pixel)", title="Kích thước ảnh gốc")
audit["mode"].value_counts().plot.bar(ax=axes[1], color="#25786b", rot=0)
axes[1].set(title="Chế độ màu ảnh gốc", ylabel="Số ảnh")
save_figure(fig, "02_image_properties", "Kích thước và chế độ màu trước tiền xử lý.")
sample_pages = (len(ALL_LABELS) + 17) // 18
for page, start in enumerate(range(1, len(ALL_LABELS) + 1, 18), 1):
    class_ids = list(range(start, min(start + 18, len(ALL_LABELS) + 1)))
    fig, axes = plt.subplots(3, 6, figsize=(16, 8), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, label in zip(axes.ravel(), class_ids):
        sample = df[(df.split == "train") & (df.label == label)].iloc[0]
        with Image.open(sample.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{label:03d}: {name_of[label]}", fontsize=8, wrap=True)
    fig.suptitle(f"Ảnh train đại diện từng lớp · trang {page}/{sample_pages}")
    save_figure(fig, f"03_samples_page_{page}", f"Ảnh train của các lớp {class_ids[0]}-{class_ids[-1]}.")

# %% [markdown]
# ## Bước 6 · Tiền xử lý và đặc trưng
# Resize tất cả biểu diễn về cùng 64×64 để so sánh nhất quán; đổi RGB và chia 255.
# Resize trực tiếp có thể làm méo tỷ lệ ảnh, được ghi nhận là hạn chế.
# - **Pixel:** 64×64×3 = 12.288 chiều, giữ màu và vị trí pixel.
# - **HOG:** ảnh xám, 9 hướng, cell 8×8, block 2×2 → 1.764 chiều; mô tả biên/hình dạng.
# - **HSV:** 16 bin mỗi kênh H/S/V → 48 chiều; histogram màu không lưu vị trí.
# - **HOG+HSV:** chuẩn hóa L2 từng nhóm, ghép với trọng số bằng nhau → 1.812 chiều.
# Không chuyển toàn bộ bài toán sang ảnh xám vì màu có thể giúp phân loại hoa.
# Đây là các phép biến đổi từng ảnh với cấu hình cố định, không học từ test.
# Không dùng augmentation ở thí nghiệm chính để giữ số mẫu so sánh nhất quán.
# **Minh chứng:** ảnh gốc, RGB resize, grayscale, HOG và histogram HSV.

# %%
def read_rgb(path, size=IMAGE_SIZE):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

HOG_CONFIG = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
save_json({"image_size": IMAGE_SIZE, "resize": "bilinear", "pixel_range": [0, 1],
           "hog": HOG_CONFIG, "hog_global_norm": "L2", "hsv_bins_per_channel": 16,
           "color_hist_norm": "L2", "combined_weighting": "equal L2 groups / sqrt(2)"}, OUT / "feature_config.json")
def l2(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)

def extract_features(path):
    rgb = read_rgb(path)
    h = l2(hog(rgb2gray(rgb), **HOG_CONFIG).astype(np.float32))
    hsv = rgb2hsv(rgb)
    color_hist = np.concatenate([np.histogram(hsv[:, :, c], bins=16, range=(0, 1))[0]
                                 for c in range(3)]).astype(np.float32)
    color_hist = l2(color_hist)
    return {"pixels": rgb.ravel(), "hog": h,
            "hog_hsv": np.concatenate([h, color_hist]).astype(np.float32) / np.sqrt(np.float32(2))}

sample_path = df[df.split == "train"].iloc[0].path
rgb = read_rgb(sample_path)
gray = rgb2gray(rgb)
_, hog_image = hog(gray, visualize=True, **HOG_CONFIG)
fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
with Image.open(sample_path) as im:
    axes[0].imshow(im.convert("RGB"))
axes[1].imshow(rgb)
axes[2].imshow(gray, cmap="gray")
axes[3].imshow(rescale_intensity(hog_image, out_range=(0, 1)), cmap="gray")
for ax, title in zip(axes[:4], ["Ảnh gốc", "RGB 64×64 / 255", "Grayscale cho HOG", "HOG: hướng biên"]):
    ax.set_title(title)
    ax.axis("off")
hsv = rgb2hsv(rgb)
for c, name in enumerate(["H", "S", "V"]):
    axes[4].hist(hsv[:, :, c].ravel(), bins=16, range=(0, 1), histtype="step", label=name)
axes[4].set_title("Histogram màu HSV")
axes[4].legend()
save_figure(fig, "04_preprocessing_hog_hsv", "Các biểu diễn của cùng một ảnh train.")

# %% [markdown]
# ## Bước 7 · Trích đặc trưng train và validation
# Chưa trích test ở bước này. Cache gồm ID ảnh và dấu vân tay cấu hình/nội dung;
# tránh nhầm thứ tự nhãn hoặc tái sử dụng đặc trưng cũ khi đổi tiền xử lý.
# **Cần hiểu:** hàng i của X phải khớp ảnh và nhãn y[i].

# %%
parts = {name: df[df.split == name].reset_index(drop=True) for name in ["train", "val", "test"]}
y = {name: part.label.to_numpy() for name, part in parts.items()}
FEATURE_VERSION = "rgb-bilinear64_hog9-8-2-sqrt-l2_hsv16-l2_equal-v1"
def feature_matrix(split):
    part = parts[split]
    audit_hash = audit.set_index("image_id").loc[part.image_id, "rgb_sha256"].tolist()
    signature = hashlib.sha256(json.dumps({"version": FEATURE_VERSION, "size": IMAGE_SIZE,
        "hog_config": HOG_CONFIG, "hsv_bins_per_channel": 16,
        "ids": part.image_id.tolist(), "sha256": audit_hash,
        "libraries": {p: versions[p] for p in ["numpy", "scikit-image", "Pillow"]}}).encode()).hexdigest()[:16]
    cache_file = CACHE / f"{split}_{signature}.npz"
    start = time.perf_counter()
    if cache_file.exists():
        with np.load(cache_file) as archive:
            assert np.array_equal(archive["image_ids"], part.image_id.to_numpy())
            matrices = {key: archive[key] for key in ["pixels", "hog", "hog_hsv"]}
        cache_hit = True
    else:
        first = extract_features(part.iloc[0].path)
        matrices = {key: np.empty((len(part), len(value)), dtype=np.float32) for key, value in first.items()}
        for i, path in enumerate(tqdm(part.path, desc=f"Đặc trưng {split}")):
            values = first if i == 0 else extract_features(path)
            for key in matrices:
                matrices[key][i] = values[key]
        np.savez_compressed(cache_file, image_ids=part.image_id.to_numpy(), **matrices)
        cache_hit = False
    assert all(np.isfinite(a).all() and len(a) == len(part) for a in matrices.values())
    save_json({"split": split, "seconds": time.perf_counter() - start,
               "cache_hit": cache_hit, "signature": signature,
               "shapes": {k: list(v.shape) for k, v in matrices.items()}}, OUT / "logs" / f"features_{split}.json")
    return matrices

X = {"train": feature_matrix("train"), "val": feature_matrix("val")}
display(pd.DataFrame([{"split": s, "feature": key, "images": value.shape[0], "dimensions": value.shape[1]}
                      for s in X for key, value in X[s].items()]))

# %% [markdown]
# ## Bước 8 · Baseline và thiết kế so sánh
# Baseline bắt buộc: **pixel + K-NN, k=5, Euclidean, uniform**.
# Thêm DummyClassifier đo mốc không học đặc trưng, nhưng không thay pixel baseline.
# Cùng k/metric/weights giữa pixel và HOG giúp tách ảnh hưởng của biểu diễn.
# Cùng max_depth=20 giữa hai cây giúp so sánh đặc trưng với thuật toán khác.
# Metric chọn mô hình: **F1-macro**, coi các lớp có trọng số bằng nhau.
# Precision/Recall/F1 dùng `zero_division=0` khi mô hình không dự đoán được một lớp.
# **Minh chứng:** bảng validation của 5 cấu hình cố định và mốc Dummy.

# %%
def score_metrics(truth, pred):
    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=ALL_LABELS, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(truth, pred, labels=ALL_LABELS, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(truth, pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f), "f1_weighted": float(fw)}

models, model_features, val_rows = {}, {}, []
def fit_validate(name, feature, estimator):
    start = time.perf_counter()
    estimator.fit(X["train"][feature], y["train"])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = estimator.predict(X["val"][feature])
    predict_seconds = time.perf_counter() - start
    row = {"name": name, "feature": feature, **score_metrics(y["val"], pred),
           "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
           "parameters": str(estimator.get_params())}
    models[name], model_features[name] = estimator, feature
    val_rows[:] = [r for r in val_rows if r["name"] != name]
    val_rows.append(row)
    pd.DataFrame(val_rows).to_csv(TABLE / "validation_results.csv", index=False)
    print(name, "| validation F1-macro:", round(row["f1_macro"], 4))
    return row

knn_fixed = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform", algorithm="brute", n_jobs=2)
tree_fixed = DecisionTreeClassifier(max_depth=20, criterion="gini", random_state=SEED)
fit_validate("M0_Dummy", "pixels", DummyClassifier(strategy="most_frequent"))
fit_validate("M1_Pixel_KNN_baseline", "pixels", clone(knn_fixed))
fit_validate("M2_Pixel_Tree_fixed", "pixels", clone(tree_fixed))
fit_validate("M3_HOG_KNN_fixed", "hog", clone(knn_fixed))
fit_validate("M4_HOG_Tree_fixed", "hog", clone(tree_fixed))
fit_validate("M5_HOG_HSV_KNN_fixed", "hog_hsv", clone(knn_fixed))
display(pd.DataFrame(val_rows).drop(columns="parameters").round(4))

# %% [markdown]
# ## Bước 9 · Thử tham số chỉ trên validation
# Thử K-NN: k ∈ {3,5,9}, khoảng cách {Euclidean,Manhattan}, weights=uniform.
# Thử Cây quyết định: max_depth ∈ {10,20,None}, criterion ∈ {gini,entropy}.
# Tối ưu riêng HOG+KNN, HOG+Tree, HOG+HSV+KNN để có so sánh hai thuật toán
# và tác động của màu. Chọn F1-macro cao nhất; nếu bằng nhau giữ cấu hình đầu.
# Đây là validation holdout, **không phải** cross-validation. Grid nhỏ giúp tiết kiệm CPU.
# Không điều chỉnh grid dựa trên test. Kết quả validation có thể lạc quan sau nhiều thử nghiệm.

# %%
tuning_rows = []
def tune(name, feature, base, grid):
    best_score, best_params = -1.0, None
    for params in tqdm(list(ParameterGrid(grid)), desc=name):
        candidate = clone(base).set_params(**params)
        start = time.perf_counter()
        candidate.fit(X["train"][feature], y["train"])
        pred = candidate.predict(X["val"][feature])
        row = {"experiment": name, "feature": feature, "params": json.dumps(params),
               **params, **score_metrics(y["val"], pred), "fit_and_predict_seconds": time.perf_counter() - start}
        tuning_rows.append(row)
        if row["f1_macro"] > best_score:
            best_score, best_params = row["f1_macro"], params
    print("Chọn", name, best_params)
    fit_validate(name, feature, clone(base).set_params(**best_params))

knn_grid = {"n_neighbors": [3, 5, 9], "metric": ["euclidean", "manhattan"]}
tree_grid = {"max_depth": [10, 20, None], "criterion": ["gini", "entropy"]}
tune("M6_HOG_KNN_tuned", "hog", knn_fixed, knn_grid)
tune("M7_HOG_Tree_tuned", "hog", tree_fixed, tree_grid)
tune("M8_HOG_HSV_KNN_tuned", "hog_hsv", knn_fixed, knn_grid)
pd.DataFrame(tuning_rows).to_csv(TABLE / "hyperparameter_trials.csv", index=False)
trials = pd.DataFrame(tuning_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, experiment in zip(axes, ["M6_HOG_KNN_tuned", "M8_HOG_HSV_KNN_tuned"]):
    subset = trials[trials.experiment == experiment]
    for metric, group in subset.groupby("metric"):
        group = group.sort_values("n_neighbors")
        ax.plot(group.n_neighbors, group.f1_macro, "o-", label=metric)
    ax.set(title=experiment, xlabel="k", ylabel="Validation F1-macro", xticks=[3, 5, 9])
    ax.legend()
save_figure(fig, "05_knn_tuning", "Ảnh hưởng của k và khoảng cách, chỉ đo trên validation.")
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
tree_trials = trials[trials.experiment == "M7_HOG_Tree_tuned"]
tick_labels = [f"depth={json.loads(p)['max_depth']}\n{json.loads(p)['criterion']}" for p in tree_trials.params]
ax.bar(tick_labels, tree_trials.f1_macro, color="#25786b")
ax.set(ylabel="Validation F1-macro", title="Ảnh hưởng độ sâu và tiêu chí chia của cây")
save_figure(fig, "06_tree_tuning", "Thử tham số Cây quyết định trên validation.")

# %% [markdown]
# ## Bước 10 · PCA (bổ sung) và chốt mô hình
# Pipeline StandardScaler → PCA(128) → K-NN chỉ được fit trên train.
# Dùng cùng cấu hình K-NN của M8 cho cả trước/sau PCA để đánh giá tác động của
# toàn bộ bước scale+PCA. Không kết luận rằng mọi thay đổi đều do PCA đơn lẻ.
# So sánh số chiều, thời gian fit/predict và F1. Hình phương sai PCA chỉ là chẩn đoán;
# không dùng nó để thay 128 sau khi xem test. Không gộp validation vào train trong notebook này.
# **Minh chứng:** bảng validation, biểu đồ so sánh, cấu hình được chốt.

# %%
if RUN_PCA:
    pca_components = min(128, len(y["train"]) - 1, X["train"]["hog_hsv"].shape[1])
    pipeline = Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components=pca_components, svd_solver="randomized", random_state=SEED)),
                         ("knn", clone(models["M8_HOG_HSV_KNN_tuned"]))])
    fit_validate("M9_HOG_HSV_PCA_KNN", "hog_hsv", pipeline)
    explained = pipeline.named_steps["pca"].explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(np.arange(1, len(explained) + 1), explained)
    ax.set(xlabel="Số thành phần PCA", ylabel="Tỷ lệ phương sai tích lũy", title="PCA fit trên train")
    save_figure(fig, "07_pca_variance", "Phương sai giữ lại của PCA; không đồng nghĩa độ chính xác phân loại.")

validation = pd.DataFrame(val_rows)
eligible = validation[validation.name != "M0_Dummy"]
winner = eligible.sort_values("f1_macro", ascending=False, kind="stable").iloc[0]["name"]
save_json({"selected_name": winner, "criterion": "validation f1_macro; stable first tie",
           "test_used_for_selection": False, "refit_train_val": False,
           "models": {name: {"feature": model_features[name], "params": model.get_params()}
                      for name, model in models.items()}}, OUT / "frozen_selection.json")
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
validation.set_index("name")[["accuracy", "f1_macro"]].plot.barh(ax=ax, color=["#8ca6c3", "#25786b"])
ax.set(xlabel="Điểm validation (0-1)", ylabel="", title="Chọn mô hình bằng F1-macro trên validation")
save_figure(fig, "08_validation_comparison", "So sánh validation trước khi mở tập test.")
display(validation.drop(columns="parameters").round(4))
print("Mô hình đã chốt:", winner)
print("ĐIỂM DỪNG 2: ghi lý do chọn; từ đây không thay grid theo kết quả test.")
# export_checkpoint("truoc_test")

# %% [markdown]
# ## Bước 11 · Đánh giá test sau khi chốt
# Mọi cấu hình cố định/đã chọn bằng validation được đo trên cùng 6.149 ảnh test.
# Mô hình dùng demo vẫn là mô hình chốt ở Bước 10, kể cả khi mô hình khác có test F1 cao hơn.
# Bảng chính giữ thứ tự thí nghiệm, không chọn lại theo test. Lưu dự đoán từng ảnh
# để kiểm tra kết quả; confusion matrix và classification report có cho mọi mô hình.
# Vì dataset có ảnh trùng, lưu thêm `test_results_unique_unseen.csv` trên các ID
# đã xác định từ audit (6.145 ảnh với dữ liệu hiện tại). Không fit lại hay chọn lại
# mô hình; báo cáo cả hai bảng và không coi test gốc hoàn toàn độc lập với train/val.
# **Minh chứng:** test_results.csv, bảng so sánh và biểu đồ metric/thời gian.

# %%
assert (OUT / "frozen_selection.json").exists()
X["test"] = feature_matrix("test")
test_rows, clean_test_rows, predictions = [], [], {}
clean_test_mask = np.isin(parts["test"].image_id.to_numpy(), clean_test_ids)
assert clean_test_mask.sum() == len(clean_test_ids)
for name, model in tqdm(models.items(), desc="Đánh giá test (không fit lại)"):
    feature = model_features[name]
    start = time.perf_counter()
    pred = model.predict(X["test"][feature])
    seconds = time.perf_counter() - start
    predictions[name] = pred
    test_rows.append({"name": name, "feature": feature, **score_metrics(y["test"], pred),
                      "predict_seconds": seconds, "selected_on_validation": name == winner})
    clean_test_rows.append({"name": name, "feature": feature,
        **score_metrics(y["test"][clean_test_mask], pred[clean_test_mask]),
        "evaluated_images": int(clean_test_mask.sum()), "selected_on_validation": name == winner})
    details = parts["test"][["image_id", "label", "class_name"]].copy()
    details["predicted_label"] = pred
    details["predicted_name"] = details.predicted_label.map(name_of)
    details["correct"] = details.label == details.predicted_label
    details.to_csv(TABLE / f"predictions_{name}.csv", index=False)
    cm = confusion_matrix(y["test"], pred, labels=ALL_LABELS)
    pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS).to_csv(TABLE / f"confusion_{name}.csv")
    report = classification_report(y["test"], pred, labels=ALL_LABELS,
        target_names=[f"{c:03d} {name_of[c]}" for c in ALL_LABELS], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(TABLE / f"classification_report_{name}.csv")
    clean_cm = confusion_matrix(y["test"][clean_test_mask], pred[clean_test_mask], labels=ALL_LABELS)
    pd.DataFrame(clean_cm, index=ALL_LABELS, columns=ALL_LABELS).to_csv(TABLE / f"confusion_unique_unseen_{name}.csv")

test_results = pd.DataFrame(test_rows)
test_results.to_csv(TABLE / "test_results.csv", index=False)
clean_test_results = pd.DataFrame(clean_test_rows)
clean_test_results.to_csv(TABLE / "test_results_unique_unseen.csv", index=False)
print("Kết quả test Oxford gốc (có ảnh trùng):")
display(test_results.round(4))
print("Kiểm tra bổ sung trên test không trùng train/val và không trùng nội bộ:")
display(clean_test_results.round(4))
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
test_results.set_index("name")[["accuracy", "f1_macro", "f1_weighted"]].plot.barh(
    ax=axes[0], color=["#8ca6c3", "#25786b", "#e6ae45"])
axes[0].set(title="Kết quả test của các cấu hình đã chốt", xlabel="Điểm (0-1)", ylabel="")
test_results.set_index("name")["predict_seconds"].plot.barh(ax=axes[1], color="#859db9")
axes[1].set(title="Thời gian dự đoán cùng tập test", xlabel="Giây (không gồm trích đặc trưng)", ylabel="")
save_figure(fig, "09_test_comparison", "So sánh test; mô hình cuối được chọn từ validation, không từ biểu đồ này.")

# %% [markdown]
# ## Bước 12 · Confusion matrix và phân tích lỗi
# Ma trận đầy đủ có 102×102 ô. Ma trận chuẩn hóa theo hàng giúp đọc recall của
# từng lớp. Biểu đồ con lấy các lớp có nhiều lỗi nhầm qua lại nhất để dễ nhìn;
# đây chỉ là phần của ma trận đầy đủ, không phải một bài toán phân loại mới.
# Ảnh sai được lấy ngẫu nhiên với seed cố định, tránh chỉ chọn ảnh có lợi cho mô hình.
# Ghi nhận điều quan sát được (màu, nền, góc nhìn, kích thước hoa), phân biệt với
# giả thuyết nguyên nhân. Nếu sửa pipeline sau bước này, cần một lần đánh giá độc lập mới.
# **Minh chứng:** ma trận toàn bộ/chi tiết, lớp khó, cặp nhầm và ảnh đúng/sai.

# %%
pred = predictions[winner]
cm = confusion_matrix(y["test"], pred, labels=ALL_LABELS)
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ticks = np.arange(0, len(ALL_LABELS), 10)
ax.set(xticks=ticks, yticks=ticks, xticklabels=ALL_LABELS[ticks], yticklabels=ALL_LABELS[ticks],
       xlabel="Lớp dự đoán", ylabel="Lớp thật", title=f"Confusion matrix chuẩn hóa theo hàng · {winner}")
fig.colorbar(im, ax=ax, label="Tỷ lệ trên số ảnh của lớp thật")
save_figure(fig, "10_confusion_full", "Confusion matrix 102 lớp của mô hình chọn trên validation.")

offdiag = cm.copy()
np.fill_diagonal(offdiag, 0)
pair_rows = []
for flat in np.argsort(offdiag.ravel())[::-1]:
    a, b = np.unravel_index(flat, offdiag.shape)
    if offdiag[a, b] <= 0 or len(pair_rows) >= 15:
        break
    pair_rows.append({"true_label": int(ALL_LABELS[a]), "predicted_label": int(ALL_LABELS[b]),
        "true_name": name_of[int(ALL_LABELS[a])], "predicted_name": name_of[int(ALL_LABELS[b])],
        "count": int(offdiag[a, b]), "share_of_true_class": float(cm_norm[a, b])})
pairs = pd.DataFrame(pair_rows, columns=["true_label", "predicted_label", "true_name", "predicted_name", "count", "share_of_true_class"])
pairs.to_csv(TABLE / "top_confused_pairs.csv", index=False)
display(pairs.head(10))
focus = list(dict.fromkeys([v for r in pair_rows for v in [r["true_label"], r["predicted_label"]]]))[:10]
if not focus:
    focus = ALL_LABELS[:10].tolist()
idx = np.array(focus) - 1
fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
sub = cm_norm[np.ix_(idx, idx)]
im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
tick_names = [f"{c}: {name_of[c]}" for c in focus]
ax.set(xticks=range(len(focus)), yticks=range(len(focus)), xticklabels=tick_names,
       yticklabels=tick_names, xlabel="Lớp dự đoán", ylabel="Lớp thật", title="Chi tiết các lớp thường nhầm (trích ma trận 102 lớp)")
plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
for a in range(len(focus)):
    for b in range(len(focus)):
        ax.text(b, a, f"{sub[a, b]:.2f}", ha="center", va="center",
                color="white" if sub[a, b] > .5 else "black", fontsize=8)
fig.colorbar(im, ax=ax)
save_figure(fig, "11_confusion_detail", "Ma trận con; mẫu số vẫn là toàn bộ số ảnh của lớp thật, hàng có thể không cộng thành 1.")

p, r, f, support = precision_recall_fscore_support(y["test"], pred, labels=ALL_LABELS, zero_division=0)
per_class = pd.DataFrame({"label": ALL_LABELS, "name": CLASS_NAMES, "precision": p, "recall": r, "f1": f, "support": support})
per_class.to_csv(TABLE / "selected_model_per_class.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
hard = per_class.sort_values(["f1", "label"]).head(15).iloc[::-1]
ax.barh(hard.label.astype(str) + ": " + hard.name, hard.f1, color="#be6e50")
ax.set(xlim=(0, 1), xlabel="F1 của lớp", title="15 lớp F1 thấp nhất trên test")
save_figure(fig, "12_hardest_classes", "Các lớp khó theo F1, cần đọc kèm support trong bảng per-class.")

for correct, filename in [(False, "13_wrong_predictions"), (True, "14_correct_predictions")]:
    candidates = np.flatnonzero((pred == y["test"]) == correct)
    chosen = np.random.default_rng(SEED).choice(candidates, min(12, len(candidates)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(14, 11), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), chosen):
        with Image.open(parts["test"].iloc[i].path) as im:
            ax.imshow(im.convert("RGB"))
        actual, guessed = int(y["test"][i]), int(pred[i])
        ax.set_title(f"ID {parts['test'].iloc[i].image_id}\nThật: {actual} {name_of[actual]}\nĐoán: {guessed} {name_of[guessed]}", fontsize=8)
    fig.suptitle(f"{winner} · ảnh dự đoán {'đúng' if correct else 'sai'} · chọn ngẫu nhiên, seed=42")
    if not len(chosen):
        axes.ravel()[0].text(.1, .5, "Không có mẫu trong nhóm này.")
    save_figure(fig, filename, "Ảnh test minh họa dự đoán " + ("đúng." if correct else "sai; cần viết nhận xét quan sát."))
errors_for_notes = parts["test"].loc[pred != y["test"], ["image_id", "label", "class_name"]].copy()
errors_for_notes["predicted_label"] = pred[pred != y["test"]]
errors_for_notes["predicted_name"] = errors_for_notes.predicted_label.map(name_of)
errors_for_notes["observed_image_characteristics"] = ""
errors_for_notes["hypothesis_not_proven"] = ""
errors_for_notes.to_csv(TABLE / "error_analysis_to_complete.csv", index=False)

# %% [markdown]
# ## Bước 13 · Lưu mô hình và demo trong notebook
# Demo chỉ phân loại trong 102 lớp đã học, không có lớp "không phải hoa".
# Điểm predict_proba của K-NN/cây chưa được hiệu chỉnh; không diễn giải là xác suất
# chắc chắn đúng ngoài thực tế. Demo bên dưới dùng một ảnh test, không phải đánh giá mới.
# Ô kế tiếp cho phép đổi số thứ tự ảnh test và đối chiếu nhãn thật với dự đoán.
# Ô tùy chọn cuối cùng cho phép chọn một ảnh trong thư mục dự án và lưu hình dự đoán.
# Đây là demo notebook; muốn xin điểm GUI cần hoàn thiện giao diện tương tác riêng.

# %%
bundle = {"model": models[winner], "name": winner, "feature": model_features[winner],
          "class_names": name_of, "image_size": IMAGE_SIZE, "feature_version": FEATURE_VERSION,
          "hog_config": HOG_CONFIG, "versions": versions, "seed": SEED}
joblib.dump(bundle, OUT / "models" / "selected_model.joblib")
restored = joblib.load(OUT / "models" / "selected_model.joblib")
assert np.array_equal(restored["model"].predict(X["test"][restored["feature"]][:5]), predictions[winner][:5])

def demo_image(path, figure_name="15_demo", true_label=None):
    vector = extract_features(path)[restored["feature"]][None, :]
    predicted_label = int(restored["model"].predict(vector)[0])
    probabilities = restored["model"].predict_proba(vector)[0]
    # Giữ thứ tự lớp khi bằng điểm, phù hợp quy tắc argmax của K-NN/cây.
    ranking = np.argsort(-probabilities, kind="stable")[:5]
    class_ids = restored["model"].classes_[ranking].astype(int)
    result = pd.DataFrame({"label": class_ids, "flower": [name_of[c] for c in class_ids],
                           "model_score": probabilities[ranking]})
    result.attrs["predicted_label"] = predicted_label
    print("Ảnh:", Path(path).as_posix())
    print("Mô hình:", restored["name"])
    print(f"Dự đoán: {predicted_label} — {name_of[predicted_label]}")
    title = f"Dự đoán: {name_of[predicted_label]} (lớp {predicted_label})"
    if true_label is not None:
        true_label = int(true_label)
        outcome = "ĐÚNG" if predicted_label == true_label else "SAI"
        print(f"Nhãn thật: {true_label} — {name_of[true_label]} | Kết quả: {outcome}")
        title += f"\nNhãn thật: {name_of[true_label]} (lớp {true_label}) · {outcome}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    with Image.open(path) as im:
        axes[0].imshow(ImageOps.exif_transpose(im).convert("RGB"))
    axes[0].axis("off")
    axes[0].set_title(title, fontsize=10)
    axes[1].barh(result.flower.iloc[::-1], result.model_score.iloc[::-1], color="#25786b")
    axes[1].set(xlim=(0, 1), title="Top 5 điểm mô hình (chưa hiệu chỉnh)", xlabel="Điểm")
    save_figure(fig, figure_name, "Demo mô hình đã lưu; điểm dự đoán không phải độ tin cậy đã hiệu chỉnh.")
    display(result)
    return result

def demo_test_image(index=0):
    """Chọn ảnh test theo vị trí; nhãn thật chỉ dùng để đối chiếu sau dự đoán."""
    pool = parts["test"].loc[parts["test"].image_id.isin(clean_test_ids)].reset_index(drop=True)
    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(pool):
        raise ValueError(f"TEST_INDEX phải là số nguyên từ 0 đến {len(pool) - 1}.")
    sample = pool.iloc[index]
    image_id, actual = int(sample.image_id), int(sample.label)
    print(f"Ảnh thứ {index} trong {len(pool)} ảnh test không trùng | image_id={image_id}")
    stem = f"15_demo_test_{image_id:05d}"
    result = demo_image(sample.path, stem, true_label=actual)
    guessed = result.attrs["predicted_label"]
    result.to_csv(TABLE / f"{stem}_top5.csv", index=False)
    pd.DataFrame([{"image_id": image_id, "path": str(sample.path), "split": "test",
        "true_label": actual, "true_name": name_of[actual],
        "predicted_label": guessed, "predicted_name": name_of[guessed],
        "correct": guessed == actual, "model": restored["name"]}]).to_csv(
            TABLE / f"{stem}_comparison.csv", index=False)
    print("Đã lưu hình:", (FIG / f"{stem}.png").as_posix())
    return result

# %% [markdown]
# ### Thực hành · chọn một ảnh trong tập test
# **Chạy lần lượt đến ô Bước 13 ở trên trước**, để có mô hình và hàm dự đoán.
# Đổi `TEST_INDEX = 0` thành `1`, `2`, `100`... rồi chạy lại riêng ô này bằng
# **Shift+Enter**. Chỉ số là vị trí trong danh sách test không trùng, không phải
# số trong tên tệp; với dữ liệu hiện tại chọn từ **0 đến 6144**.
# Kết quả gồm ảnh, nhãn thật, dự đoán, ĐÚNG/SAI và 5 lớp có điểm cao nhất.
# Nhãn thật lấy từ metadata Oxford, không đưa vào mô hình khi dự đoán.
# Hình PNG/PDF và bảng CSV được lưu theo image_id, tiện chọn minh chứng báo cáo.
# Chụp cả ví dụ đúng và sai; một ảnh không đại diện độ chính xác toàn bộ tập test.
# Mỗi lần đổi ảnh chỉ cần chạy lại ô này, không cần huấn luyện lại trong cùng phiên.

# %%
TEST_INDEX = 0  # Thử 1, 2, 100... để lấy ảnh khác trong tập test.
demo_result = demo_test_image(TEST_INDEX)
demo_result.to_csv(TABLE / "demo_top5.csv", index=False)

# %% [markdown]
# ### Tùy chọn · thử ảnh của bạn
# Chép ảnh vào `demo_images/`, điền đường dẫn tương đối ở `MY_IMAGE`, rồi chạy ô.
# Để chuỗi rỗng nếu bỏ qua. Không dùng ảnh này để sửa kết quả test.

# %%
MY_IMAGE = ""  # Ví dụ: "demo_images/hoa_cua_toi.jpg"
if MY_IMAGE:
    image_path = Path(MY_IMAGE)
    if image_path.is_absolute():
        raise ValueError("Dùng đường dẫn tương đối từ thư mục dự án, ví dụ demo_images/hoa.jpg.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Chưa tìm thấy ảnh: {image_path}")
    demo_image(image_path, "16_demo_uploaded")

# %% [markdown]
# ## Bước 14 · Xuất minh chứng và chuẩn bị viết báo cáo
# ZIP chứa PNG/PDF, CSV, model, cấu hình và phiên bản môi trường; không kèm dataset.
# Nhấn **Ctrl+S** để lưu notebook có outputs. ZIP nằm trong thư mục `exports/`.
# Nếu Restart Kernel hoặc mở lại VS Code, chạy lại các ô từ đầu để tạo biến trong bộ nhớ.
# Outputs cũ hiển thị trong notebook không có nghĩa là các biến vẫn tồn tại.
#
# Báo cáo LaTeX theo template học phần, đúng 6 phần: (1) Đặt vấn đề & EDA;
# (2) Thiết kế giải pháp; (3) Phân tích mã nguồn/chức năng; (4) Thử nghiệm & kết quả;
# (5) Thảo luận & so sánh; (6) Tổng kết & hướng phát triển.
# Chưa viết kết luận "HOG tốt hơn" trước khi kiểm tra bảng thật. So sánh M1/M3,
# M2/M4, M3/M5 và M8/M9; giải thích cả cấu hình không cải thiện.
#
# **Tự kiểm tra:** Vì sao train chỉ có 10 ảnh/lớp? Vì sao giữ màu? HOG đo gì?
# Tại sao fit scaler/PCA trên train? F1-macro khác Accuracy thế nào? Mô hình nào
# được chọn và tại sao? Nhầm lẫn nào có bằng chứng? Các thành viên đã đóng góp gì?

# %%
save_json({"completed": True, "selected_model": winner,
           "selected_validation": validation.loc[validation.name == winner].to_dict("records")[0],
           "selected_test": test_results.loc[test_results.name == winner].to_dict("records")[0],
           "selected_test_unique_unseen": clean_test_results.loc[clean_test_results.name == winner].to_dict("records")[0],
           "evaluation_policy": evaluation_policy,
           "number_of_figures": len(figure_records),
           "note": "Kết quả sinh từ lần chạy này; chưa thay thế phần thảo luận do người học viết."}, OUT / "run_summary.json")
(OUT / "READ_ME_FIRST.txt").write_text(
    "Minh chứng thực nghiệm Flowers102.\n"
    "1. run_config.json: môi trường, seed và cách chia tập.\n"
    "2. tables/validation_results.csv và frozen_selection.json: quá trình chọn mô hình.\n"
    "3. tables/test_results.csv: test Oxford gốc; test_results_unique_unseen.csv: test bổ sung loại trùng.\n"
    "4. figure_index.csv: chú thích các hình PNG/PDF.\n"
    "5. tables/error_analysis_to_complete.csv: bổ sung nhận xét quan sát, không bịa nguyên nhân.\n"
    "6. requirements-lock.txt: phiên bản trực tiếp của lần chạy; giữ cùng phiên bản Python.\n"
    "7. models/selected_model.joblib: chỉ mở tệp model từ nguồn tin cậy.\n"
    "Nhấn Ctrl+S trong VS Code để lưu notebook có outputs.\n", encoding="utf-8")
archive_path = export_checkpoint("ket_qua")
print("ĐIỂM DỪNG 3: gửi ZIP kết quả và notebook đã chạy để cùng phân tích và viết báo cáo.")
