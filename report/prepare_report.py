"""Build report tables/figures from ONE completed Oxford run, without model fitting.

Run from flower_project: .venv/Scripts/python.exe report/prepare_report.py RUN_ID
The authored prose refers to the initial run. Review it if using another run.
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))
import joblib
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_id")
args = parser.parse_args()
if Path(args.run_id).name != args.run_id:
    raise ValueError("Use a run folder name, not a path")
run = Path("outputs") / args.run_id
summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
assert summary["completed"]
assert summary["evaluation_policy"]["official_test_images"] == 6149
latex = Path("report/latex")
generated, figures = latex / "generated", latex / "figures"
evidence = Path("report/evidence") / args.run_id
for d in [generated, figures, evidence, latex / "screenshots"]:
    d.mkdir(parents=True, exist_ok=True)
tables = run / "tables"
val = pd.read_csv(tables / "validation_results.csv")
test = pd.read_csv(tables / "test_results.csv")
clean = pd.read_csv(tables / "test_results_unique_unseen.csv")
assert len(val) == len(test) == len(clean) == 10
winner = summary["selected_model"]
assert winner == val[val.name != "M0_Dummy"].sort_values("f1_macro", ascending=False, kind="stable").iloc[0]["name"]
pred = pd.read_csv(tables / f"predictions_{winner}.csv")
assert len(pred) == 6149
assert np.isclose(pred.correct.mean(), summary["selected_test"]["accuracy"])
cm = pd.read_csv(tables / f"confusion_{winner}.csv", index_col=0).to_numpy()
assert cm.sum() == 6149 and cm.trace() == pred.correct.sum()
for name in test.name:
    assert pd.read_csv(tables / f"confusion_unique_unseen_{name}.csv", index_col=0).to_numpy().sum() == 6145

def tex(value):
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(mapping.get(c, c) for c in str(value))

def number(x, digits=2):
    return f"{float(x):.{digits}f}".replace(".", ",")

def table(filename, caption, header, rows, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[H]\centering\small", r"\caption{" + caption + "}",
             r"\begin{tabular}{" + align + r"}\toprule", " & ".join(header) + r"\\\midrule"]
    lines += [" & ".join(map(str, row)) + r"\\" for row in rows]
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (generated / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
for frame, filename, caption in [
    (val, "validation_table.tex", "Kết quả validation (\\%). P, R, F1 dùng macro; F1w dùng weighted. M8 được chọn."),
    (test, "test_table.tex", "Kết quả trên 6.149 ảnh test Oxford (\\%). M8 là mô hình đã chốt bằng validation.")]:
    rows = []
    for row in frame.itertuples():
        values = [row.name.split("_")[0]] + [number(100 * getattr(row, m)) for m in metrics]
        if row.name == winner:
            values = [r"\textbf{" + v + "}" for v in values]
        rows.append(values)
    table(filename, caption, ["Mã", "Accuracy", "P", "R", "F1", "F1w"], rows)

merged = test.merge(clean, on="name", suffixes=("_original", "_clean"))
table("clean_table.tex", "Đối chiếu test Oxford và test không trùng (\\%). Cùng mô hình và dự đoán, không huấn luyện lại.",
      ["Mã", "Acc. gốc", "Acc. loại trùng", "F1 gốc", "F1 loại trùng"],
      [[r.name.split("_")[0], number(r.accuracy_original*100), number(r.accuracy_clean*100),
        number(r.f1_macro_original*100), number(r.f1_macro_clean*100)] for r in merged.itertuples()])
times = val.merge(test[["name", "predict_seconds"]], on="name", suffixes=("_val", "_test"))
dimensions = {"pixels": "12.288", "hog": "1.764", "hog_hsv": "1.812"}
table("time_table.tex", "Số chiều và thời gian mô hình trong một lần chạy (giây). Fit không gồm toàn bộ quá trình tuning.",
      ["Mã", "Số chiều", "Fit train", "Predict val", "Predict test"],
      [[r.name.split("_")[0], "128" if r.name.startswith("M9") else dimensions[r.feature],
        number(r.fit_seconds, 4), number(r.predict_seconds_val, 4), number(r.predict_seconds_test, 4)] for r in times.itertuples()])
lookup = test.set_index(test.name.str.split("_").str[0])
pairs = [("M1", "M3", "Pixel sang HOG; cùng K-NN"), ("M2", "M4", "Pixel sang HOG; cùng cây"),
         ("M3", "M5", "Thêm HSV; cùng K-NN"), ("M3", "M6", "Tuning K-NN trên HOG"),
         ("M4", "M7", "Tuning cây trên HOG"), ("M5", "M8", "Tuning K-NN trên HOG+HSV"),
         ("M8", "M9", "Thêm scale và PCA128")]
table("ablation_table.tex", "So sánh có kiểm soát: chênh lệch F1-macro test tính theo điểm phần trăm (sau trừ trước).",
      ["Cặp", "Yếu tố thay đổi", "Chênh lệch"],
      [[f"{a} $\\to$ {b}", tex(label), number(100*(lookup.loc[b].f1_macro-lookup.loc[a].f1_macro))] for a,b,label in pairs], "llr")
confused = pd.read_csv(tables / "top_confused_pairs.csv").head(6)
table("confused_table.tex", "Sáu cặp nhầm nhiều nhất của M8 trên test chính thức. Tỷ lệ tính trên support lớp thật.",
      ["Lớp thật", "Dự đoán", "Số ảnh", "Tỷ lệ (\\%)"],
      [[tex(r.true_name), tex(r.predicted_name), r.count, number(r.share_of_true_class*100)] for r in confused.itertuples()], "llrr")
macros = {"RunID": tex(args.run_id), "ValFOne": number(summary["selected_validation"]["f1_macro"]*100),
          "ValAccuracy": number(summary["selected_validation"]["accuracy"]*100),
          "TestAccuracy": number(summary["selected_test"]["accuracy"]*100),
          "TestFOne": number(summary["selected_test"]["f1_macro"]*100),
          "CleanAccuracy": number(summary["selected_test_unique_unseen"]["accuracy"]*100),
          "CleanFOne": number(summary["selected_test_unique_unseen"]["f1_macro"]*100),
          "CorrectCount": str(int(pred.correct.sum())),
          "BaselineGain": number(100*(lookup.loc["M8"].f1_macro-lookup.loc["M1"].f1_macro))}
(generated / "so_lieu.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in macros.items())+"\n", encoding="utf-8")
needed = ["01_class_distribution", "00_exact_duplicates", "04_preprocessing_hog_hsv", "05_knn_tuning",
          "09_test_comparison", "11_confusion_detail", "15_demo_test_00001"]
for stem in needed:
    shutil.copy2(run / "figures" / f"{stem}.pdf", figures / f"{stem}.pdf")
for src in list(tables.glob("*.csv")) + list(run.glob("*.json")):
    dest = evidence / ("tables" if src.suffix == ".csv" else "metadata") / src.name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(src, dest)

names = {int(k): v for k,v in json.loads((run / "class_names.json").read_text(encoding="utf-8")).items()}
manifest = pd.read_csv(tables / "split_manifest.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
sample_rows = [manifest[(manifest.split == "train") & (manifest.label == label)].iloc[0] for label in [1,8,20,31,54,58,77,102]]
fig, axes = plt.subplots(2,4,figsize=(11,6),constrained_layout=True)
for ax, row in zip(axes.ravel(), sample_rows):
    with Image.open(Path("data/jpg") / f"image_{int(row.image_id):05d}.jpg") as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(f"{row.label}: {names[int(row.label)]}\nID {row.image_id}", fontsize=10)
    ax.axis("off")
fig.savefig(figures / "report_samples.pdf", bbox_inches="tight")
plt.close(fig)

error_ids = [2988,499,6007,3721,7958,1258]
fig, axes = plt.subplots(3,2,figsize=(10,11),constrained_layout=True)
for ax, image_id in zip(axes.ravel(), error_ids):
    row = pred[pred.image_id == image_id].iloc[0]
    assert not row.correct
    with Image.open(Path("data/jpg") / f"image_{image_id:05d}.jpg") as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(f"ID {image_id} | Thật: {row['class_name']}\nĐoán: {row.predicted_name}", fontsize=11)
    ax.axis("off")
fig.savefig(figures / "report_errors.pdf", bbox_inches="tight")
plt.close(fig)

# Reuse the actual notebook's feature and demo functions, without fitting a model.
source = Path("flowers102_steps.py").read_text(encoding="utf-8")
function_names = ["read_rgb", "l2", "extract_features", "demo_image", "demo_test_image"]
nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in function_names]
functions = "\n\n".join(ast.get_source_segment(source,n) for n in nodes)
setup = '''from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from IPython.display import display
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog

if not Path("flowers102_steps.py").is_file():
    raise RuntimeError("Mở notebook tại thư mục flower_project và chọn kernel .venv.")
OUT = Path("outputs") / "RUN_TOKEN"
restored = joblib.load(OUT / "models/selected_model.joblib")
assert restored["feature_version"] == "rgb-bilinear64_hog9-8-2-sqrt-l2_hsv16-l2_equal-v1"
IMAGE_SIZE, HOG_CONFIG = restored["image_size"], restored["hog_config"]
name_of = {int(k): v for k, v in restored["class_names"].items()}
manifest = pd.read_csv(OUT / "tables/split_manifest.csv")
test_part = manifest[manifest.split == "test"].copy()
test_part["path"] = test_part.image_id.map(lambda i: f"data/jpg/image_{i:05d}.jpg")
parts = {"test": test_part.reset_index(drop=True)}
policy = json.loads((OUT / "evaluation_policy.json").read_text(encoding="utf-8"))
clean_test_ids = test_part.loc[~test_part.image_id.isin(policy["excluded_test_ids"]), "image_id"].to_numpy()
FIG = OUT / "demo_additional/figures"
TABLE = OUT / "demo_additional/tables"
FIG.mkdir(parents=True, exist_ok=True)
TABLE.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
def save_figure(fig, name, caption):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
print("Đã nạp mô hình:", restored["name"])
print("Số ảnh test không trùng:", len(clean_test_ids))
'''.replace("RUN_TOKEN", args.run_id)
pool = manifest[(manifest.split == "test") & ~manifest.image_id.isin(summary["evaluation_policy"]["excluded_test_ids"])].reset_index(drop=True)
first_correct_id = int(pred[pred.correct & pred.image_id.isin(pool.image_id)].iloc[0].image_id)
correct_index = int(pool.index[pool.image_id == first_correct_id][0])
nb = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell("# Demo mô hình hoa đã lưu\nChọn kernel `.venv`. Chạy ô khởi tạo một lần, sau đó đổi `TEST_INDEX` trong ô demo. Không huấn luyện lại. Nhãn thật chỉ dùng đối chiếu. Các ví dụ đúng/sai bên dưới được chọn từ kết quả để minh họa, không đại diện cho độ chính xác.\n\nMô hình và dữ liệu phải có tại đường dẫn tương đối được ghi trong ô đầu."),
    nbformat.v4.new_code_cell(setup+"\n"+functions),
    nbformat.v4.new_markdown_cell("## Ví dụ dự đoán sai\nGiữ cả ví dụ sai để báo cáo trung thực. Chỉ số 0 ứng với ID 1 trong lần chạy này."),
    nbformat.v4.new_code_cell("TEST_INDEX = 0\ndemo_result = demo_test_image(TEST_INDEX)"),
    nbformat.v4.new_markdown_cell(f"## Ví dụ dự đoán đúng\nChỉ số {correct_index} ứng với ID {first_correct_id}. Có thể đổi chỉ số từ 0 đến 6144 để thử ảnh khác."),
    nbformat.v4.new_code_cell(f"TEST_INDEX = {correct_index}\ndemo_result = demo_test_image(TEST_INDEX)")],
    metadata={"kernelspec":{"display_name":"Python 3 (.venv)","language":"python","name":"python3"}})
nbformat.write(nb, "Demo_mo_hinh_da_luu.ipynb")
ctx = dict(__name__="__main__")
exec(compile(setup+"\n"+functions,"saved_model_demo","exec"),ctx)
demo = ctx["demo_test_image"](correct_index)
assert demo.attrs["predicted_label"] == int(pred[pred.image_id==first_correct_id].iloc[0].predicted_label)
shutil.copy2(ctx["FIG"] / f"15_demo_test_{first_correct_id:05d}.pdf", figures / "demo_correct.pdf")
shutil.copy2("outputs/checked_through_step_14.ipynb", "Ket_qua_thuc_nghiem_FIT4013.ipynb")
provenance={"run_id":args.run_id,"data":"actual Oxford Flowers102","selected_model":winner,
            "source_metrics_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [tables/"validation_results.csv",tables/"test_results.csv",tables/"test_results_unique_unseen.csv"]},
            "demo_correct_index":correct_index,"demo_correct_image_id":first_correct_id,"demo_wrong_index":0,
            "note":"Authored prose is specific to this run; review if regenerating with another run."}
(evidence/"report_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
print("Report evidence ready:", args.run_id, "correct demo index:", correct_index)
