"""Convert the readable step-by-step source into a VS Code/Jupyter notebook.

Run only when intentionally rebuilding from flowers102_steps.py: outputs are cleared.
"""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / "flowers102_steps.py").read_text(encoding="utf-8")
cells = []
for i, chunk in enumerate(text.split("# %%")[1:]):
    first, _, body = chunk.partition("\n")
    markdown = first.strip() == "[markdown]"
    if markdown:
        source = "\n".join(line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                           for line in body.splitlines()).strip() + "\n"
    else:
        source = body.strip() + "\n"
        ast.parse(source)
    cell = {"cell_type": "markdown" if markdown else "code", "id": f"step-{i:03}",
            "metadata": {"id": f"step-{i:03}"}, "source": source.splitlines(keepends=True)}
    if not markdown:
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
notebook = {"cells": cells, "metadata": {
              "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
              "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = root / "De_tai_07_Phan_loai_hoa_FIT4013.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Created {path.name}: {len(cells)} cells")
