"""Run a bounded portion of the real notebook; retain outputs in a separate file."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
for key, subdir in {"MPLCONFIGDIR": "matplotlib", "IPYTHONDIR": "ipython",
                    "JUPYTER_RUNTIME_DIR": "jupyter_runtime"}.items():
    target = root / "cache" / subdir
    target.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(target)

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--through-step", type=int, choices=range(1, 15), default=4)
args = parser.parse_args()
source = root / "De_tai_07_Phan_loai_hoa_FIT4013.ipynb"
notebook = nbformat.read(source, as_version=4)
nbformat.validate(notebook)
selected = []
step = 0
for cell in notebook.cells:
    if cell.cell_type == "markdown":
        match = re.search(r"^## Bước (\d+)", cell.source, re.MULTILINE)
        if match:
            step = int(match.group(1))
    if step > args.through_step:
        break
    selected.append(cell)
notebook.cells = selected
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

def progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Running code cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

client = NotebookClient(notebook, km=manager, timeout=1800, allow_errors=False,
                        resources={"metadata": {"path": str(root)}}, on_cell_start=progress)
destination = root / "outputs" / f"checked_through_step_{args.through_step:02d}.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, destination)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    print("Saved execution evidence:", destination.relative_to(root), flush=True)

errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
assert not errors
print(json.dumps({"status": "passed", "through_step": args.through_step,
                  "executed_code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                  "data": "actual Oxford Flowers102", "notebook": str(destination.relative_to(root))}, indent=2))
