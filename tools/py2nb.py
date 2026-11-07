"""Convert a `# %%` cell-marked .py file into a .ipynb (no dependencies).

Usage: python tools/py2nb.py kaggle/01_data.py   -> kaggle/01_data.ipynb
"""
import json
import pathlib
import sys


def convert(src: pathlib.Path) -> pathlib.Path:
    cells, cur, kind = [], [], None

    def flush():
        if kind is None:
            return
        lines = cur[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        src_lines = [l + "\n" for l in lines]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src_lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in src.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind, cur = ("markdown" if "[markdown]" in line else "code"), []
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1))
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(convert(pathlib.Path(p)))
