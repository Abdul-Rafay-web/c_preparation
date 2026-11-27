"""Execute Jupyter notebooks cleanly and embed stdout + matplotlib PNG outputs inline."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def execute_notebook(nb_path: Path) -> None:
    print(f"\n{'='*70}\nExecuting {nb_path.name} ...\n{'='*70}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    old_cwd = Path.cwd()
    os.chdir(nb_path.parent)

    ns: dict = {"__name__": "__main__"}
    exec_count = 0

    try:
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            if not code.strip():
                continue

            exec_count += 1
            cell["execution_count"] = exec_count
            outputs: list[dict] = []

            captured_figs: list[bytes] = []

            def _capture_show(*_args, **_kwargs):
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                    captured_figs.append(buf.getvalue())
                plt.close("all")

            plt.show = _capture_show  # type: ignore[assignment]
            stdout_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(code, f"{nb_path.name}[cell {idx}]", "exec"), ns)

            text_out = stdout_buf.getvalue()
            if text_out:
                print(text_out, end="" if text_out.endswith("\n") else "\n")
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text_out.splitlines(keepends=True),
                })

            for png_bytes in captured_figs:
                b64 = base64.b64encode(png_bytes).decode("ascii")
                outputs.append({
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {
                        "image/png": b64,
                        "text/plain": ["<Figure>"],
                    },
                })

            cell["outputs"] = outputs
    finally:
        os.chdir(old_cwd)

    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Finished & saved outputs into {nb_path.name}")


def main() -> None:
    nb_dir = Path(__file__).resolve().parent / "notebooks"
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "01_data_exploration.ipynb",
        "02_float_baseline.ipynb",
        "03_pruning_and_quantization.ipynb",
        "04_hls4ml_hardware_verification.ipynb",
    ]
    for name in targets:
        execute_notebook(nb_dir / name)


if __name__ == "__main__":
    main()
