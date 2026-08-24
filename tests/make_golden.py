"""Regenerate tests/golden/tube.npz from the current meshik.py output.

Run from the repo root:  uv run python tests/make_golden.py
Only do this when a behaviour change is intended; test_golden.py will then pin the new output.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import run_meshik  # noqa: E402

if __name__ == "__main__":
    g = run_meshik()
    out = os.path.join(os.path.dirname(__file__), "golden", "tube.npz")
    np.savez(out, x_result=g["x_result"], w_result=g["w_result"], error=g["error"])
    print(f"wrote {out}: error={g['error']!r} w={g['w_result'].ravel()!r}")
