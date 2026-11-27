"""Loading runs from .npz files.

One file per training run (seed). Required arrays:
  test        (E, I, 4) or (I, 4)  confusion counts per epoch and test image
  test_names  (I,)                 test image file names (used to form bootstrap units and to align runs)
Optional:
  val         (E, J, 4)            validation counts per epoch (enables the 'val' rule)
  columns     (4,)                 column order; (TP, TN, P, N) [default] or (TP, FN, TN, FP)
Other array names can be given with the keyword arguments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Run, to_tp_tn_p_n


def load_run(path, test_key="test", val_key="val", names_key="test_names", columns=None, label=None) -> Run:
    z = np.load(path, allow_pickle=False)
    cols = columns or ([str(c) for c in z["columns"]] if "columns" in z.files else ["TP", "TN", "P", "N"])
    test = to_tp_tn_p_n(z[test_key], cols)
    val = to_tp_tn_p_n(z[val_key], cols) if val_key and val_key in z.files else None
    names = [str(n) for n in z[names_key]]
    return Run(test=test, names=names, val=val, label=label or Path(path).name)
