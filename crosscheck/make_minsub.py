"""
make_minsub.py
Ensemble minimum subtraction baseline as TIFFs: <tiff>/minsub from <tiff>/dirty,
one minimum per laser cavity (frame A and frame B separately).

    python make_minsub.py [tiff_dir]
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("POD_PIV_DATA", os.path.join(ROOT, "data")), "tiff")
os.makedirs(os.path.join(T, "minsub"), exist_ok=True)
for f in "AB":
    files = sorted(glob.glob(os.path.join(T, "dirty", f"*_{f}.tif")))
    S = np.stack([np.asarray(Image.open(p)) for p in files])
    mn = S.min(0)
    for p, im in zip(files, S):
        Image.fromarray((im - mn).astype(np.uint16)).save(os.path.join(T, "minsub", os.path.basename(p)))
print("wrote", os.path.join(T, "minsub"))
