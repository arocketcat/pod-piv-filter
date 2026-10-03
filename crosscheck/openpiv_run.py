"""
openpiv_run.py
OpenPIV (windef, multipass with symmetric window deformation) on the TIFF pairs.

    python openpiv_run.py <image_dir> <out.npz> [k0] [k1]
"""
import sys
import time

import numpy as np
from PIL import Image
from openpiv import windef


def settings():
    s = windef.PIVSettings()
    s.windowsizes = (64, 32, 32, 32)       # same schedule as the other two processors
    s.overlap = (32, 16, 16, 16)
    s.num_iterations = 4
    s.deformation_method = "symmetric"
    s.interpolation_order = 3
    s.subpixel_method = "gaussian"
    s.correlation_method = "circular"
    s.normalized_correlation = False
    s.validation_first_pass = True
    s.sig2noise_validate = False           # OpenPIV defaults otherwise
    s.median_threshold = 2
    s.replace_vectors = True
    s.smoothn = False
    s.show_plot = False
    s.save_plot = False
    return s


def run_pair(a, b, s):
    x, y, u, v, flags = windef.simple_multipass(a, b, s)
    # simple_multipass ends with transform_coordinates, which reverses the y
    # coordinate array and negates v but leaves the vector arrays in image row
    # order. Undo it to get image coordinates (y down) like the truth.
    y = y[::-1, :]
    v = -v
    flags = np.asarray(flags) if np.ndim(flags) == 2 else np.zeros_like(u, bool)
    return x, y, u, v, flags


if __name__ == "__main__":
    image_dir, out = sys.argv[1], sys.argv[2]
    k0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    k1 = int(sys.argv[4]) if len(sys.argv) > 4 else 999
    s = settings()
    U, V, F = [], [], []
    t = time.time()
    for k in range(k0, k1 + 1):
        a = np.asarray(Image.open(f"{image_dir}/pair_{k:04d}_A.tif")).astype(np.float64)
        b = np.asarray(Image.open(f"{image_dir}/pair_{k:04d}_B.tif")).astype(np.float64)
        x, y, u, v, fl = run_pair(a, b, s)
        U.append(u); V.append(v); F.append(fl)
    np.savez(out, X=x, Y=y, U=np.array(U), V=np.array(V), F=np.array(F), k0=k0, k1=k1)
    print(f"{image_dir}: pairs {k0}-{k1} in {time.time() - t:.0f} s")
