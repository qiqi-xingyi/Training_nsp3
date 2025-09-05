# --*-- conding:utf-8 --*--
# @time:2025/9/4 22:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:npz_to_fp32.py.py

import numpy as np
from pathlib import Path

def convert_to_fp32(in_path: Path, out_path: Path):
    """Convert one .npz file from float64 to float32."""
    with np.load(in_path, allow_pickle=False) as data:
        arrays = {}
        for k, v in data.items():
            if v.dtype == np.float64:
                arrays[k] = v.astype(np.float32)
                print(f"  - {k}: float64 -> float32, shape={v.shape}")
            else:
                arrays[k] = v
                print(f"  - {k}: keep {v.dtype}, shape={v.shape}")
        np.savez_compressed(out_path, **arrays)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[OK] {in_path.name} -> {out_path.name} ({size_mb:.2f} MB)\n")

if __name__ == "__main__":

    root = Path(__file__).resolve().parent
    print(f"[INFO] Scanning directory: {root}")


    for npz_file in root.glob("*.npz"):
        out_file = npz_file.with_name(npz_file.stem + "_fp32.npz")
        if out_file.exists():
            print(f"[SKIP] {out_file.name} already exists")
            continue
        convert_to_fp32(npz_file, out_file)

    print("[DONE] All conversions finished.")
