# --*-- conding:utf-8 --*--
# @time:2025/9/4 23:05
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:shrink_npz_2over3.py.py


from pathlib import Path
import numpy as np
from math import ceil


FRACTION = 0.3
SHUFFLE = True
SEED = 42
# ==========================

def shrink_one(in_path: Path) -> Path:
    out_path = in_path.with_name(f"{in_path.stem}_p{int(FRACTION*100):02d}_fp32.npz")
    with np.load(in_path, allow_pickle=False) as z:
        keys = list(z.files)
        # 用 pdbids 的长度作为蛋白数量（否则用第一个形状匹配的数组）
        if "pdbids" in keys:
            N = len(z["pdbids"])
        else:
            # 找到首个满足“有第0维”的数组，作为 N
            N = None
            for k in keys:
                a = z[k]
                if a.ndim >= 1:
                    N = a.shape[0]
                    break
            if N is None:
                raise RuntimeError(f"{in_path.name}: 无法推断样本数 N")

        k = max(1, ceil(N * FRACTION))
        if SHUFFLE:
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(N)[:k]
            idx = np.sort(perm)  # 保持顺序
        else:
            idx = np.arange(k)

        out = {}
        for kname in keys:
            arr = z[kname]
            # 若数组第 0 维等于 N，则沿第 0 维做选择
            if arr.ndim >= 1 and arr.shape[0] == N:
                arr = arr[idx]
            # 降精度到 float32（其他 dtype 保持不变）
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            out[kname] = arr

        np.savez_compressed(out_path, **out)
        print(f"[OK] {in_path.name} -> {out_path.name}  "
              f"({out_path.stat().st_size/1024/1024:.2f} MB)")
        return out_path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(f"[INFO] scan: {root}")
    for f in sorted(root.glob("*.npz")):

        if any(tag in f.stem for tag in ["_fp32", "_p"]):
            print(f"[SKIP] {f.name}")
            continue
        try:
            shrink_one(f)
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}")
    print("[DONE] all.")
