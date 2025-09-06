# --*-- conding:utf-8 --*--
# @time:2025/9/5 23:25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:check_predictions.py

# check_predictions.py
import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean

def parse_float(s):
    try:
        x = float(s)
        if math.isfinite(x):
            return x
        return float("nan")
    except Exception:
        return float("nan")

def collect_columns(header):
    """
    Return dict with keys: ss8, ss3, dis, rsa, asa, phi, psi
    Values are lists of indices for multi-cols (ss8/ss3/dis) or single index (others, or None if absent)
    """
    name_to_idx = {h.strip(): i for i, h in enumerate(header)}
    patt = {
        "ss8": re.compile(r"^ss8_p\d+$", re.I),
        "ss3": re.compile(r"^ss3_p\d+$", re.I),
        "dis": re.compile(r"^dis_p\d+$", re.I),
    }
    cols = {"ss8": [], "ss3": [], "dis": [], "rsa": None, "asa": None, "phi": None, "psi": None}
    for name, i in name_to_idx.items():
        low = name.lower()
        if patt["ss8"].match(low):
            cols["ss8"].append(i)
        elif patt["ss3"].match(low):
            cols["ss3"].append(i)
        elif patt["dis"].match(low):
            cols["dis"].append(i)
        elif low == "rsa":
            cols["rsa"] = i
        elif low == "asa":
            cols["asa"] = i
        elif low == "phi":
            cols["phi"] = i
        elif low == "psi":
            cols["psi"] = i
    cols["ss8"].sort()
    cols["ss3"].sort()
    cols["dis"].sort()
    return cols

def sum_close_to_one(vals, tol=1e-3):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return False, float("nan")
    s = sum(vals)
    return (abs(s - 1.0) <= tol), s

def safe_stats(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"count": 0, "nan": len(values), "mean": "nan", "min": "nan", "max": "nan"}
    return {
        "count": len(vals),
        "nan": len(values) - len(vals),
        "mean": f"{mean(vals):.6f}",
        "min": f"{min(vals):.6f}",
        "max": f"{max(vals):.6f}",
    }

def check_file(tsv_path, tol=1e-3):
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return None
        cols = collect_columns(header)

        ss8_fail = 0
        ss3_fail = 0
        dis_fail = 0
        n_rows = 0

        rsa_vals, asa_vals, phi_vals, psi_vals = [], [], [], []

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            n_rows += 1

            # ss8 / ss3 / dis normalization checks
            if cols["ss8"]:
                s8 = [parse_float(row[i]) for i in cols["ss8"] if i < len(row)]
                ok, s = sum_close_to_one(s8, tol=tol)
                if not ok:
                    ss8_fail += 1
            if cols["ss3"]:
                s3 = [parse_float(row[i]) for i in cols["ss3"] if i < len(row)]
                ok, s = sum_close_to_one(s3, tol=tol)
                if not ok:
                    ss3_fail += 1
            if cols["dis"]:
                d = [parse_float(row[i]) for i in cols["dis"] if i < len(row)]
                ok, s = sum_close_to_one(d, tol=tol)
                if not ok:
                    dis_fail += 1

            # collect scalar heads
            if cols["rsa"] is not None and cols["rsa"] < len(row):
                rsa_vals.append(parse_float(row[cols["rsa"]]))
            if cols["asa"] is not None and cols["asa"] < len(row):
                asa_vals.append(parse_float(row[cols["asa"]]))
            if cols["phi"] is not None and cols["phi"] < len(row):
                phi_vals.append(parse_float(row[cols["phi"]]))
            if cols["psi"] is not None and cols["psi"] < len(row):
                psi_vals.append(parse_float(row[cols["psi"]]))

        result = {
            "file": str(tsv_path),
            "rows": n_rows,
            "ss8_fail": ss8_fail,
            "ss3_fail": ss3_fail,
            "dis_fail": dis_fail,
            "rsa_stats": safe_stats(rsa_vals),
            "asa_stats": safe_stats(asa_vals) if asa_vals else {"count": 0, "nan": n_rows, "mean": "nan", "min": "nan", "max": "nan"},
            "phi_stats": safe_stats(phi_vals),
            "psi_stats": safe_stats(psi_vals),
        }
        return result

def print_summary(res):
    print(f"\nFile: {res['file']}")
    print(f"  Rows: {res['rows']}")
    print(f"  SS8 not ~1.0: {res['ss8_fail']}")
    print(f"  SS3 not ~1.0: {res['ss3_fail']}")
    print(f"  DIS not ~1.0: {res['dis_fail']}")
    rsa = res["rsa_stats"]; asa = res["asa_stats"]; phi = res["phi_stats"]; psi = res["psi_stats"]
    print(f"  RSA  -> count: {rsa['count']}, nan: {rsa['nan']}, mean: {rsa['mean']}, min: {rsa['min']}, max: {rsa['max']}")
    print(f"  ASA  -> count: {asa['count']}, nan: {asa['nan']}, mean: {asa['mean']}, min: {asa['min']}, max: {asa['max']}")
    print(f"  PHI  -> count: {phi['count']}, nan: {phi['nan']}, mean: {phi['mean']}, min: {phi['min']}, max: {phi['max']}")
    print(f"  PSI  -> count: {psi['count']}, nan: {psi['nan']}, mean: {psi['mean']}, min: {psi['min']}, max: {psi['max']}")

def write_report_csv(results, out_csv: Path):
    headers = [
        "file","rows","ss8_fail","ss3_fail","dis_fail",
        "rsa_count","rsa_nan","rsa_mean","rsa_min","rsa_max",
        "asa_count","asa_nan","asa_mean","asa_min","asa_max",
        "phi_count","phi_nan","phi_mean","phi_min","phi_max",
        "psi_count","psi_nan","psi_mean","psi_min","psi_max",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            rsa = r["rsa_stats"]; asa = r["asa_stats"]; phi = r["phi_stats"]; psi = r["psi_stats"]
            w.writerow([
                r["file"], r["rows"], r["ss8_fail"], r["ss3_fail"], r["dis_fail"],
                rsa["count"], rsa["nan"], rsa["mean"], rsa["min"], rsa["max"],
                asa["count"], asa["nan"], asa["mean"], asa["min"], asa["max"],
                phi["count"], phi["nan"], phi["mean"], phi["min"], phi["max"],
                psi["count"], psi["nan"], psi["mean"], psi["min"], psi["max"],
            ])
    print(f"\n[OK] Report written: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Validate NetSurfP-3.0 TSV predictions.")
    ap.add_argument("--dir", default=r"G:\Train_NSP\result", help="Folder containing per-sequence TSV files.")
    ap.add_argument("--tol", type=float, default=1e-3, help="Tolerance for softmax sums (default: 1e-3).")
    ap.add_argument("--report", default="", help="Optional path to write CSV report.")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tsv"])
    if not files:
        raise SystemExit(f"No .tsv files found in {in_dir}")

    all_results = []
    for fp in files:
        res = check_file(fp, tol=args.tol)
        if res is None:
            print(f"Skip (empty or invalid): {fp}")
            continue
        print_summary(res)
        all_results.append(res)

    if args.report:
        out_csv = Path(args.report)
        write_report_csv(all_results, out_csv)

if __name__ == "__main__":
    main()
