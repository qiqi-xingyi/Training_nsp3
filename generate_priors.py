# batch_predict_individual_fixed.py
import sys
import time
import shutil
import subprocess
from pathlib import Path

# --- Project root and I/O folders ---
ROOT = Path(r"G:\Train_NSP\NetSurfP-3.0")  # repository root
SEQ_DIR = ROOT / "data" / "seqs_len10"      # input FASTA folder
RESULT_DIR = ROOT / "result"                # output folder
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --- Exact checkpoint paths you provided ---
CFG_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\config.yml")
MODEL_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\model_best.pth")

# --- Predictor class (change if yours differs) ---
PREDICTOR = "SecondaryFeatures"

# --- Fallback search locations in case the predictor writes files elsewhere ---
SEARCH_OUTPUT_FOLDERS = [
    ROOT / "saved",
    ROOT / "runs",
    ROOT / "saved" / "nsp3",
    ROOT / "nsp3" / "predict",
]
OUTPUT_EXTS = (".tsv", ".csv", ".txt")

def is_fasta(path: Path) -> bool:
    if path.suffix.lower() in (".fa", ".fasta", ".fsa"):
        return True
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    return s.startswith(">")
    except Exception:
        return False
    return False

def try_predict_with_output_flag(in_fasta: Path, out_file: Path) -> bool:
    """Try '--output' and then '-o'. Return True if out_file is created with content."""
    for flag in ("--output", "-o"):
        cmd = [
            sys.executable, "-m", "nsp3.cli", "predict",
            "-c", str(CFG_PATH),
            "-d", str(MODEL_PATH),
            "-p", PREDICTOR,
            "-i", str(in_fasta),
            flag, str(out_file),
        ]
        print("[RUN]", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return True
        if proc.returncode != 0:
            print(f"[WARN] predictor call failed with {flag}. stderr:\n{proc.stderr.strip()}")
    return False

def copy_latest_generated_file(since_ts: float, out_file: Path) -> bool:
    """Search typical output folders for a new table file and copy it to out_file."""
    candidates = []
    for base in SEARCH_OUTPUT_FOLDERS:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in OUTPUT_EXTS:
                try:
                    if p.stat().st_mtime >= since_ts:
                        candidates.append(p)
                except FileNotFoundError:
                    pass
    if not candidates:
        return False
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0]
    shutil.copy2(latest, out_file)
    print(f"[INFO] copied latest result: {latest} -> {out_file}")
    return out_file.exists() and out_file.stat().st_size > 0

def predict_one(in_fasta: Path, out_file: Path) -> bool:
    start_ts = time.time()

    # Preferred path: explicit output flag
    if try_predict_with_output_flag(in_fasta, out_file):
        return True

    # Fallback: run and then pick up any produced file
    cmd = [
        sys.executable, "-m", "nsp3.cli", "predict",
        "-c", str(CFG_PATH),
        "-d", str(MODEL_PATH),
        "-p", PREDICTOR,
        "-i", str(in_fasta),
    ]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"[ERROR] prediction failed for {in_fasta.name}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return False

    if copy_latest_generated_file(start_ts, out_file):
        return True

    # Last resort: write stdout (likely logs)
    out_file.write_text(proc.stdout, encoding="utf-8")
    print(f"[WARN] structured prediction file not detected; wrote stdout logs to {out_file}")
    return False

def main():
    if not CFG_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("config.yml or model_best.pth not found at the hard-coded paths.")
    inputs = [p for p in sorted(SEQ_DIR.iterdir()) if p.is_file() and is_fasta(p)]
    if not inputs:
        raise SystemExit(f"No FASTA files found in {SEQ_DIR}")

    print(f"[INFO] config: {CFG_PATH}")
    print(f"[INFO] model:  {MODEL_PATH}")
    print(f"[INFO] inputs: {len(inputs)} files in {SEQ_DIR}")
    print(f"[INFO] output: {RESULT_DIR}")

    for fasta in inputs:
        out_path = RESULT_DIR / (fasta.stem + ".tsv")
        print(f"\n=== Predict {fasta.name} -> {out_path.name} ===")
        ok = predict_one(fasta, out_path)
        if ok:
            print("[OK] saved:", out_path)
        else:
            print("[NOTE] no structured table found for this run; see warnings above.")

if __name__ == "__main__":
    main()
