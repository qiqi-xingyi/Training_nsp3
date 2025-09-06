# batch_predict_individual_safe.py
import sys
import time
import shutil
import subprocess
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
SEQ_DIR = ROOT / "data" / "seqs_len10"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    # --- Settings ---
    PREDICTOR = "SecondaryFeatures"  # change if your predictor class has a different name
    SEARCH_OUTPUT_FOLDERS = [
        ROOT / "saved",
        ROOT / "runs",
        ROOT / "saved" / "nsp3",
        ROOT / "nsp3" / "predict",
    ]
    OUTPUT_EXTS = (".tsv", ".csv", ".txt")

    def find_latest_checkpoint():
        saved_root = ROOT / "saved" / "nsp3"
        candidates = []
        for p in saved_root.rglob("checkpoints"):
            model = p / "model_best.pth"
            cfg = p / "config.yml"
            if model.exists() and cfg.exists():
                candidates.append((model.stat().st_mtime, model, cfg))
        if not candidates:
            raise FileNotFoundError("No checkpoints with both model_best.pth and config.yml were found.")
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][2]

    MODEL_PATH, CFG_PATH = find_latest_checkpoint()
    print(f"[INFO] model:  {MODEL_PATH}")
    print(f"[INFO] config: {CFG_PATH}")

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
                flag, str(out_file)
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

        # 1) Preferred path: predictor supports explicit output flag
        if try_predict_with_output_flag(in_fasta, out_file):
            return True

        # 2) Fallback: run predictor without output flag, then collect from default locations
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

        # Try to locate a file produced during this run
        if copy_latest_generated_file(start_ts, out_file):
            return True

        # As a last resort, persist stdout (usually logs). This indicates no structured output was detected.
        out_file.write_text(proc.stdout, encoding="utf-8")
        print(f"[WARN] structured prediction file not detected; wrote stdout logs to {out_file}")
        return False

    # --- Scan inputs and run ---
    inputs = [p for p in sorted(SEQ_DIR.iterdir()) if p.is_file() and is_fasta(p)]
    if not inputs:
        raise SystemExit(f"No FASTA files found in {SEQ_DIR}")

    for fasta in inputs:
        out_path = RESULT_DIR / (fasta.stem + ".tsv")
        print(f"\n=== Predict {fasta.name} -> {out_path.name} ===")
        ok = predict_one(fasta, out_path)
        if ok:
            print("[OK] saved:", out_path)
        else:
            print("[NOTE] no structured table found for this run; see warnings above.")

