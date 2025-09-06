# --*-- coding:utf-8 --*--
# @time: 9/3/25 11:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: generate_priors.py
#
# Generate priors locally with NetSurfP-3 using our wrapper (Python 3.9).
# - Input folder: data/seqs_len10/*.fasta (format: '>ID' + one-line/multi-line sequence)
# - Outputs: runs/<RUN_ID>/inputs/*.fasta and runs/<RUN_ID>/raw/<seq_id>/results.csv|json

# batch_predict.py
import os
import sys
import re
import glob
import subprocess
from pathlib import Path
from datetime import datetime

if __name__ == '__main__':


    PROJECT_ROOT = Path(__file__).resolve().parent

    SEQ_DIR = PROJECT_ROOT / "data" / "seqs_len10"
    RESULT_DIR = PROJECT_ROOT / "result"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1) Find latest checkpoint folder that contains model_best.pth + config.yml
    def find_latest_checkpoint():
        saved_root = PROJECT_ROOT / "NetSurfP-3.0" / "saved" / "nsp3"
        if not saved_root.exists():
            raise FileNotFoundError(f"Cannot find {saved_root}")

        candidates = []
        for p in saved_root.rglob("checkpoints"):
            model = p / "model_best.pth"
            cfg = p / "config.yml"
            if model.exists() and cfg.exists():
                # pick the most recent by mtime of model
                candidates.append((model.stat().st_mtime, model, cfg))

        if not candidates:
            raise FileNotFoundError("No checkpoints found with both model_best.pth and config.yml")

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, model_path, cfg_path = candidates[0]
        return model_path, cfg_path

    MODEL_PATH, CFG_PATH = find_latest_checkpoint()

    # ---- 2) Collect FASTA files
    FA_EXTS = (".fa", ".fasta", ".fsa")

    def is_fasta_file(path: Path) -> bool:
        if path.suffix.lower() in FA_EXTS:
            return True
        # Heuristic: first non-empty line starts with ">"
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        return s.startswith(">")
        except Exception:
            pass
        return False

    fasta_files = [p for p in sorted(SEQ_DIR.iterdir()) if p.is_file() and is_fasta_file(p)]
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA-like files found in {SEQ_DIR}")

    print(f"[INFO] Using model: {MODEL_PATH}")
    print(f"[INFO] Using config: {CFG_PATH}")
    print(f"[INFO] Found {len(fasta_files)} FASTA files in {SEQ_DIR}")

    # ---- 3) Run prediction per FASTA via NSP3 CLI
    # README example: nsp3 predict -c config.yml -d model.pth -p "SecondaryFeatures" -i example_input.txt
    # We capture stdout and write it to TSV files.
    def run_predict(in_fasta: Path, out_tsv: Path):
        cmd = [
            "nsp3", "predict",
            "-c", str(CFG_PATH),
            "-d", str(MODEL_PATH),
            "-p", "SecondaryFeatures",
            "-i", str(in_fasta)
        ]
        print(f"[RUN] {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Bubble up error with stderr for quick diagnosis
            raise RuntimeError(f"Prediction failed for {in_fasta.name}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        # Save stdout to file
        out_tsv.write_text(proc.stdout, encoding="utf-8")

    per_file_outputs = []
    for f in fasta_files:
        out_path = RESULT_DIR / (f.stem + ".tsv")
        run_predict(f, out_path)
        per_file_outputs.append(out_path)
        print(f"[OK] Saved: {out_path}")

    # ---- 4) Merge per-file TSVs into one (simple concatenation with a header check)
    def merge_tsvs(paths, merged_path: Path):
        header = None
        lines_accum = []
        for p in paths:
            content = p.read_text(encoding="utf-8").splitlines()
            if not content:
                continue
            # First non-empty line as header; skip duplicate headers afterward
            # Many NSP3 predictors print a header; adjust if not
            first_non_empty = next((i for i, L in enumerate(content) if L.strip()), None)
            if first_non_empty is None:
                continue
            file_header = content[first_non_empty].strip()
            body = content[first_non_empty+1:]

            if header is None:
                header = file_header
            elif file_header != header:
                # If headers differ, still proceed but keep the first as canonical.
                pass
            lines_accum.extend(body)

        if header is None:
            # No content? create empty merged file
            merged_path.write_text("", encoding="utf-8")
            return

        merged = [header] + lines_accum
        merged_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    MERGED_PATH = RESULT_DIR / "all_predictions.tsv"
    merge_tsvs(per_file_outputs, MERGED_PATH)
    print(f"[DONE] Merged results → {MERGED_PATH}")
