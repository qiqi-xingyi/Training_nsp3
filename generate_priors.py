# generate_priors.py
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

# ----- Fixed paths from your message -----
CFG_PATH = Path(
    r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\config.yml"
)
MODEL_PATH = Path(
    r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\model_best.pth"
)

# ----- Default IO -----
DEFAULT_INPUT_DIR = Path(r"G:\Train_NSP\data\seqs_len10")
DEFAULT_OUTPUT_DIR = Path(r"G:\Train_NSP\result")
PREDICTOR = "SecondaryFeatures"  # change if your predictor class differs

# If the predictor writes its own files elsewhere, search here as a fallback
FALLBACK_SEARCH_DIRS = [
    Path(r"G:\Train_NSP\saved"),
    Path(r"G:\Train_NSP\runs"),
]
OUTPUT_EXTS = (".tsv", ".csv", ".txt")


# ---------------- Helpers ----------------
def parse_fasta(fpath: Path):
    """Return (id, sequence) for the FIRST record in a FASTA file."""
    header = None
    seq_parts = []
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if header is None:
                    header = s[1:].strip().split()[0]
                else:
                    # multiple records; stop at the first
                    break
            else:
                seq_parts.append(s)
    if header is None or not seq_parts:
        raise ValueError(f"Invalid FASTA (no header/sequence): {fpath}")
    return header, "".join(seq_parts).upper()


def write_tmp_input_txt(out_dir: Path, rec_id: str, seq: str, sep: str = "\t") -> Path:
    """Write 'id <sep> sequence' to a temporary txt near outputs."""
    tmp_dir = out_dir / "_tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"{rec_id}.txt"
    tmp_file.write_text(f"{rec_id}{sep}{seq}\n", encoding="utf-8")
    return tmp_file


def try_predict_with_output_flag(cfg: Path, model: Path, predictor: str,
                                 in_path: Path, out_file: Path) -> bool:
    """Try to run predictor with explicit output flag."""
    for flag in ("--output", "-o"):
        cmd = [
            sys.executable, "-m", "nsp3.cli", "predict",
            "-c", str(cfg),
            "-d", str(model),
            "-p", predictor,
            "-i", str(in_path),
            flag, str(out_file),
        ]
        print("[RUN]", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return True
        if proc.returncode != 0:
            print(f"[WARN] predictor failed with {flag}. stderr:\n{proc.stderr.strip()}")
    return False


def run_predict_plain(cfg: Path, model: Path, predictor: str, in_path: Path):
    """Run predictor without explicit output flag and return CompletedProcess."""
    cmd = [
        sys.executable, "-m", "nsp3.cli", "predict",
        "-c", str(cfg),
        "-d", str(model),
        "-p", predictor,
        "-i", str(in_path),
    ]
    print("[RUN]", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def copy_latest_generated_file(search_roots, since_ts: float, dest: Path) -> bool:
    """Find newest table file created since 'since_ts' and copy it to 'dest'."""
    candidates = []
    for base in search_roots:
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
    newest = candidates[0]
    shutil.copy2(newest, dest)
    print(f"[INFO] copied latest result: {newest} -> {dest}")
    return dest.exists() and dest.stat().st_size > 0


def predict_file(fasta_path: Path, out_file: Path) -> bool:
    """End-to-end prediction for one FASTA file."""
    # 1) Read FASTA and prepare "id<sep>sequence" inputs (tab and space variants)
    rec_id, seq = parse_fasta(fasta_path)
    tmp_tab = write_tmp_input_txt(out_file.parent, rec_id, seq, sep="\t")
    tmp_spc = write_tmp_input_txt(out_file.parent, rec_id, seq, sep=" ")

    # 2) Try direct output with tab-separated input
    start_ts = time.time()
    if try_predict_with_output_flag(CFG_PATH, MODEL_PATH, PREDICTOR, tmp_tab, out_file):
        return True

    # 3) Try with space-separated input
    if try_predict_with_output_flag(CFG_PATH, MODEL_PATH, PREDICTOR, tmp_spc, out_file):
        return True

    # 4) Fallback: run plain (tab input), then collect a file produced by the predictor
    proc = run_predict_plain(CFG_PATH, MODEL_PATH, PREDICTOR, tmp_tab)
    if proc.returncode == 0 and copy_latest_generated_file(FALLBACK_SEARCH_DIRS, start_ts, out_file):
        return True

    # 5) Last resort: write stdout (usually logs) so nothing is lost
    out_file.write_text(proc.stdout, encoding="utf-8")
    print(f"[WARN] no structured table detected; wrote stdout logs to {out_file}")
    return False


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch predict NetSurfP-3.0 on FASTA files.")
    ap.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Folder with FASTA files.")
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Folder for per-file outputs.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    # Validate essential paths
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"config.yml not found: {CFG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model_best.pth not found: {MODEL_PATH}")
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather inputs (only FASTA-like files)
    def is_fasta_file(p: Path) -> bool:
        if p.suffix.lower() in (".fa", ".fasta", ".fsa"):
            return True
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        return s.startswith(">")
        except Exception:
            return False
        return False

    inputs = [p for p in sorted(in_dir.iterdir()) if p.is_file() and is_fasta_file(p)]
    if not inputs:
        raise SystemExit(f"No FASTA files found in {in_dir}")

    print(f"[INFO] config:    {CFG_PATH}")
    print(f"[INFO] model:     {MODEL_PATH}")
    print(f"[INFO] predictor: {PREDICTOR}")
    print(f"[INFO] inputs:    {len(inputs)} files in {in_dir}")
    print(f"[INFO] output:    {out_dir}")

    for fasta in inputs:
        out_path = out_dir / (fasta.stem + ".tsv")
        print(f"\n=== Predict {fasta.name} -> {out_path.name} ===")
        ok = predict_file(fasta, out_path)
        if ok:
            print("[OK] saved:", out_path)
        else:
            print("[NOTE] structured output not found; see warnings above.")

if __name__ == "__main__":
    main()
