# generate_priors.py
import sys
import time
import csv
import io
import argparse
import subprocess
from pathlib import Path
from typing import List

# ---- fixed paths you provided ----
CFG_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\config.yml")
MODEL_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\model_best.pth")

DEFAULT_INPUT_DIR = Path(r"G:\Train_NSP\data\seqs_len10")
DEFAULT_OUTPUT_DIR = Path(r"G:\Train_NSP\result")
PREDICTOR = "SecondaryFeatures"  # keep as in README

# most repos drop prediction files under these
SEARCH_DIRS = [
    Path(r"G:\Train_NSP\NetSurfP-3.0\saved"),
    Path(r"G:\Train_NSP\NetSurfP-3.0\runs"),
    Path(r"G:\Train_NSP\saved"),
    Path(r"G:\Train_NSP\runs"),
]

RESULT_EXTS = (".csv", ".tsv", ".txt")


def read_first_fasta_record(fp: Path):
    header = None
    parts = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if header is None:
                    header = s[1:].split()[0]
                else:
                    break
            else:
                parts.append(s)
    if not header or not parts:
        raise ValueError(f"Invalid FASTA: {fp}")
    return header, "".join(parts).upper()


def write_id_seq_txt(tmp_dir: Path, rec_id: str, seq: str) -> Path:
    """
    Many predictors expect 'ID<tab>SEQUENCE' per line (README uses a .txt example).
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"{rec_id}.txt"
    p.write_text(f"{rec_id}\t{seq}\n", encoding="utf-8")
    return p


def run_predict(input_file: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "nsp3.cli", "predict",
        "-c", str(CFG_PATH),
        "-d", str(MODEL_PATH),
        "-p", PREDICTOR,
        "-i", str(input_file),
    ]
    print("[RUN]", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def list_new_results(since: float) -> List[Path]:
    found = []
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in RESULT_EXTS:
                try:
                    if p.stat().st_mtime >= since:
                        found.append(p)
                except FileNotFoundError:
                    pass
    # newest first
    found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return found


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    # quick sniff first 1KB
    raw = path.read_bytes()[:1024].decode("utf-8", errors="ignore")
    if "\t" in raw and "," not in raw:
        return "\t"
    return ","


def filter_rows_for_id(text: str, rec_id: str) -> str:
    """
    Keep rows belonging to this sequence ID if a 'id' or 'name' column exists.
    Fall back to returning the whole text if we cannot parse columns.
    """
    # try TSV
    for delim in ("\t", ","):
        buf = io.StringIO(text)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(text[:1024], delimiters=delim)
            buf.seek(0)
            reader = csv.DictReader(buf, dialect=dialect)
        except Exception:
            continue

        lower_headers = [h.lower() for h in reader.fieldnames or []]
        id_col = None
        for cand in ("id", "name", "seq_id", "sequence", "protein", "header"):
            if cand in lower_headers:
                id_col = reader.fieldnames[lower_headers.index(cand)]
                break

        rows = []
        for row in reader:
            if not id_col:
                rows.append(row)
            else:
                # match exact or prefix (e.g., FASTA "1bai|..." vs "1bai")
                if row.get(id_col, "").split()[0].split("|")[0] == rec_id:
                    rows.append(row)

        if rows:
            out_buf = io.StringIO()
            writer = csv.DictWriter(out_buf, fieldnames=list(rows[0].keys()), delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            return out_buf.getvalue()
    # give up: return as-is
    return text


def handle_one_fasta(fasta: Path, out_tsv: Path):
    rec_id, seq = read_first_fasta_record(fasta)
    tmp_dir = out_tsv.parent / "_tmp_inputs"
    tmp_txt = write_id_seq_txt(tmp_dir, rec_id, seq)

    start = time.time()
    proc = run_predict(tmp_txt)

    # Try: if tool prints a CSV/TSV to stdout (rare), capture it.
    if proc.returncode == 0 and any(k in (proc.stdout or "").lower() for k in ("residue", "rsa", "ss3", "ss8", "disorder", "phi", "psi")):
        filtered = filter_rows_for_id(proc.stdout, rec_id)
        out_tsv.write_text(filtered, encoding="utf-8")
        print(f"[OK] saved (stdout-parsed): {out_tsv}")
        return True

    # Otherwise search filesystem for newly-created result tables
    new_files = list_new_results(start)
    for cand in new_files:
        try:
            text = cand.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(key in text.lower() for key in ("residue", "rsa", "ss3", "ss8", "disorder", "phi", "psi")):
            filtered = filter_rows_for_id(text, rec_id)
            out_tsv.write_text(filtered, encoding="utf-8")
            print(f"[OK] saved (picked file): {out_tsv}  <-  {cand}")
            return True

    # If still nothing, persist logs so you can inspect
    out_tsv.write_text(proc.stdout or "", encoding="utf-8")
    print(f"[WARN] no structured table found; wrote logs to {out_tsv}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Batch NetSurfP-3.0 predictions (one output file per FASTA).")
    ap.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = ap.parse_args()

    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Missing config.yml: {CFG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model_best.pth: {MODEL_PATH}")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in (".fa", ".fasta", ".fsa")]
    if not inputs:
        raise SystemExit(f"No FASTA files found in {in_dir}")

    print(f"[INFO] Using config: {CFG_PATH}")
    print(f"[INFO] Using model : {MODEL_PATH}")
    print(f"[INFO] Inputs      : {len(inputs)} file(s)")
    print(f"[INFO] Output dir  : {out_dir}")

    for fp in inputs:
        out_tsv = out_dir / (fp.stem + ".tsv")
        print(f"\n=== {fp.name} -> {out_tsv.name} ===")
        handle_one_fasta(fp, out_tsv)


if __name__ == "__main__":
    main()
