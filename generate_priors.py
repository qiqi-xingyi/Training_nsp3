# --*-- coding:utf-8 --*--
# @time: 9/3/25 11:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: generate_priors.py
#
# Generate priors locally with NetSurfP-3 using our wrapper (Python 3.9).
# - Input folder: data/seqs_len10/*.fasta (format: '>ID' + one-line/multi-line sequence)
# - Outputs: runs/<RUN_ID>/inputs/*.fasta and runs/<RUN_ID>/raw/<seq_id>/results.csv|json

from __future__ import annotations
from pathlib import Path
import argparse
import time
import sys
from typing import List, Optional

# import our wrapper package (make sure 'nsp3_client/__init__.py' exists)
from nsp3_client import SequenceReader, LocalNSP3Client, Dataset

def project_root() -> Path:
    """Return the project root assuming this file lives under the project root."""
    return Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local NetSurfP-3 predictions via wrapper.")
    p.add_argument("--input_folder", type=str, default=None,
                   help="Folder containing *.fasta or a CSV with [id,seq]. If omitted, use data/seqs_len10.")
    p.add_argument("--repo_root", type=str, default=None,
                   help="NetSurfP-3 repo root (contains experiments/ and models/). If omitted, use ./NetSurfP-3.0")
    p.add_argument("--config_rel", type=str,
                   default="experiments/netsurfp_3/CNNbLSTM_ESM1b_v2/config.yml",
                   help="Path to config.yml relative to repo_root.")
    p.add_argument("--model_rel", type=str,
                   default="models/esm1b_t33_650M_UR50S.pt",
                   help="Path to model checkpoint relative to repo_root.")
    p.add_argument("--run_prefix", type=str, default="exp_local",
                   help="Run folder prefix under ./runs/")
    p.add_argument("--only_ids", type=str, default="",
                   help="Comma-separated subset of IDs to run (e.g., '1bai,1dhj').")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit the number of records to run (0 = no limit).")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Device for NSP3 subprocess (via env).")
    p.add_argument("--no_amp", action="store_true", default=True,
                   help="Keep safer precision path (default True).")
    p.add_argument("--min_len", type=int, default=3, help="Minimum sequence length after sanitization.")
    p.add_argument("--max_len", type=int, default=10000, help="Maximum allowed length after sanitization.")
    p.add_argument("--dedup", action="store_true", default=True, help="Deduplicate identical (id, seq) pairs.")
    return p.parse_args()

def filter_dataset_by_ids(ds: Dataset, id_list: Optional[List[str]]) -> Dataset:
    if not id_list:
        return ds
    ids = set([x.strip() for x in id_list if x.strip()])
    recs = [r for r in ds.records if r.seq_id in ids]
    return Dataset(records=recs)

def main() -> None:
    args = parse_args()
    root = project_root()

    # --------- 1) INPUTS ----------
    if args.input_folder:
        input_folder = Path(args.input_folder).resolve()
    else:
        input_folder = (root / "data" / "seqs_len10").resolve()

    if not input_folder.exists():
        print(f"[ERROR] Input folder not found: {input_folder}")
        sys.exit(1)

    # --------- 2) OUTPUT ROOT -----
    run_id = f"{args.run_prefix}_" + time.strftime("%m%d_%H%M%S")
    out_root = (root / "runs" / run_id).resolve()
    inputs_dir = out_root / "inputs"
    raw_dir = out_root / "raw"

    # --------- 3) NetSurfP-3 config -----
    repo_root = Path(args.repo_root).resolve() if args.repo_root else (root / "NetSurfP-3.0").resolve()
    config_yml = (repo_root / args.config_rel).resolve()
    model_pth = (repo_root / args.model_rel).resolve()

    # Sanity checks for config/model
    for p in [config_yml, model_pth]:
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    # --------- 4) Read sequences & emit FASTA to inputs_dir ----------
    reader = SequenceReader(min_len=args.min_len, dedup=args.dedup, max_len=args.max_len)
    pairs = reader.load(input_folder)  # returns List[(seq_id, sanitized_seq)]

    # Optional: filter by --only_ids
    if args.only_ids.strip():
        wanted = [x.strip() for x in args.only_ids.split(",") if x.strip()]
        pairs = [p for p in pairs if p[0] in wanted]

    # Optional: limit
    if args.limit and args.limit > 0:
        pairs = pairs[:args.limit]

    dataset = reader.save_fasta(pairs, inputs_dir)
    print(f"[INFO] Prepared {len(dataset.records)} FASTA files at: {inputs_dir}")

    if len(dataset.records) == 0:
        print("[ERROR] No records to run after filtering/sanitization.")
        sys.exit(1)

    # --------- 5) Run local NetSurfP-3 ----------
    # Tip: start with CPU for 1–3 samples to validate inputs, then switch to CUDA.
    client = LocalNSP3Client(
        config_path=config_yml,
        model_ckpt=model_pth,
        predictor="SecondaryFeatures",
        repo_root=repo_root,     # make relative paths in YAML resolvable
        overwrite=True,
        verbose=True,
        device=args.device,
        no_amp=args.no_amp,
    )

    print(f"[STEP] Submitting {len(dataset.records)} jobs to local NetSurfP-3 ...")
    meta = client.predict(dataset, raw_dir)

    # --------- 6) Report ----------
    print(f"[DONE] {meta.get('success', 0)} succeeded, {len(meta.get('failures', []))} failed.")
    print(f"[OUT] Raw outputs: {raw_dir}")
    print(f"[OUT] Run index:   {raw_dir / 'raw_index.json'}")

if __name__ == "__main__":
    main()
