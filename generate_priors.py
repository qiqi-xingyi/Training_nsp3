# generate_priors.py
import os
import sys
import yaml
import math
import torch
import importlib
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, List

# ----- Fixed paths (your exact paths) -----
CFG_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\config.yml")
MODEL_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\model_best.pth")
INPUT_DIR = Path(r"G:\Train_NSP\data\seqs_len10")
OUTPUT_DIR = Path(r"G:\Train_NSP\result")

# ----- Imports from your package -----
# predictor you pasted lives in nsp3.predict, class SecondaryFeatures
from nsp3.predict import SecondaryFeatures  # if this import fails, adjust path to your local package

# Biopython for FASTA parsing
from Bio import SeqIO

# ------------- Utilities -------------
def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_model_class(arch_type: str):
    """
    Dynamically find a class named `arch_type` somewhere under nsp3.models.*
    """
    base_pkg = "nsp3.models"
    pkg = importlib.import_module(base_pkg)

    # try common modules first
    common_modules = [
        "architectures",
        "nets",
        "models",
        "__init__",
    ]
    tried = set()
    for mod in common_modules:
        try:
            m = importlib.import_module(f"{base_pkg}.{mod}")
            if hasattr(m, arch_type):
                return getattr(m, arch_type)
            tried.add(f"{base_pkg}.{mod}")
        except Exception:
            pass

    # exhaustive walk: import all submodules below nsp3.models
    pkg_path = Path(pkg.__file__).parent
    for py in pkg_path.rglob("*.py"):
        mod_name = f"{base_pkg}." + ".".join(py.relative_to(pkg_path).with_suffix("").parts)
        if mod_name in tried:
            continue
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, arch_type):
                return getattr(m, arch_type)
        except Exception:
            continue

    raise ImportError(f"Could not find model class '{arch_type}' in package '{base_pkg}'.")

def build_model_from_config(cfg: Dict[str, Any]) -> torch.nn.Module:
    arch_cfg = cfg.get("arch", {})
    arch_type = arch_cfg.get("type")
    arch_args = arch_cfg.get("args", {}) or {}

    if not arch_type:
        raise ValueError("Missing 'arch.type' in config.yml")

    ModelCls = find_model_class(arch_type)
    model = ModelCls(**arch_args)
    return model

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)  # allow plain state_dict as fallback
    model.load_state_dict(state_dict, strict=False)
    return ckpt

def to_device(model: torch.nn.Module) -> torch.nn.Module:
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

def parse_first_fasta(fp: Path) -> Tuple[str, str]:
    recs = list(SeqIO.parse(str(fp), "fasta"))
    if not recs:
        raise ValueError(f"No FASTA records in {fp}")
    rec = recs[0]
    return rec.id, str(rec.seq).upper()

def ensure_outdir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_tsv_header(fh, n_ss8: int, n_ss3: int, n_dis: int):
    cols = ["res_idx", "aa"]
    cols += [f"ss8_p{i}" for i in range(n_ss8)]
    cols += [f"ss3_p{i}" for i in range(n_ss3)]
    cols += [f"dis_p{i}" for i in range(n_dis)]
    cols += ["rsa", "asa", "phi", "psi"]
    fh.write("\t".join(cols) + "\n")

def safe_get(arr, i, default=None):
    try:
        return arr[i]
    except Exception:
        return default

def to_list(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return x

# ------------- Main prediction -------------
def predict_one(model: torch.nn.Module, fasta_path: Path, out_path: Path):
    seq_id, seq = parse_first_fasta(fasta_path)

    # Call your SecondaryFeatures predictor directly
    predictor = SecondaryFeatures(model=model, model_data=str(MODEL_PATH))

    # The __call__ expects a fasta path or a FASTA string; we pass the file path
    identifiers, sequences, preds = predictor(str(fasta_path))

    # `preds` shape assumptions (from your pasted code and repo):
    # preds is a list of batches -> we concatenated them per chunk in predictor,
    # so here it's a list with one element (since we passed one fasta file), but they stored as lists per chunk.
    # We will flatten them.
    flat = []
    for batch in preds:
        # batch is a list of outputs x[i] already moved to cpu and converted to numpy
        flat.append(batch)
    # `flat` is list of length 1 per chunk; but in the predictor they appended for each chunk.
    # Rebuild per-output arrays by concatenating chunks along the sequence dimension.
    # flat[k][i] -> for chunk k, output i (0..5)
    n_out = len(flat[0])  # should be 6
    merged = []
    for i in range(n_out):
        # concatenate along axis 0 (sequence length)
        parts = [safe_get(flat[k], i) for k in range(len(flat))]
        parts = [p for p in parts if p is not None]
        if len(parts) == 1:
            merged.append(parts[0])
        else:
            import numpy as np
            merged.append(np.concatenate(parts, axis=0))

    ss8 = to_list(safe_get(merged, 0))  # shape [L, 8]
    ss3 = to_list(safe_get(merged, 1))  # shape [L, 3]
    dis = to_list(safe_get(merged, 2))  # shape [L, Cdis] (likely 2)
    rsa_asa = to_list(safe_get(merged, 3))  # shape [L, 2] expected
    phi = to_list(safe_get(merged, 4))  # shape [L, 1]
    psi = to_list(safe_get(merged, 5))  # shape [L, 1]

    # Basic shape checks and fallbacks
    L = len(seq)
    def rows_or_default(arr, L, ncols):
        if arr is None:
            return [[math.nan]*ncols for _ in range(L)]
        # if array is longer (multiple sequences), take first L
        return [list(arr[i]) if i < len(arr) else [math.nan]*ncols for i in range(L)]

    n_ss8 = len(ss8[0]) if ss8 and len(ss8) > 0 and hasattr(ss8[0], "__len__") else 8
    n_ss3 = len(ss3[0]) if ss3 and len(ss3) > 0 and hasattr(ss3[0], "__len__") else 3
    n_dis = len(dis[0]) if dis and len(dis) > 0 and hasattr(dis[0], "__len__") else 2

    ss8_rows = rows_or_default(ss8, L, n_ss8)
    ss3_rows = rows_or_default(ss3, L, n_ss3)
    dis_rows = rows_or_default(dis, L, n_dis)

    # rsa/asa might be [L,2] or [L,1] (some configs); we handle both
    rsa_rows = [math.nan]*L
    asa_rows = [math.nan]*L
    if rsa_asa and len(rsa_asa) > 0:
        if hasattr(rsa_asa[0], "__len__"):
            # assume col0 = rsa, col1 = asa if available
            for i in range(L):
                if i < len(rsa_asa):
                    row = rsa_asa[i]
                    rsa_rows[i] = float(row[0]) if len(row) >= 1 else math.nan
                    asa_rows[i] = float(row[1]) if len(row) >= 2 else math.nan
        else:
            # single value per residue -> treat as rsa
            for i in range(L):
                rsa_rows[i] = float(rsa_asa[i]) if i < len(rsa_asa) else math.nan

    # phi/psi are [L,1]; flatten safely
    phi_rows = [float(phi[i][0]) if phi and i < len(phi) and hasattr(phi[i], "__len__") else (float(phi[i]) if phi and i < len(phi) else math.nan) for i in range(L)]
    psi_rows = [float(psi[i][0]) if psi and i < len(psi) and hasattr(psi[i], "__len__") else (float(psi[i]) if psi and i < len(psi) else math.nan) for i in range(L)]

    # Write TSV
    with out_path.open("w", encoding="utf-8") as fh:
        write_tsv_header(fh, n_ss8, n_ss3, n_dis)
        for i in range(L):
            aa = seq[i]
            row = [str(i+1), aa]  # 1-based index
            row += [f"{v:.6f}" for v in ss8_rows[i]]
            row += [f"{v:.6f}" for v in ss3_rows[i]]
            row += [f"{v:.6f}" for v in dis_rows[i]]
            row += [f"{rsa_rows[i]:.6f}" if not math.isnan(rsa_rows[i]) else "nan"]
            row += [f"{asa_rows[i]:.6f}" if not math.isnan(asa_rows[i]) else "nan"]
            row += [f"{phi_rows[i]:.6f}" if not math.isnan(phi_rows[i]) else "nan"]
            row += [f"{psi_rows[i]:.6f}" if not math.isnan(psi_rows[i]) else "nan"]
            fh.write("\t".join(row) + "\n")

def main():
    # Validate paths
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"config.yml not found: {CFG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model_best.pth not found: {MODEL_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    ensure_outdir()

    # Build model from config and load checkpoint
    cfg = load_yaml(CFG_PATH)
    model = build_model_from_config(cfg)
    load_checkpoint_into_model(model, MODEL_PATH)
    model = to_device(model)

    # Iterate FASTA files
    fasta_files = [p for p in sorted(INPUT_DIR.iterdir()) if p.is_file() and p.suffix.lower() in (".fa", ".fasta", ".fsa")]
    if not fasta_files:
        raise SystemExit(f"No FASTA files found in {INPUT_DIR}")

    print(f"[INFO] Using config : {CFG_PATH}")
    print(f"[INFO] Using model  : {MODEL_PATH}")
    print(f"[INFO] Input files  : {len(fasta_files)}")
    print(f"[INFO] Output folder: {OUTPUT_DIR}")

    for fp in fasta_files:
        out_path = OUTPUT_DIR / (fp.stem + ".tsv")
        try:
            print(f"\n=== {fp.name} -> {out_path.name} ===")
            predict_one(model, fp, out_path)
            print(f"[OK] saved: {out_path}")
        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
