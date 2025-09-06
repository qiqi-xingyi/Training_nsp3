# generate_priors.py
import os
import sys
import math
import yaml
import torch
import importlib
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from Bio import SeqIO
from nsp3.predict import SecondaryFeatures  # predictor class you shared

# ---- Fixed paths (your exact paths) ----
CFG_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\config.yml")
MODEL_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\model_best.pth")
INPUT_DIR = Path(r"G:\Train_NSP\data\seqs_len10")
OUTPUT_DIR = Path(r"G:\Train_NSP\result")

# ---------------- I/O helpers ----------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_outdir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_first_fasta(fp: Path) -> Tuple[str, str]:
    recs = list(SeqIO.parse(str(fp), "fasta"))
    if not recs:
        raise ValueError(f"No FASTA records in {fp}")
    rec = recs[0]
    return rec.id, str(rec.seq).upper()

def write_tsv_header(fh, n_ss8: int, n_ss3: int, n_dis: int):
    cols = ["res_idx", "aa"]
    cols += [f"ss8_p{i}" for i in range(n_ss8)]
    cols += [f"ss3_p{i}" for i in range(n_ss3)]
    cols += [f"dis_p{i}" for i in range(n_dis)]
    cols += ["rsa", "asa", "phi", "psi"]
    fh.write("\t".join(cols) + "\n")

# ---------------- model loading ----------------
def find_model_class(arch_type: str):
    """
    Dynamically locate the model class `arch_type` under nsp3.models.*
    """
    base_pkg = "nsp3.models"
    import pkgutil

    # import nsp3.models to get its path
    base = importlib.import_module(base_pkg)
    base_path = Path(base.__file__).parent

    # quick probes
    for mod in ("architectures", "nets", "models", "__init__"):
        try:
            m = importlib.import_module(f"{base_pkg}.{mod}")
            if hasattr(m, arch_type):
                return getattr(m, arch_type)
        except Exception:
            pass

    # exhaustive walk through submodules
    for module_info in pkgutil.walk_packages([str(base_path)], prefix=f"{base_pkg}."):
        name = module_info.name
        try:
            m = importlib.import_module(name)
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
    state_dict = ckpt.get("state_dict", ckpt)  # allow raw state_dict as fallback
    model.load_state_dict(state_dict, strict=False)
    return ckpt

def to_device_and_eval(model: torch.nn.Module) -> torch.nn.Module:
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

# ---------------- output normalization ----------------
def npify(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None

def squeeze_leading_ones(a: np.ndarray) -> np.ndarray:
    # drop any number of leading singleton dims: [1,L,C] -> [L,C], [1,1,L,C] -> [L,C]
    while a.ndim >= 2 and a.shape[0] == 1:
        a = a[0]
    return a

def normalize_to_LC(arr, L: int, default_C: int) -> np.ndarray:
    """
    Convert an array-like into shape [L, C].
    Handles inputs like [1,L,C], [L,C,1], [L], [1,L], [L,1], [L,C,K=1], etc.
    If fewer than L rows, pad with NaNs; if more, truncate to L.
    """
    a = npify(arr)
    if a is None:
        return np.full((L, default_C), np.nan, dtype=float)

    a = squeeze_leading_ones(a)
    a = np.squeeze(a)  # remove trailing singleton dims if any

    if a.ndim == 1:
        a = a.reshape((-1, 1))
    elif a.ndim > 2:
        last = a.shape[-1]
        a = a.reshape((-1, last))

    # now a is 2D: [N, C]
    N, C = a.shape
    if N > L:
        a = a[:L]
    elif N < L:
        pad = np.full((L - N, C), np.nan, dtype=float)
        a = np.vstack([a, pad])
    return a

def infer_channel_count(arr, fallback: int) -> int:
    a = npify(arr)
    if a is None:
        return fallback
    a = squeeze_leading_ones(a)
    a = np.squeeze(a)
    if a.ndim == 1:
        return 1
    try:
        return int(a.shape[1])
    except Exception:
        return fallback

# ---------------- prediction per FASTA ----------------
def predict_one(model: torch.nn.Module, fasta_path: Path, out_path: Path):
    # parse once to get length and residue letters (for 'aa' column)
    seq_id, seq = parse_first_fasta(fasta_path)
    L = len(seq)

    # call the predictor; it returns (identifier, sequence, prediction)
    predictor = SecondaryFeatures(model=model, model_data=str(MODEL_PATH))
    identifiers, sequences, preds = predictor(str(fasta_path))
    # preds is chunked by the predictor; flatten per head and vstack along length
    flat_chunks: List[List[np.ndarray]] = []
    for chunk in preds:
        flat_chunks.append(chunk)  # each chunk: list of heads [0..5]

    if not flat_chunks:
        raise RuntimeError("Predictor returned no outputs.")

    n_heads = len(flat_chunks[0])  # expected 6
    merged: List[np.ndarray] = []
    for i in range(n_heads):
        parts = [npify(flat_chunks[k][i]) for k in range(len(flat_chunks)) if flat_chunks[k][i] is not None]
        if not parts:
            merged.append(None)
            continue
        normed = []
        for p in parts:
            p = squeeze_leading_ones(p)
            p = np.squeeze(p)
            if p.ndim == 1:
                p = p.reshape((-1, 1))
            elif p.ndim > 2:
                last = p.shape[-1]
                p = p.reshape((-1, last))
            normed.append(p)
        merged.append(np.vstack(normed))

    # heads (best-effort semantics based on repo):
    # 0: ss8 probabilities [L,8]
    # 1: ss3 probabilities [L,3]
    # 2: disorder probabilities [L,?]
    # 3: rsa/asa (continuous) [L,1 or 2]
    # 4: phi [L,1]
    # 5: psi [L,1]
    ss8 = merged[0] if len(merged) > 0 else None
    ss3 = merged[1] if len(merged) > 1 else None
    dis = merged[2] if len(merged) > 2 else None
    rsa_asa = merged[3] if len(merged) > 3 else None
    phi = merged[4] if len(merged) > 4 else None
    psi = merged[5] if len(merged) > 5 else None

    # infer channel counts, then normalize all to [L, C]
    n_ss8 = infer_channel_count(ss8, 8)
    n_ss3 = infer_channel_count(ss3, 3)
    n_dis = infer_channel_count(dis, 2)

    ss8 = normalize_to_LC(ss8, L, n_ss8)
    ss3 = normalize_to_LC(ss3, L, n_ss3)
    dis = normalize_to_LC(dis, L, n_dis)

    rsa_asa = normalize_to_LC(rsa_asa, L, 2)  # if only one channel present, treat as RSA
    if rsa_asa.shape[1] == 1:
        rsa = rsa_asa[:, 0]
        asa = np.full((L,), np.nan, dtype=float)
    else:
        rsa = rsa_asa[:, 0]
        asa = rsa_asa[:, 1]

    phi = normalize_to_LC(phi, L, 1)[:, 0]
    psi = normalize_to_LC(psi, L, 1)[:, 0]

    # write TSV
    with out_path.open("w", encoding="utf-8") as fh:
        write_tsv_header(fh, n_ss8, n_ss3, n_dis)
        for i in range(L):
            row = [str(i + 1), seq[i]]
            row += [f"{v:.6f}" if not np.isnan(v) else "nan" for v in ss8[i]]
            row += [f"{v:.6f}" if not np.isnan(v) else "nan" for v in ss3[i]]
            row += [f"{v:.6f}" if not np.isnan(v) else "nan" for v in dis[i]]
            row += [f"{rsa[i]:.6f}" if not np.isnan(rsa[i]) else "nan"]
            row += [f"{asa[i]:.6f}" if not np.isnan(asa[i]) else "nan"]
            row += [f"{phi[i]:.6f}" if not np.isnan(phi[i]) else "nan"]
            row += [f"{psi[i]:.6f}" if not np.isnan(psi[i]) else "nan"]
            fh.write("\t".join(row) + "\n")

# ---------------- main ----------------
def main():
    # basic checks
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"config.yml not found: {CFG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model_best.pth not found: {MODEL_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    ensure_outdir()

    # build model and load weights
    cfg = load_yaml(CFG_PATH)
    model = build_model_from_config(cfg)
    load_checkpoint_into_model(model, MODEL_PATH)
    model = to_device_and_eval(model)

    # gather inputs
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
