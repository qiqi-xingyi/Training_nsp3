# generate_priors.py
import os
import sys
import yaml
import math
import torch
import importlib
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from nsp3.predict import SecondaryFeatures
from Bio import SeqIO

CFG_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\config.yml")
MODEL_PATH = Path(r"G:\Train_NSP\NetSurfP-3.0\saved\nsp3\CNNbLSTM_ESM1b_v2\CNNbLSTM_ESM1b_v2_trial\0905-150926\checkpoints\model_best.pth")
INPUT_DIR = Path(r"G:\Train_NSP\data\seqs_len10")
OUTPUT_DIR = Path(r"G:\Train_NSP\result")

def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_model_class(arch_type: str):
    base_pkg = "nsp3.models"
    pkg = importlib.import_module(base_pkg)
    common_modules = ["architectures", "nets", "models", "__init__"]
    tried = set()
    for mod in common_modules:
        try:
            m = importlib.import_module(f"{base_pkg}.{mod}")
            if hasattr(m, arch_type):
                return getattr(m, arch_type)
            tried.add(f"{base_pkg}.{mod}")
        except Exception:
            pass
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
    state_dict = ckpt.get("state_dict", ckpt)
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

def first_number(x):
    try:
        while isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return float("nan")
            x = x[0]
        return float(x)
    except Exception:
        return float("nan")

def row_vector(arr, i, ncols):
    vec = []
    for k in range(ncols):
        v = None
        if arr is not None and i < len(arr):
            try:
                v = arr[i][k]
            except Exception:
                v = None
        vec.append(first_number(v))
    return vec

def predict_one(model: torch.nn.Module, fasta_path: Path, out_path: Path):
    seq_id, seq = parse_first_fasta(fasta_path)
    predictor = SecondaryFeatures(model=model, model_data=str(MODEL_PATH))
    identifiers, sequences, preds = predictor(str(fasta_path))

    flat = []
    for batch in preds:
        flat.append(batch)
    n_out = len(flat[0])
    merged = []
    for i in range(n_out):
        parts = [safe_get(flat[k], i) for k in range(len(flat))]
        parts = [p for p in parts if p is not None]
        if len(parts) == 1:
            merged.append(parts[0])
        else:
            merged.append(np.concatenate(parts, axis=0))

    ss8 = to_list(safe_get(merged, 0))
    ss3 = to_list(safe_get(merged, 1))
    dis = to_list(safe_get(merged, 2))
    rsa_asa = to_list(safe_get(merged, 3))
    phi = to_list(safe_get(merged, 4))
    psi = to_list(safe_get(merged, 5))

    L = len(seq)

    def infer_cols(arr, default_n):
        try:
            if arr is not None and len(arr) > 0:
                inner = arr[0]
                if isinstance(inner, (list, tuple, np.ndarray)):
                    return len(inner)
        except Exception:
            pass
        return default_n

    n_ss8 = infer_cols(ss8, 8)
    n_ss3 = infer_cols(ss3, 3)
    n_dis = infer_cols(dis, 2)

    with out_path.open("w", encoding="utf-8") as fh:
        write_tsv_header(fh, n_ss8, n_ss3, n_dis)
        for i in range(L):
            aa = seq[i]

            ss8_row = row_vector(ss8, i, n_ss8)
            ss3_row = row_vector(ss3, i, n_ss3)
            dis_row = row_vector(dis, i, n_dis)

            rsa = first_number((rsa_asa[i] if rsa_asa is not None and i < len(rsa_asa) else None))
            # if rsa_asa provides two channels, take the second as ASA
            asa = float("nan")
            if rsa_asa is not None and i < len(rsa_asa):
                try:
                    asa = first_number(rsa_asa[i][1])
                except Exception:
                    asa = float("nan")

            phi_val = first_number(phi[i] if phi is not None and i < len(phi) else None)
            psi_val = first_number(psi[i] if psi is not None and i < len(psi) else None)

            row = [str(i+1), aa]
            row += [f"{v:.6f}" for v in ss8_row]
            row += [f"{v:.6f}" for v in ss3_row]
            row += [f"{v:.6f}" for v in dis_row]
            row += [f"{rsa:.6f}" if not math.isnan(rsa) else "nan"]
            row += [f"{asa:.6f}" if not math.isnan(asa) else "nan"]
            row += [f"{phi_val:.6f}" if not math.isnan(phi_val) else "nan"]
            row += [f"{psi_val:.6f}" if not math.isnan(psi_val) else "nan"]
            fh.write("\t".join(row) + "\n")

def main():
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"config.yml not found: {CFG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model_best.pth not found: {MODEL_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    ensure_outdir()

    cfg = load_yaml(CFG_PATH)
    model = build_model_from_config(cfg)
    load_checkpoint_into_model(model, MODEL_PATH)
    model = to_device(model)

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
