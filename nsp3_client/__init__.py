# nsp3_client/__init__.py
# Python 3.9 compatible wrappers around local NetSurfP-3 (nsp3) via subprocess.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import csv
import os
import time
import json
import subprocess
import shutil

# -----------------------------
# Simple data containers
# -----------------------------

@dataclass
class SeqRecord:
    seq_id: str
    fasta_path: Path

@dataclass
class Dataset:
    records: List[SeqRecord]

# -----------------------------
# Reader for FASTA / CSV with strict sanitization
# -----------------------------

class SequenceReader:
    """Load sequences and enforce strict sanitization to 20AA (ACDEFGHIKLMNPQRSTVWY)."""

    VALID = set("ACDEFGHIKLMNPQRSTVWY")
    MAP = {"U": "C", "O": "K", "B": "D", "Z": "E", "J": "L"}  # conservative mapping

    def __init__(self, min_len: int = 3, dedup: bool = True, max_len: int = 10000):
        self.min_len = min_len
        self.dedup = dedup
        self.max_len = max_len

    # --- public API ---

    def load(self, path: Union[str, Path]) -> List[Tuple[str, str]]:
        """Return a list of (id, sanitized_seq)."""
        path = Path(path)
        if path.is_dir():
            pairs: List[Tuple[str, str]] = []
            for fp in sorted(list(path.glob("*.fa")) + list(path.glob("*.fasta"))):
                sid, seq = self._read_one_fasta(fp)
                seq = self.sanitize(seq)
                if seq and len(seq) >= self.min_len:
                    pairs.append((sid or fp.stem, seq))
            return self._maybe_dedup(pairs)
        if path.suffix.lower() == ".csv":
            return self._maybe_dedup(self._read_csv(path))
        raise ValueError(f"Unsupported input: {path}. Provide a folder of FASTA or a CSV with [id,seq].")

    def save_fasta(self, pairs: List[Tuple[str, str]], out_dir: Union[str, Path]) -> Dataset:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        recs: List[SeqRecord] = []
        for sid, seq in pairs:
            fasta_path = out_dir / f"{sid}.fasta"
            with open(fasta_path, "w") as f:
                f.write(f">{sid}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")
            recs.append(SeqRecord(seq_id=sid, fasta_path=fasta_path))
        return Dataset(records=recs)

    # --- sanitization ---

    def sanitize(self, seq: str) -> str:
        """Uppercase, map uncommon AA codes to 20AA, drop others; enforce max_len."""
        s = (seq or "").strip().upper()
        out: List[str] = []
        for ch in s:
            if ch in self.VALID:
                out.append(ch)
            elif ch in self.MAP:
                out.append(self.MAP[ch])
            # else: drop any other char (X, *, -, ., ?, whitespace, etc.)
        s2 = "".join(out)
        if not s2:
            raise ValueError("Empty sequence after sanitization")
        if len(s2) > self.max_len:
            raise ValueError(f"Sequence too long after sanitization ({len(s2)} > {self.max_len}).")
        return s2

    # --- helpers ---

    def _read_one_fasta(self, fp: Path) -> Tuple[str, str]:
        sid: Optional[str] = None
        seq_lines: List[str] = []
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if sid is None:
                        sid = line[1:].strip()
                    else:
                        # Only take the first record in a multi-FASTA file
                        break
                else:
                    seq_lines.append(line)
        raw = "".join(seq_lines).replace(" ", "").upper()
        return (sid or fp.stem), self.sanitize(raw)

    def _read_csv(self, fp: Path) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        with open(fp, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                sid = str(row[0]).strip()
                seq = str(row[1]).strip().replace(" ", "").upper() if len(row) > 1 else ""
                if not sid or not seq:
                    continue
                try:
                    seq = self.sanitize(seq)
                except Exception:
                    continue
                if len(seq) >= self.min_len:
                    pairs.append((sid, seq))
        return pairs

    def _maybe_dedup(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not self.dedup:
            return pairs
        seen: set = set()
        out: List[Tuple[str, str]] = []
        for sid, seq in pairs:
            key = (sid, seq)
            if key not in seen:
                seen.add(key)
                out.append((sid, seq))
        return out

# -----------------------------
# Local NetSurfP-3 runner (subprocess)
# -----------------------------

class LocalNSP3Client:
    """
    Execute the installed `nsp3` console script via subprocess:
        nsp3 predict -c <cfg.yml> -d <model.pth> -p SecondaryFeatures -i <in.fasta>
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        model_ckpt: Union[str, Path],
        predictor: str = "SecondaryFeatures",
        repo_root: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        verbose: bool = True,
        device: str = "cuda",   # "cuda" or "cpu"
        no_amp: bool = True     # keep kernels in safer precision paths
    ):
        self.config_path = str(Path(config_path).resolve())
        self.model_ckpt = str(Path(model_ckpt).resolve())
        self.predictor = predictor
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.overwrite = overwrite
        self.verbose = verbose
        self.device = device
        self.no_amp = no_amp

        # Find the nsp3 executable in PATH
        self.nsp3_exe = shutil.which("nsp3")
        if self.nsp3_exe is None:
            raise RuntimeError(
                "Could not find 'nsp3' executable on PATH.\n"
                "Make sure the environment where you installed NetSurfP-3.0 is active, "
                "and that 'pip install .' in NetSurfP-3.0/nsp3 succeeded."
            )

    def predict(self, dataset: Dataset, out_dir: Union[str, Path]) -> Dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        failures: List[Dict] = []
        ok_count = 0
        t0 = time.time()

        for i, rec in enumerate(dataset.records):
            sid = rec.seq_id
            fasta_path = rec.fasta_path.resolve()
            if self.verbose:
                print(f"[{i+1}/{len(dataset.records)}] NetSurfP-3 local: {sid}")

            rec_out = out_dir / self._safe(sid)
            if rec_out.exists() and not self.overwrite:
                if self.verbose:
                    print(f"  -> exists, skip (overwrite=False): {rec_out}")
                ok_count += 1
                continue
            rec_out.mkdir(parents=True, exist_ok=True)

            argv = [
                self.nsp3_exe,
                "predict",
                "-c", self.config_path,  # absolute path
                "-d", self.model_ckpt,   # absolute path
                "-p", self.predictor,
                "-i", str(fasta_path),
            ]
            # If upstream CLI supports flags like --device/--no-amp, append here.

            try:
                # Subprocess environment for better debugging & device control
                env = os.environ.copy()
                env["CUDA_LAUNCH_BLOCKING"] = "1"
                env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
                if self.device == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = "-1"
                # (no_amp is advisory; actual AMP is controlled within nsp3 codebase)

                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    cwd=str(rec_out),
                    env=env,
                    timeout=60 * 60  # one hour per sample safety timeout
                )

                if proc.returncode != 0:
                    failures.append({
                        "seq_id": sid,
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-1000:],
                        "stderr": proc.stderr[-1000:],
                    })
                    if self.verbose:
                        print(f"  -> returncode={proc.returncode}")
                        if proc.stderr:
                            print("     stderr:", proc.stderr.strip().splitlines()[-1])
                else:
                    ok_count += 1
                    if self.verbose and proc.stdout:
                        last = proc.stdout.strip().splitlines()[-1]
                        print("  ->", last)

            finally:
                pass

        meta = {
            "runner": "LocalNSP3Client(subprocess)",
            "config_path": self.config_path,
            "model_ckpt": self.model_ckpt,
            "predictor": self.predictor,
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "device": self.device,
            "no_amp": self.no_amp,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 3),
            "success": ok_count,
            "failures": failures,
        }
        (out_dir / "raw_index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        if self.verbose:
            print(f"[OK] Local runs: {ok_count} ok, {len(failures)} failed. See {out_dir/'raw_index.json'}")
        return meta

    @staticmethod
    def _safe(text: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)
