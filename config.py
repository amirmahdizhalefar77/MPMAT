"""
config.py
=========
Shared configuration, path helpers, and argument parser for the
MPMAT meta-path pre-computation pipeline.

Every step script imports from here so that all file paths and
hyperparameters are defined in a single place.
"""

import os
import argparse
import pickle


# ──────────────────────────────────────────────────────────────────────────────
# Canonical file names inside out_dir
# ──────────────────────────────────────────────────────────────────────────────

FILE_SP          = "SP.npy"
FILE_SD          = "SD.npy"
FILE_Y           = "Y.npy"
FILE_PT          = "PT.npy"
FILE_PD          = "PD.npy"
FILE_PDT         = "PDT.npy"
FILE_DRUG_IDX    = "drug_to_idx.pkl"
FILE_PROT_IDX    = "prot_to_idx.pkl"
FILE_META_CONFIG = "meta_config.pkl"
FILE_BEST_TAU    = "best_tau.pkl"       # written by step4, read by step5


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def cache_path(out_dir: str, filename: str) -> str:
    """Return the full path for a cache file inside out_dir."""
    return os.path.join(out_dir, filename)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ──────────────────────────────────────────────────────────────────────────────
# Shared argument parser (each step adds its own flags on top)
# ──────────────────────────────────────────────────────────────────────────────

def base_parser(description: str) -> argparse.ArgumentParser:
    """
    Return a parser pre-loaded with arguments that are common across
    all pipeline steps.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--protein_file", default="protein_train.csv",
        help="CSV with columns ProteinID, Target_Sequence",
    )
    p.add_argument(
        "--drug_file", default="morgan_train.csv",
        help="CSV with columns DrugID, SMILES",
    )
    p.add_argument(
        "--train_file", default="train.csv",
        help="CSV with columns DrugID, ProteinID, Label",
    )
    p.add_argument(
        "--out_dir", default="metapath_cache",
        help="Directory where all outputs are cached",
    )
    return p
