"""
compute_metapaths.py
====================
Offline pre-computation script for MPMAT meta-path proximity matrices.
Run ONCE before training. Outputs are stored in --out_dir and loaded at
training time by pardata.py.

Paper sections implemented:
  Section II-C : Protein-protein similarity (Needleman-Wunsch, Eq. 1)
  Section II-C : Drug-drug similarity (Tanimoto on Morgan fingerprints, Eq. 2)
  Section II-C : Threshold τ via 5-fold CV grid search (Eq. 3)
  Section II-D : PT  = Y · S'_P                              (Eq. 4)
               : PD  = S'_D · Y                              (Eq. 6)
               : PDT = S'_D · Y · S'_P                      (Eq. 8)

Usage:
  cd model-main/
  python compute_metapaths.py \
      --protein_file protein_train.csv \
      --drug_file    morgan_train.csv  \
      --train_file   train.csv         \
      --out_dir      metapath_cache    \
      --tau_grid 0.0 0.05 0.1 0.2 0.3 0.4 0.5 \
      --n_folds  5                     \
      --n_workers 4
"""

import os
import argparse
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Protein-Protein Similarity  (Section II-C-1, Eq. 1)
# ──────────────────────────────────────────────────────────────────────────────

def _nw_score_pair(args):
    """Worker: compute normalised Needleman-Wunsch similarity for one pair."""
    si, sj = args
    try:
        from Bio import pairwise2
        alignments = pairwise2.align.globalxx(si, sj, score_only=True)
        self_i     = pairwise2.align.globalxx(si, si, score_only=True)
        self_j     = pairwise2.align.globalxx(sj, sj, score_only=True)
        denom      = max(self_i, self_j)
        return float(alignments) / float(denom) if denom > 0 else 0.0
    except Exception:
        return 0.0


def compute_protein_similarity(sequences, n_workers=4, verbose=True):
    """
    Compute symmetric NW similarity matrix S_P ∈ R^{Np × Np}.

    Parameters
    ----------
    sequences : list of str
    n_workers : int
    verbose   : bool

    Returns
    -------
    SP : np.ndarray  shape (Np, Np)
    """
    from multiprocessing import Pool

    Np = len(sequences)
    SP = np.eye(Np, dtype=np.float32)          # diagonal = 1 (self-score / self-score)

    # Build upper-triangle pairs
    pairs = [(i, j) for i in range(Np) for j in range(i + 1, Np)]
    args  = [(sequences[i], sequences[j]) for i, j in pairs]

    if verbose:
        print(f"  Computing protein similarity: {Np} proteins, "
              f"{len(pairs):,} pairs, {n_workers} workers ...")
        t0 = time.time()

    with Pool(processes=n_workers) as pool:
        scores = pool.map(_nw_score_pair, args)

    for (i, j), s in zip(pairs, scores):
        SP[i, j] = s
        SP[j, i] = s

    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    return SP


# ──────────────────────────────────────────────────────────────────────────────
# 2. Drug-Drug Similarity  (Section II-C-2, Eq. 2, Algorithm 1)
# ──────────────────────────────────────────────────────────────────────────────

def _morgan_fp(smiles, radius=2, n_bits=2048):
    """Return a 1-D binary numpy array (ECFP4 fingerprint)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.uint8)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.uint8)
    except Exception:
        return np.zeros(n_bits, dtype=np.uint8)


def _tanimoto_chunk(F, i_start, j_start, chunk_i, chunk_j):
    """
    Vectorised Tanimoto for sub-blocks.
    Returns float32 matrix of shape (chunk_i, chunk_j).
    Tanimoto(Fi, Fj) = |Fi ∩ Fj| / |Fi ∪ Fj|  (Eq. 2)
    """
    A = F[i_start:i_start + chunk_i].astype(np.float32)   # (ci, m)
    B = F[j_start:j_start + chunk_j].astype(np.float32)   # (cj, m)
    inter = A @ B.T                                         # (ci, cj)
    sum_a = A.sum(axis=1, keepdims=True)                   # (ci, 1)
    sum_b = B.sum(axis=1, keepdims=True)                   # (cj, 1)
    union = sum_a + sum_b.T - inter                        # (ci, cj)
    union = np.where(union == 0, 1.0, union)               # avoid /0
    return (inter / union).astype(np.float32)


def compute_drug_similarity(smiles_list, chunk_size=500, verbose=True):
    """
    Compute symmetric Tanimoto similarity matrix S_D ∈ [0,1]^{Nd × Nd}.

    Parameters
    ----------
    smiles_list : list of str
    chunk_size  : int   – process drugs in blocks to save memory
    verbose     : bool

    Returns
    -------
    SD : np.ndarray  shape (Nd, Nd)
    """
    Nd = len(smiles_list)
    if verbose:
        print(f"  Computing Morgan fingerprints for {Nd} drugs ...")
        t0 = time.time()

    F = np.stack([_morgan_fp(s) for s in smiles_list])    # (Nd, 2048)

    if verbose:
        print(f"  Fingerprints done in {time.time()-t0:.1f}s. "
              f"Computing Tanimoto matrix ...")
        t0 = time.time()

    SD = np.eye(Nd, dtype=np.float32)
    for i in range(0, Nd, chunk_size):
        ci = min(chunk_size, Nd - i)
        for j in range(i, Nd, chunk_size):
            cj = min(chunk_size, Nd - j)
            block = _tanimoto_chunk(F, i, j, ci, cj)
            SD[i:i + ci, j:j + cj] = block
            if j != i:
                SD[j:j + cj, i:i + ci] = block.T

    if verbose:
        print(f"  Tanimoto done in {time.time()-t0:.1f}s")

    return SD


# ──────────────────────────────────────────────────────────────────────────────
# 3. Adjacency Matrix  (Section II-C-3)
# ──────────────────────────────────────────────────────────────────────────────

def build_adjacency(train_dti_path, drug_ids, prot_ids, verbose=True):
    """
    Build binary adjacency matrix Y ∈ {0,1}^{Nd × Nt} from TRAINING pairs only.
    Rows = drugs (drug_ids order), Cols = proteins (prot_ids order).
    """
    dti = pd.read_csv(train_dti_path)
    drug_col   = 'DrugID'
    prot_col   = 'ProteinID'
    label_col  = 'Label'

    drug_idx = {d: i for i, d in enumerate(drug_ids)}
    prot_idx = {p: i for i, p in enumerate(prot_ids)}

    Nd, Nt = len(drug_ids), len(prot_ids)
    Y = np.zeros((Nd, Nt), dtype=np.float32)

    for _, row in dti.iterrows():
        d, p, y = row[drug_col], row[prot_col], int(row[label_col])
        if y == 1 and d in drug_idx and p in prot_idx:
            Y[drug_idx[d], prot_idx[p]] = 1.0

    if verbose:
        print(f"  Adjacency matrix Y: shape={Y.shape}, "
              f"interactions={int(Y.sum())}")
    return Y


# ──────────────────────────────────────────────────────────────────────────────
# 4. Threshold selection via 5-fold CV  (Section II-C-3, Eq. 3)
# ──────────────────────────────────────────────────────────────────────────────

def cv_select_threshold(train_dti_path, SP, SD, drug_ids, prot_ids,
                        tau_grid, n_folds=5, verbose=True):
    """
    Select threshold τ by 5-fold CV on training pairs.
    For each fold, build Y_train, compute PT/PD/PDT, then score
    held-out pairs using the row-sum of their meta-path vector.
    Returns the τ that maximises mean validation AUC.
    """
    from sklearn.metrics import roc_auc_score

    dti = pd.read_csv(train_dti_path)
    drug_col  = 'DrugID'
    prot_col  = 'ProteinID'
    label_col = 'Label'

    drug_idx = {d: i for i, d in enumerate(drug_ids)}
    prot_idx = {p: i for i, p in enumerate(prot_ids)}

    # Keep only pairs whose DrugID and ProteinID are known
    mask = dti[drug_col].isin(drug_idx) & dti[prot_col].isin(prot_idx)
    dti  = dti[mask].reset_index(drop=True)

    if verbose:
        print(f"  CV threshold search over {tau_grid} with {n_folds} folds ...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    tau_aucs = {tau: [] for tau in tau_grid}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dti)):
        train_rows = dti.iloc[train_idx]
        val_rows   = dti.iloc[val_idx]

        Nd, Nt = len(drug_ids), len(prot_ids)
        Y_fold = np.zeros((Nd, Nt), dtype=np.float32)
        for _, row in train_rows.iterrows():
            d, p, y = row[drug_col], row[prot_col], int(row[label_col])
            if y == 1:
                Y_fold[drug_idx[d], prot_idx[p]] = 1.0

        for tau in tau_grid:
            SP_t = np.where(SP >= tau, SP, 0.0)
            SD_t = np.where(SD >= tau, SD, 0.0)
            # normalise to [0,1]
            m_p = SP_t.max(); m_d = SD_t.max()
            SP_t = SP_t / m_p if m_p > 0 else SP_t
            SD_t = SD_t / m_d if m_d > 0 else SD_t

            PT  = Y_fold @ SP_t          # (Nd, Nt)
            PD  = SD_t  @ Y_fold         # (Nd, Nt)
            PDT = SD_t  @ Y_fold @ SP_t  # (Nd, Nt)

            scores, labels = [], []
            for _, row in val_rows.iterrows():
                di = drug_idx[row[drug_col]]
                pi = prot_idx[row[prot_col]]
                score = float(PT[di, pi] + PD[di, pi] + PDT[di, pi])
                scores.append(score)
                labels.append(int(row[label_col]))

            if len(set(labels)) < 2:
                tau_aucs[tau].append(0.5)
            else:
                try:
                    tau_aucs[tau].append(roc_auc_score(labels, scores))
                except Exception:
                    tau_aucs[tau].append(0.5)

    mean_aucs = {tau: np.mean(v) for tau, v in tau_aucs.items()}
    if verbose:
        for tau, auc in sorted(mean_aucs.items()):
            print(f"    τ={tau:.2f}  mean_AUC={auc:.4f}")

    best_tau = max(mean_aucs, key=mean_aucs.get)
    if verbose:
        print(f"  → Best τ = {best_tau}  (mean_AUC={mean_aucs[best_tau]:.4f})")
    return best_tau


# ──────────────────────────────────────────────────────────────────────────────
# 5. Meta-path matrices  (Section II-D, Eqs. 4, 6, 8)
# ──────────────────────────────────────────────────────────────────────────────

def compute_metapath_matrices(Y, SP, SD, tau, verbose=True):
    """
    Apply threshold τ, normalise, then compute:
      PT  = Y  · S'_P                       (Eq. 4)
      PD  = S'_D · Y                        (Eq. 6)
      PDT = S'_D · Y · S'_P                (Eq. 8)

    Parameters
    ----------
    Y   : np.ndarray  (Nd, Nt)  binary adjacency (training only)
    SP  : np.ndarray  (Nt, Nt)  protein similarity
    SD  : np.ndarray  (Nd, Nd)  drug similarity
    tau : float

    Returns
    -------
    PT, PD, PDT : each np.ndarray (Nd, Nt)
    SP_t, SD_t  : thresholded-normalised matrices
    """
    # Threshold (Eq. 3)
    SP_t = np.where(SP >= tau, SP, 0.0).astype(np.float32)
    SD_t = np.where(SD >= tau, SD, 0.0).astype(np.float32)

    # Normalise to [0,1] by dividing by max absolute value (Section II-C-3)
    m_p = SP_t.max(); m_d = SD_t.max()
    SP_t = SP_t / m_p if m_p > 0 else SP_t
    SD_t = SD_t / m_d if m_d > 0 else SD_t

    if verbose:
        print(f"  S'_P nnz={np.count_nonzero(SP_t)}, "
              f"S'_D nnz={np.count_nonzero(SD_t)}")

    # Meta-path matrices (matrix multiplications)
    if verbose:
        print("  Computing PT  = Y · S'_P ...")
    PT  = Y @ SP_t                    # (Nd, Nt)

    if verbose:
        print("  Computing PD  = S'_D · Y ...")
    PD  = SD_t @ Y                    # (Nd, Nt)

    if verbose:
        print("  Computing PDT = S'_D · Y · S'_P ...")
    PDT = SD_t @ Y @ SP_t             # (Nd, Nt)

    if verbose:
        print(f"  PT  range=[{PT.min():.4f}, {PT.max():.4f}]")
        print(f"  PD  range=[{PD.min():.4f}, {PD.max():.4f}]")
        print(f"  PDT range=[{PDT.min():.4f}, {PDT.max():.4f}]")

    return PT, PD, PDT, SP_t, SD_t


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main driver
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute MPMAT meta-path proximity matrices.')
    parser.add_argument('--protein_file', default='protein_train.csv')
    parser.add_argument('--drug_file',    default='morgan_train.csv')
    parser.add_argument('--train_file',   default='train.csv')
    parser.add_argument('--out_dir',      default='metapath_cache')
    parser.add_argument('--tau_grid',     nargs='+', type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--n_folds',      type=int, default=5)
    parser.add_argument('--n_workers',    type=int, default=4)
    parser.add_argument('--skip_cv',      action='store_true',
                        help='Skip CV; use tau=0.0')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load drug and protein tables ─────────────────────────────────────────
    print("\n[1/6] Loading drug and protein data ...")
    drug_df = pd.read_csv(args.drug_file, dtype={'morgan_fp': str})
    prot_df = pd.read_csv(args.protein_file)

    drug_ids  = drug_df['DrugID'].tolist()
    prot_ids  = prot_df['ProteinID'].tolist()
    smiles    = drug_df['SMILES'].tolist()
    sequences = prot_df['Target_Sequence'].tolist()

    print(f"  Drugs={len(drug_ids)}, Proteins={len(prot_ids)}")

    # ── Protein similarity matrix  ───────────────────────────────────────────
    sp_path = os.path.join(args.out_dir, 'SP.npy')
    if os.path.exists(sp_path):
        print("\n[2/6] Loading cached SP ...")
        SP = np.load(sp_path)
    else:
        print("\n[2/6] Computing protein-protein similarity matrix (NW) ...")
        SP = compute_protein_similarity(sequences, n_workers=args.n_workers)
        np.save(sp_path, SP)
        print(f"  Saved → {sp_path}")

    # ── Drug similarity matrix  ──────────────────────────────────────────────
    sd_path = os.path.join(args.out_dir, 'SD.npy')
    if os.path.exists(sd_path):
        print("\n[3/6] Loading cached SD ...")
        SD = np.load(sd_path)
    else:
        print("\n[3/6] Computing drug-drug similarity matrix (Tanimoto) ...")
        SD = compute_drug_similarity(smiles)
        np.save(sd_path, SD)
        print(f"  Saved → {sd_path}")

    # ── Adjacency matrix  ────────────────────────────────────────────────────
    print("\n[4/6] Building adjacency matrix Y (training pairs only) ...")
    Y = build_adjacency(args.train_file, drug_ids, prot_ids)

    # ── Threshold selection  ─────────────────────────────────────────────────
    print("\n[5/6] Selecting threshold τ ...")
    if args.skip_cv:
        best_tau = 0.0
        print(f"  Skipped CV → τ=0.0")
    else:
        best_tau = cv_select_threshold(
            args.train_file, SP, SD, drug_ids, prot_ids,
            tau_grid=args.tau_grid, n_folds=args.n_folds)

    # ── Meta-path matrices  ──────────────────────────────────────────────────
    print("\n[6/6] Computing meta-path matrices (PT, PD, PDT) ...")
    PT, PD, PDT, SP_t, SD_t = compute_metapath_matrices(
        Y, SP, SD, tau=best_tau)

    # ── Save outputs  ────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, 'PT.npy'),  PT)
    np.save(os.path.join(args.out_dir, 'PD.npy'),  PD)
    np.save(os.path.join(args.out_dir, 'PDT.npy'), PDT)

    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
    prot_to_idx = {p: i for i, p in enumerate(prot_ids)}
    with open(os.path.join(args.out_dir, 'drug_to_idx.pkl'), 'wb') as f:
        pickle.dump(drug_to_idx, f)
    with open(os.path.join(args.out_dir, 'prot_to_idx.pkl'), 'wb') as f:
        pickle.dump(prot_to_idx, f)

    meta_config = {
        'Nd': len(drug_ids),
        'Nt': len(prot_ids),
        'metapath_len': 3 * len(prot_ids),
        'best_tau': best_tau,
        'drug_ids': drug_ids,
        'prot_ids': prot_ids,
    }
    with open(os.path.join(args.out_dir, 'meta_config.pkl'), 'wb') as f:
        pickle.dump(meta_config, f)

    print(f"\n✓ All outputs saved to '{args.out_dir}/'")
    print(f"  metapath_len (3·Nt) = {meta_config['metapath_len']}")
    print(f"  best τ = {best_tau}")
    print(f"  PT.npy  shape={PT.shape}")
    print(f"  PD.npy  shape={PD.shape}")
    print(f"  PDT.npy shape={PDT.shape}")
    print("\nDone. You can now run main.py with --metapath-cache-dir", args.out_dir)


if __name__ == '__main__':
    main()
