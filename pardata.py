"""
pardata.py  –  MPMAT version
============================
All original protein/drug parsing logic is UNCHANGED.
New additions (clearly marked):
  • _load_metapath_cache()        – lazy-loads pre-computed matrices
  • extract_metapath_features()   – builds per-pair meta-path vector
  • get_metapath_len()            – returns 3·Nt from cache
  • parse_data()                  – accepts optional metapath_cache_dir;
                                    returns augmented dict with key
                                    "metapath_feature" when cache is provided

The metapath feature for drug-target pair (di, tj) is the row vector:
    [PT[i_d, :], PD[i_d, :], PDT[i_d, :]]  ∈  R^{3·Nt}
This encodes the DRUG's network fingerprint (all three meta-path scores
against every target protein).  i_d is the drug's index in drug_to_idx.pkl.
"""

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from subword_nmt.apply_bpe import BPE
import tensorflow as tf

drug_col    = 'DrugID'
protein_col = 'ProteinID'
label_col   = 'Label'

# ── Original protein vocabulary ───────────────────────────────────────────────
vocab_csv = pd.read_csv('vocab.csv')
values = vocab_csv['Values'].values
protein_dict = dict(zip(values, range(1, len(values) + 1)))
print(protein_dict)

# ── Original SMILES character vocabulary ─────────────────────────────────────
csv1 = pd.read_csv('morgan_train.csv', usecols=['SMILES'])
csv2 = pd.read_csv('morgan_valid.csv', usecols=['SMILES'])
csv3 = pd.read_csv('morgan_test.csv',  usecols=['SMILES'])
final = []
for k in range(len(csv1)):
    l = [i for i in csv1['SMILES'][k]]
    final += l
for k in range(len(csv2)):
    l = [i for i in csv2['SMILES'][k]]
    final += l
for k in range(len(csv3)):
    l = [i for i in csv3['SMILES'][k]]
    final += l
kk = list(set(final))
kk_dict = {k: v + 1 for v, k in enumerate(kk)}
print('drug dict : ', kk_dict)
print('drug dict length : ', len(kk_dict))


def encod_SMILES(seq, kk_dict):
    """Original SMILES encoding (unchanged)."""
    if pd.isnull(seq):
        return [0]
    else:
        return [kk_dict[a] for a in seq]


vocab_txt = open('vocab.txt')
bpe = BPE(vocab_txt, merges=-1, separator='')


def encodeSeq(seq, protein_dict):
    """Original BPE protein encoding (unchanged)."""
    firststep = bpe.process_line(seq).split()
    return [protein_dict[a] for a in firststep]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NEW: Meta-path cache utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_mp_cache = {}   # module-level cache: loaded once per process


def _load_metapath_cache(cache_dir):
    """
    Lazy-load pre-computed meta-path matrices from cache_dir.
    Expected files (produced by compute_metapaths.py):
        PT.npy, PD.npy, PDT.npy       – (Nd, Nt) float32
        drug_to_idx.pkl                – {DrugID → row index}
        prot_to_idx.pkl                – {ProteinID → col index}
        meta_config.pkl                – {metapath_len, Nt, Nd, ...}
    """
    global _mp_cache
    if cache_dir in _mp_cache:
        return _mp_cache[cache_dir]

    for fname in ['PT.npy', 'PD.npy', 'PDT.npy',
                  'drug_to_idx.pkl', 'prot_to_idx.pkl', 'meta_config.pkl']:
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Meta-path cache file not found: {path}\n"
                "Run compute_metapaths.py first.")

    PT  = np.load(os.path.join(cache_dir, 'PT.npy'),  allow_pickle=False)
    PD  = np.load(os.path.join(cache_dir, 'PD.npy'),  allow_pickle=False)
    PDT = np.load(os.path.join(cache_dir, 'PDT.npy'), allow_pickle=False)

    with open(os.path.join(cache_dir, 'drug_to_idx.pkl'), 'rb') as fh:
        drug_to_idx = pickle.load(fh)
    with open(os.path.join(cache_dir, 'prot_to_idx.pkl'), 'rb') as fh:
        prot_to_idx = pickle.load(fh)
    with open(os.path.join(cache_dir, 'meta_config.pkl'), 'rb') as fh:
        meta_config = pickle.load(fh)

    entry = {
        'PT': PT, 'PD': PD, 'PDT': PDT,
        'drug_to_idx': drug_to_idx,
        'prot_to_idx': prot_to_idx,
        'meta_config': meta_config,
    }
    _mp_cache[cache_dir] = entry
    print(f"  [meta-path] Loaded cache from '{cache_dir}': "
          f"PT{PT.shape}, PD{PD.shape}, PDT{PDT.shape}")
    return entry


def get_metapath_len(cache_dir='metapath_cache'):
    """Return 3*Nt (input dimension of the meta-path encoder)."""
    cache = _load_metapath_cache(cache_dir)
    return cache['meta_config']['metapath_len']


def extract_metapath_features(dti_df, cache_dir):
    """
    For each (DrugID, ProteinID) pair in dti_df, build:
        [PT[i_d, :], PD[i_d, :], PDT[i_d, :]]  ∈  R^{3·Nt}

    Unknown DrugIDs → zero vector (cold-start).

    Returns
    -------
    features : np.ndarray  shape (N_pairs, 3·Nt)  dtype float32
    """
    cache       = _load_metapath_cache(cache_dir)
    PT          = cache['PT']
    PD          = cache['PD']
    PDT         = cache['PDT']
    drug_to_idx = cache['drug_to_idx']
    Nt          = cache['meta_config']['Nt']

    N        = len(dti_df)
    features = np.zeros((N, 3 * Nt), dtype=np.float32)
    unknown  = set()

    for idx, (_, row) in enumerate(dti_df.iterrows()):
        drug_id = row[drug_col]
        if drug_id in drug_to_idx:
            i_d = drug_to_idx[drug_id]
            features[idx, :Nt]       = PT[i_d, :]
            features[idx, Nt:2*Nt]   = PD[i_d, :]
            features[idx, 2*Nt:]     = PDT[i_d, :]
        else:
            unknown.add(drug_id)

    if unknown:
        print(f"  [meta-path] {len(unknown)} unknown DrugID(s) → zero vector. "
              f"Examples: {list(unknown)[:3]}")
    return features


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  parse_data  – original + optional metapath extension
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_data(dti_dir, drug_dir, protein_dir,
               prot_vec='Convolution', prot_len=800,
               drug_vec='Convolution', drug_len=2048, drug_len2=100,
               metapath_cache_dir=None):
    """
    Parse DTI data and return feature dict.
    Original logic is PRESERVED.  When metapath_cache_dir is provided,
    key 'metapath_feature' (N_pairs × 3·Nt) is added to the returned dict.
    """
    # ── Original parsing block (UNCHANGED) ───────────────────────────────────
    print("parsing data : {0},{1},{2}".format(dti_dir, drug_dir, protein_dir))
    dti_df = pd.read_csv(dti_dir)

    drug_df = pd.read_csv(drug_dir, index_col=drug_col, dtype={'morgan_fp': str})
    drug_df['drug_embedding'] = drug_df['SMILES'].map(
        lambda a: encod_SMILES(a, kk_dict))

    protein_df = pd.read_csv(protein_dir, index_col=protein_col)
    protein_df['encoded_sequence'] = protein_df['Target_Sequence'].map(
        lambda a: encodeSeq(a, protein_dict))

    dti_df = pd.merge(dti_df, drug_df,    left_on=drug_col,    right_index=True)
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)

    drug_feature = np.array([
        [int(bit) for bit in fp]
        for fp in dti_df['morgan_fp'].values
    ])

    drug_feature2    = sequence.pad_sequences(
        dti_df['drug_embedding'].values, drug_len2, padding='post')
    protein_feature  = sequence.pad_sequences(
        dti_df['encoded_sequence'].values, prot_len, padding='post')
    protein_feature2 = sequence.pad_sequences(
        dti_df['encoded_sequence'].values, prot_len, padding='post')

    label = np.array([int(i) for i in dti_df[label_col].values])

    print('\t Positive data :   \t')
    print(sum(dti_df[label_col]))
    print('\t Negative data :   \t')
    print(dti_df.shape[0] - sum(dti_df[label_col]))

    result = {
        "protein_feature":  protein_feature,
        "protein_feature2": protein_feature2,
        "drug_feature":     drug_feature,
        "drug_feature2":    drug_feature2,
        "Label":            label,
    }

    # ── NEW: meta-path feature block ─────────────────────────────────────────
    if metapath_cache_dir is not None:
        print(f"  [meta-path] Extracting features from '{metapath_cache_dir}' ...")
        mp_features = extract_metapath_features(dti_df, metapath_cache_dir)
        result['metapath_feature'] = mp_features
        print(f"  [meta-path] metapath_feature shape: {mp_features.shape}")

    return result
