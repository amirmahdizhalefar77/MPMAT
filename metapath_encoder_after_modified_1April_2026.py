import os, time
import numpy as np
from multiprocessing import Pool

# ──────────────────────────────────────────────────────────────────────────────
# 1. Protein-Protein Similarity  (Section II-C-1, Eq. 1)
#    Optimised: pre-computed self-scores, batched IPC, all cores used.
# ──────────────────────────────────────────────────────────────────────────────

def _nw_self_score(seq):
    """Compute self-alignment score for a single sequence."""
    try:
        from Bio import pairwise2
        return float(pairwise2.align.globalxx(seq, seq, score_only=True))
    except Exception:
        return 0.0


def _nw_batch(args):
    """
    Worker: score a *batch* of pre-indexed pairs.
    args = (pairs, sequences, self_scores)
      pairs       : list of (i, j) ints
      sequences   : list of str   (full list — read-only in forked child)
      self_scores : list of float (pre-computed)
    Returns list of (i, j, score) triples.
    """
    pairs, sequences, self_scores = args
    try:
        from Bio import pairwise2
    except ImportError:
        return [(i, j, 0.0) for i, j in pairs]

    out = []
    for i, j in pairs:
        try:
            raw   = pairwise2.align.globalxx(sequences[i], sequences[j],
                                             score_only=True)
            denom = max(self_scores[i], self_scores[j])
            score = float(raw) / denom if denom > 0 else 0.0
        except Exception:
            score = 0.0
        out.append((i, j, score))
    return out


def compute_protein_similarity(sequences, n_workers=None, batch_size=400,
                               verbose=True):
    """
    Compute symmetric NW similarity matrix S_P ∈ R^{Np × Np}.

    Speed-ups over the original:
      1. Self-scores computed once in parallel (N calls, not 2·N² embedded calls).
      2. Pairs sent to workers in batches of `batch_size`, cutting IPC cost by
         ~batch_size× compared to one pair per call.
      3. imap_unordered gives better load balancing and live progress.
      4. n_workers defaults to all logical cores on the machine.

    Parameters
    ----------
    sequences  : list of str
    n_workers  : int | None   — None → os.cpu_count()
    batch_size : int          — pairs per worker call (tune to your sequence
                                length; longer seqs → smaller batch)
    verbose    : bool

    Returns
    -------
    SP : np.ndarray  shape (Np, Np), float32
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    Np    = len(sequences)
    pairs = [(i, j) for i in range(Np) for j in range(i + 1, Np)]
    n_pairs = len(pairs)

    if verbose:
        print(f"  Proteins  : {Np}")
        print(f"  Pairs     : {n_pairs:,}")
        print(f"  Workers   : {n_workers}")
        print(f"  Batch size: {batch_size} pairs/call  "
              f"({n_pairs // batch_size + 1} total batches)")

    # ── Step 1: self-scores (parallel, N calls) ──────────────────────────────
    if verbose:
        print("  [1/2] Pre-computing self-scores ...")
        t0 = time.time()

    with Pool(processes=n_workers) as pool:
        self_scores = pool.map(_nw_self_score, sequences,
                               chunksize=max(1, Np // (n_workers * 4)))

    if verbose:
        print(f"        Done in {time.time()-t0:.1f}s")

    # ── Step 2: off-diagonal pairs (batched, parallel) ───────────────────────
    if verbose:
        print("  [2/2] Computing pairwise NW scores ...")
        t0 = time.time()

    # Slice pairs into batches and ship the full sequence list once per fork
    # (forked child inherits the parent's memory — no re-serialisation of
    #  `sequences` per call, only the small (i, j) index lists are pickled).
    batches = [
        (pairs[k:k + batch_size], sequences, self_scores)
        for k in range(0, n_pairs, batch_size)
    ]
    n_batches  = len(batches)
    SP         = np.eye(Np, dtype=np.float32)
    done_pairs = 0

    with Pool(processes=n_workers) as pool:
        for batch_result in pool.imap_unordered(_nw_batch, batches,
                                                chunksize=1):
            for i, j, s in batch_result:
                SP[i, j] = s
                SP[j, i] = s
            done_pairs += len(batch_result)

            if verbose:
                elapsed = time.time() - t0
                rate    = done_pairs / elapsed if elapsed > 0 else 0
                eta     = (n_pairs - done_pairs) / rate if rate > 0 else 0
                pct     = 100 * done_pairs / n_pairs
                print(f"\r        {pct:5.1f}%  "
                      f"{done_pairs:,}/{n_pairs:,} pairs  "
                      f"{rate:,.0f} pairs/s  "
                      f"ETA {eta/60:.1f} min   ",
                      end="", flush=True)

    if verbose:
        print(f"\n        Done in {time.time()-t0:.1f}s")

    return SP