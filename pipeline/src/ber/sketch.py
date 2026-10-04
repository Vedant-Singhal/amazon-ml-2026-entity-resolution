"""Char-3-gram TF-IDF views compressed to dense vectors with a count sketch (signed random
hashing, Charikar et al. 2002), so the same FAISS index code can search them. The inner product
of two sketches approximates the cosine of the TF-IDF vectors (error ~ 1/sqrt(dim)).
IDF comes from the split's own records (unsupervised)."""
import gc
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from .config import P

NF = 2**20
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=NF, alternate_sign=False,
                        norm=None, lowercase=False, dtype=np.float32)
_IDF, _R = None, None


def _df_chunk(texts):
    return np.bincount(_HV.transform(texts).indices, minlength=NF).astype(np.float64)


def _sk_chunk(texts):
    X = _HV.transform(texts).tocsr()
    X.data = np.log1p(X.data) * _IDF[X.indices]
    Y = (X @ _R).toarray()
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)
    return Y.astype(np.float16)


def _pool_map(fn, chunks):
    gc.collect()
    gc.freeze()
    try:
        with Pool(P.pool_workers) as pool:
            return pool.map(fn, chunks)
    finally:
        gc.unfreeze()


def sketch(texts: list, dim: int) -> np.ndarray:
    """-> (len(texts), dim) float16, L2-normalized (all-zero rows for empty texts)."""
    global _IDF, _R
    chunks = [texts[i:i + 100_000] for i in range(0, len(texts), 100_000)]
    df = np.sum(_pool_map(_df_chunk, chunks), axis=0)
    _IDF = (np.log((1 + len(texts)) / (1 + df)) + 1).astype(np.float32)
    rng = np.random.default_rng(P.seed)
    _R = sp.csr_matrix((rng.choice(np.array([-1.0, 1.0], np.float32), NF),
                        (np.arange(NF), rng.integers(0, dim, NF))), shape=(NF, dim))
    return np.concatenate(_pool_map(_sk_chunk, chunks))
