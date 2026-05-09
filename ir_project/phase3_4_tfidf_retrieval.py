"""
Phase 3 — TF-IDF Weighting
Phase 4 — Cosine Similarity Ranking

Uses scipy sparse matrices for efficient computation across all 225 queries × 1400 docs.

Formulas (tuned for best Cranfield performance):
  TF(t,d)  = 1 + log10(count)  if count > 0, else 0   [log-normalized]
  IDF(t)   = log10((N+1) / (df(t)+1)) + 1              [smooth IDF, avoids zero]
  TF-IDF   = TF × IDF
  Vectors  → L2-normalized → cosine sim = dot product
"""

import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags


# ─────────────────────────────────────────────
# Phase 3: TF-IDF Matrix Construction
# ─────────────────────────────────────────────

def compute_tfidf_matrix(
    inverted_index: dict,
    doc_ids: list,
    N: int,
) -> tuple:
    """
    Build a TF-IDF document matrix (D × V), L2-normalized.

    Args:
        inverted_index : {term: {doc_id: raw_count}}
        doc_ids        : ordered list of all document IDs
        N              : total number of documents

    Returns:
        tfidf_matrix   : sparse (D × V) CSR matrix, L2-normalized rows
        vocab          : list of terms in column order
        doc_id_to_row  : {doc_id: row_index}
        idf            : {term: idf_value}  (needed for query vectorization)
    """
    vocab = sorted(inverted_index.keys())
    term_to_col = {term: i for i, term in enumerate(vocab)}
    doc_id_to_row = {did: i for i, did in enumerate(doc_ids)}

    D = len(doc_ids)
    V = len(vocab)

    # Smooth IDF: log10((N+1)/(df+1)) + 1  — never zero, avoids log(0)
    idf = {}
    for term in vocab:
        df = len(inverted_index[term])
        idf[term] = math.log10((N + 1) / (df + 1)) + 1.0

    print(f"[Phase 3] Building TF-IDF matrix ({D} docs × {V} terms)...")

    mat = lil_matrix((D, V), dtype=np.float32)

    for term, postings in inverted_index.items():
        col = term_to_col[term]
        term_idf = idf[term]
        for doc_id, raw_count in postings.items():
            row = doc_id_to_row[doc_id]
            tf = 1.0 + math.log10(raw_count)
            mat[row, col] = tf * term_idf

    tfidf_csr = mat.tocsr()
    tfidf_csr = _l2_normalize_rows(tfidf_csr)

    print(
        f"[Phase 3] TF-IDF matrix built and L2-normalized | "
        f"Non-zeros: {tfidf_csr.nnz:,}"
    )
    return tfidf_csr, vocab, doc_id_to_row, idf


def _l2_normalize_rows(matrix: csr_matrix) -> csr_matrix:
    """L2-normalize each row of a CSR sparse matrix."""
    norms = np.sqrt(matrix.power(2).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    normalizer = diags(1.0 / norms)
    return normalizer @ matrix


def vectorize_query(
    query_tokens: list,
    vocab: list,
    idf: dict,
    inverted_index: dict,
) -> np.ndarray:
    """
    Convert a preprocessed query into a TF-IDF vector.
    Uses the SAME IDF values computed for documents.
    Returns L2-normalized dense vector of shape (V,).
    """
    term_to_col = {term: i for i, term in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=np.float32)

    term_counts = {}
    for token in query_tokens:
        if token in term_to_col:
            term_counts[token] = term_counts.get(token, 0) + 1

    for term, count in term_counts.items():
        col = term_to_col[term]
        tf = 1.0 + math.log10(count)
        vec[col] = tf * idf.get(term, 0.0)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ─────────────────────────────────────────────
# Phase 4: Cosine Similarity + Top-k Retrieval
# ─────────────────────────────────────────────

def retrieve_top_k(
    query_vec: np.ndarray,
    tfidf_matrix: csr_matrix,
    doc_ids: list,
    k: int = 20,
) -> list:
    """
    Compute cosine similarity (= dot product, since both L2-normalized).
    Returns list of (doc_id, score) sorted descending, length k.
    """
    scores = tfidf_matrix.dot(query_vec)

    if k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [(doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    return results


def retrieve_all_queries(
    preprocessed_queries: dict,
    tfidf_matrix: csr_matrix,
    vocab: list,
    doc_ids: list,
    idf: dict,
    inverted_index: dict,
    k: int = 20,
) -> dict:
    """
    Run retrieval for every query.
    Returns: {query_id: [(doc_id, score), ...]}
    """
    print(f"[Phase 4] Running retrieval for {len(preprocessed_queries)} queries (top-{k})...")
    results = {}
    for qid, tokens in preprocessed_queries.items():
        q_vec = vectorize_query(tokens, vocab, idf, inverted_index)
        results[qid] = retrieve_top_k(q_vec, tfidf_matrix, doc_ids, k=k)

    retrieved_counts = [len(r) for r in results.values()]
    avg = sum(retrieved_counts) / len(retrieved_counts) if retrieved_counts else 0
    print(f"[Phase 4] Retrieval done | Avg results per query: {avg:.1f}")
    return results


if __name__ == "__main__":
    import os
    from phase0_parser import parse_documents, parse_queries, parse_qrels
    from phase1_preprocessor import preprocess_corpus, preprocess_queries
    from phase2_index import build_inverted_index

    docs    = parse_documents(os.path.join("data", "cran.all.1400"))
    queries = parse_queries(os.path.join("data", "cran.qry"))

    preprocessed_docs    = preprocess_corpus(docs)
    preprocessed_queries = preprocess_queries(queries)

    index, lengths = build_inverted_index(preprocessed_docs)
    doc_ids = sorted(docs.keys())
    N = len(doc_ids)

    tfidf_matrix, vocab, doc_id_to_row, idf = compute_tfidf_matrix(index, doc_ids, N)

    all_results = retrieve_all_queries(
        preprocessed_queries, tfidf_matrix, vocab, doc_ids, idf, index, k=20
    )

    qid = 1
    print(f"\nQuery {qid}: {queries[qid][:100]}")
    print("Top-10 results:")
    for rank, (did, score) in enumerate(all_results[qid][:10], 1):
        print(f"  {rank:2d}. Doc {did:4d} (score={score:.4f})  {docs[did]['title'][:60]}")
