"""
Phase 5 — Evaluation
Computes MAP (binary relevance) and NDCG (graded relevance).

Cranfield relevance scale (REVERSED — 1 is HIGHEST relevance):
  Score  Meaning            Binary    Graded gain
  -----  -------------------  ------    -----------
    1    Most relevant          1           4
    2    Relevant               1           3
    3    Marginally relevant    1           2
    4    Marginally irrelevant  1           1
   -1    Irrelevant             0           0
  miss   Not judged             0           0

KEY NOTE on Cranfield:
  Documents NOT listed in cran.qrel for a given query are NON-RELEVANT.
  Only documents that appear with scores 1-4 are treated as relevant.
  Score -1 is explicitly irrelevant.
"""

import math
import numpy as np


# ─────────────────────────────────────────────
# Relevance converters
# ─────────────────────────────────────────────

def binary_relevance(score) -> int:
    """Scores 1,2,3,4 → relevant (1). Score -1 or None → not relevant (0)."""
    if score is None or score == -1:
        return 0
    return 1


def graded_relevance(score) -> int:
    """
    Convert Cranfield score to graded gain.
    Cranfield scale is reversed: 1=best relevance → highest gain.
      1 → 4, 2 → 3, 3 → 2, 4 → 1, -1/None → 0
    """
    mapping = {1: 4, 2: 3, 3: 2, 4: 1}
    if score is None or score == -1:
        return 0
    return mapping.get(score, 0)


# ─────────────────────────────────────────────
# Phase 5a: MAP
# ─────────────────────────────────────────────

def average_precision(
    ranked_docs: list,
    relevant_docs: set,
) -> float:
    """
    Compute Average Precision for ONE query.

    Args:
        ranked_docs  : [doc_id, ...] in ranked order (best first)
        relevant_docs: set of doc_ids that are relevant for this query

    Returns:
        AP score (0.0 if no relevant docs exist)
    """
    if not relevant_docs:
        return 0.0

    num_relevant_found = 0
    sum_precision = 0.0

    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            num_relevant_found += 1
            sum_precision += num_relevant_found / rank

    # Divide by TOTAL relevant docs (not just found ones)
    return sum_precision / len(relevant_docs)


def compute_map(
    retrieval_results: dict,
    qrels: dict,
) -> tuple:
    """
    Compute MAP over all queries.

    Args:
        retrieval_results : {query_id: [(doc_id, score), ...]}
        qrels             : {(query_id, doc_id): relevance_score}

    Returns:
        (map_score, {query_id: AP_score})
    """
    # Pre-build relevant sets per query for speed, enforcing integer types
    relevant_per_query = {}
    for (qid, did), score in qrels.items():
        if binary_relevance(score) == 1:
            relevant_per_query.setdefault(int(qid), set()).add(int(did))

    per_query_ap = {}
    for qid, ranked in retrieval_results.items():
        safe_qid = int(qid)
        # Force doc IDs to ints
        ranked_docs = [int(did) for did, _ in ranked]
        rel_set = relevant_per_query.get(safe_qid, set())
        per_query_ap[safe_qid] = average_precision(ranked_docs, rel_set)

    map_score = sum(per_query_ap.values()) / len(per_query_ap) if per_query_ap else 0.0
    return map_score, per_query_ap


# ─────────────────────────────────────────────
# Phase 5b: NDCG
# ─────────────────────────────────────────────

def dcg_at_k(gains: list, k: int) -> float:
    """Compute DCG@k from a list of gains."""
    dcg = 0.0
    for i, gain in enumerate(gains[:k], start=1):
        if gain > 0:
            dcg += gain / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    ranked_docs: list,
    qrels: dict,
    query_id: int,
    k: int = 20,
) -> float:
    """
    Compute NDCG@k for ONE query.

    Args:
        ranked_docs : [doc_id, ...] in ranked order
        qrels       : {(query_id, doc_id): score}
        query_id    : current query
        k           : cutoff rank
    """
    safe_qid = int(query_id)

    # Gains for the ranked list
    gains = []
    for doc_id in ranked_docs[:k]:
        # Force integer lookup for both query ID and document ID
        raw = qrels.get((safe_qid, int(doc_id)), None)
        gains.append(graded_relevance(raw))

    dcg = dcg_at_k(gains, k)

    # Ideal gains = all judged docs for this query, sorted best first
    ideal_gains = sorted(
        [graded_relevance(score)
         for (qid, did), score in qrels.items()
         if int(qid) == safe_qid],
        reverse=True,
    )

    idcg = dcg_at_k(ideal_gains, k)

    return (dcg / idcg) if idcg > 0 else 0.0


def compute_mean_ndcg(
    retrieval_results: dict,
    qrels: dict,
    k: int = 20,
) -> tuple:
    """
    Compute mean NDCG@k over all queries.
    Returns (mean_ndcg, {query_id: ndcg_score})
    """
    per_query_ndcg = {}
    for qid, ranked in retrieval_results.items():
        safe_qid = int(qid)
        # Force doc IDs to ints
        ranked_docs = [int(did) for did, _ in ranked]
        per_query_ndcg[safe_qid] = ndcg_at_k(ranked_docs, qrels, safe_qid, k=k)

    mean_ndcg = (
        sum(per_query_ndcg.values()) / len(per_query_ndcg)
        if per_query_ndcg else 0.0
    )
    return mean_ndcg, per_query_ndcg


# ─────────────────────────────────────────────
# Full runner
# ─────────────────────────────────────────────

def evaluate(
    retrieval_results: dict,
    qrels: dict,
    k: int = 20,
) -> dict:
    """Run MAP + NDCG@k and print results."""
    map_score, per_query_ap   = compute_map(retrieval_results, qrels)
    mean_ndcg, per_query_ndcg = compute_mean_ndcg(retrieval_results, qrels, k=k)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS (top-{k})")
    print(f"{'='*50}")
    print(f"  MAP  (binary relevance) : {map_score:.4f}")
    print(f"  NDCG@{k} (graded relev.) : {mean_ndcg:.4f}")
    print(f"{'='*50}\n")

    sorted_ap = sorted(per_query_ap.items(), key=lambda x: x[1])
    print(f"  Worst 5 queries by AP : {[(q, round(s,3)) for q,s in sorted_ap[:5]]}")
    print(f"  Best  5 queries by AP : {[(q, round(s,3)) for q,s in sorted_ap[-5:]]}")

    return {
        "map": map_score,
        f"ndcg@{k}": mean_ndcg,
        "per_query_ap": per_query_ap,
        "per_query_ndcg": per_query_ndcg,
    }


if __name__ == "__main__":
    import os
    from phase0_parser import parse_documents, parse_queries, parse_qrels
    from phase1_preprocessor import preprocess_corpus, preprocess_queries
    from phase2_index import build_inverted_index
    from phase3_4_tfidf_retrieval import compute_tfidf_matrix, retrieve_all_queries

    docs    = parse_documents(os.path.join("data", "cran.all.1400"))
    queries = parse_queries(os.path.join("data", "cran.qry"))
    qrels   = parse_qrels(os.path.join("data", "cran.qrel"))

    # ─────────────────────────────────────────────
    # THE FIX: Remap Cranfield Query IDs
    # ─────────────────────────────────────────────
    # Get all unique 365-scale query IDs from the qrels file and sort them
    unique_qrel_qids = sorted(list(set(qid for qid, did in qrels.keys())))
    
    # Map them strictly 1 to 225
    qid_map = {old_id: new_id for new_id, old_id in enumerate(unique_qrel_qids, start=1)}
    
    # Apply the mapping to the qrels dictionary
    aligned_qrels = {}
    for (old_qid, did), score in qrels.items():
        new_qid = qid_map[old_qid]
        aligned_qrels[(new_qid, did)] = score
    # ─────────────────────────────────────────────

    preprocessed_docs    = preprocess_corpus(docs)
    preprocessed_queries = preprocess_queries(queries)

    index, _ = build_inverted_index(preprocessed_docs)
    doc_ids  = sorted(docs.keys())
    N        = len(doc_ids)

    tfidf_matrix, vocab, _, idf = compute_tfidf_matrix(index, doc_ids, N)

    all_results = retrieve_all_queries(
        preprocessed_queries, tfidf_matrix, vocab, doc_ids, idf, index, k=20
    )

    # Pass the ALIGNED qrels to the evaluator
    evaluate(all_results, aligned_qrels, k=20)