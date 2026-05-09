"""
main.py — Mini Search Engine with RAG Extension
AI356 — Information Retrieval, Spring 2025-2026

KEY DESIGN DECISION:
  - Retrieval for EVALUATION uses k=1400 (all docs ranked).
    MAP requires finding ALL relevant docs across the full corpus.
    Cranfield has ~8 relevant docs per query spread across 1400 docs —
    cutting off at k=20 means missing most relevant docs, killing MAP.

  - Retrieval for DISPLAY and RAG uses top-20 and top-5 respectively.

Usage:
    python main.py              # Full run including RAG
    python main.py --no-rag     # Skip RAG (no API key needed)
    python main.py --k 20       # Display/RAG cutoff (default 20)
"""

import os
import json
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Mini IR Search Engine with RAG")
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--k",           type=int, default=20,
                        help="Top-k for display and RAG (evaluation always uses full ranking)")
    parser.add_argument("--no-rag",      action="store_true")
    parser.add_argument("--rag-queries", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--rag-top-k",   type=int, default=5)
    parser.add_argument("--output",      default="results")
    return parser.parse_args()


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    start_total = time.time()

    # ── Phase 0 ────────────────────────────────────────────────
    banner("Phase 0 — Parsing Cranfield Dataset")
    from phase0_parser import parse_documents, parse_queries, parse_qrels

    docs        = parse_documents(os.path.join(args.data_dir, "cran.all.1400"))
    raw_queries = parse_queries(os.path.join(args.data_dir, "cran.qry"))
    raw_qrels   = parse_qrels(os.path.join(args.data_dir, "cran.qrel"))

    # THE CRANFIELD ASYMMETRIC ID FIX
    # Enforce a strict chronological 1-to-225 mapping for BOTH files.

    # 1. Align Queries chronologically
    queries = {}
    for new_id, (old_id, text) in enumerate(raw_queries.items(), start=1):
        queries[new_id] = text

    # 2. Align QRELs chronologically
    unique_qrel_qids = sorted(list(set(qid for qid, did in raw_qrels.keys())))
    qid_map = {old_id: new_id for new_id, old_id in enumerate(unique_qrel_qids, start=1)}

    qrels = {}
    for (old_qid, did), score in raw_qrels.items():
        qrels[(qid_map[old_qid], did)] = score

    print("  [Phase 0] Enforced strict 1-225 chronological alignment for queries and QRELs.")

    # ── Phase 1 ────────────────────────────────────────────────
    banner("Phase 1 — Text Preprocessing")
    from phase1_preprocessor import preprocess_corpus, preprocess_queries

    preprocessed_docs    = preprocess_corpus(docs)
    preprocessed_queries = preprocess_queries(queries)

    # ── Phase 2 ────────────────────────────────────────────────
    banner("Phase 2 — Building Inverted Index")
    from phase2_index import build_inverted_index, get_index_stats

    index, doc_lengths = build_inverted_index(preprocessed_docs)
    stats = get_index_stats(index)
    print(f"  Vocabulary size : {stats['vocabulary_size']:,}")
    print(f"  Total postings  : {stats['total_postings']:,}")
    print(f"  Top terms by DF : {stats['top_terms'][:5]}")

    # ── Phase 3 ────────────────────────────────────────────────
    banner("Phase 3 — Computing TF-IDF Matrix")
    from phase3_4_tfidf_retrieval import compute_tfidf_matrix, retrieve_all_queries

    doc_ids = sorted(docs.keys())
    N = len(doc_ids)
    tfidf_matrix, vocab, doc_id_to_row, idf = compute_tfidf_matrix(index, doc_ids, N)

    # ── Phase 4a — Full ranking for evaluation (k = all docs) ──
    banner("Phase 4 — Cosine Similarity Retrieval")
    N_docs = len(doc_ids)

    print(f"  [4a] Full ranking ({N_docs} docs) for MAP/NDCG evaluation...")
    results_full = retrieve_all_queries(
        preprocessed_queries, tfidf_matrix, vocab, doc_ids, idf, index, k=N_docs
    )

    print(f"  [4b] Top-{args.k} ranking for display and RAG...")
    results_topk = retrieve_all_queries(
        preprocessed_queries, tfidf_matrix, vocab, doc_ids, idf, index, k=args.k
    )

    # ── Phase 5 — Evaluate on FULL ranking ─────────────────────
    banner("Phase 5 — Evaluation: MAP + NDCG (full ranking)")
    from phase5_evaluation import evaluate

    eval_scores = evaluate(results_full, qrels, k=N_docs)

    # Also compute NDCG@20 for reference
    from phase5_evaluation import compute_mean_ndcg
    ndcg20, _ = compute_mean_ndcg(results_full, qrels, k=20)
    print(f"  NDCG@20 (for reference) : {ndcg20:.4f}")

    eval_output = {
        "MAP":       round(eval_scores["map"], 6),
        f"NDCG@{N_docs}": round(eval_scores[f"ndcg@{N_docs}"], 6),
        "NDCG@20":   round(ndcg20, 6),
        "per_query_AP": {
            str(qid): round(ap, 6)
            for qid, ap in eval_scores["per_query_ap"].items()
        },
        "per_query_NDCG": {
            str(qid): round(nd, 6)
            for qid, nd in eval_scores["per_query_ndcg"].items()
        },
    }
    eval_path = os.path.join(args.output, "evaluation_scores.json")
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n  Saved evaluation scores → {eval_path}")

    # ── Phase 6 — Display top-k results ────────────────────────
    banner(f"Phase 6 — Sample Retrieval Results (top-{args.k})")
    retrieval_samples = []

    for qid in args.rag_queries:
        if qid not in results_topk:
            continue
        ranked = results_topk[qid][:10]
        sample = {
            "query_id"   : qid,
            "query_text" : queries.get(qid, ""),
            "top_10_docs": [
                {
                    "rank"  : rank,
                    "doc_id": did,
                    "score" : round(score, 6),
                    "title" : docs[did]["title"][:100],
                }
                for rank, (did, score) in enumerate(ranked, 1)
            ],
        }
        retrieval_samples.append(sample)
        print(f"\n  Query {qid}: {queries.get(qid,'')[:70]}...")
        for entry in sample["top_10_docs"][:5]:
            print(
                f"    #{entry['rank']:2d}  Doc {entry['doc_id']:4d} "
                f"(score={entry['score']:.4f})  {entry['title'][:55]}"
            )

    retrieval_path = os.path.join(args.output, "sample_retrieval.json")
    with open(retrieval_path, "w") as f:
        json.dump(retrieval_samples, f, indent=2)
    print(f"\n  Saved sample retrieval → {retrieval_path}")

    # ── Phase 7 — RAG ──────────────────────────────────────────
    if not args.no_rag:
        banner("Phase 7 — RAG Answer Generation")
        from phase7_rag import run_rag_for_queries, print_rag_output

        try:
            rag_outputs = run_rag_for_queries(
                query_ids         = args.rag_queries,
                raw_queries       = queries,
                retrieval_results = results_topk,   # use top-k, not full ranking
                documents         = docs,
                top_k_for_rag     = args.rag_top_k,
            )
            print_rag_output(rag_outputs, args.rag_queries)

            rag_save = {
                str(qid): {
                    "query" : out["query"],
                    "answer": out["answer"],
                    "top_docs": [
                        {"doc_id": d["doc_id"], "title": d["title"],
                         "score": round(d["score"], 6)}
                        for d in out["top_docs"]
                    ],
                }
                for qid, out in rag_outputs.items()
            }
            rag_path = os.path.join(args.output, "rag_answers.json")
            with open(rag_path, "w") as f:
                json.dump(rag_save, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved RAG answers → {rag_path}")

        except EnvironmentError as e:
            print(f"\n  [SKIPPED] {e}")
        except Exception as e:
            print(f"\n  [RAG ERROR] {e}")
    else:
        print("\n  [Phase 7 skipped — --no-rag flag set]")

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - start_total
    banner("Run Complete")
    print(f"  MAP              : {eval_scores['map']:.4f}")
    print(f"  NDCG@20          : {ndcg20:.4f}")
    print(f"  Total runtime    : {elapsed:.1f}s")
    print(f"  Outputs saved in : {os.path.abspath(args.output)}/")
    print()


if __name__ == "__main__":
    main()