"""
Phase 2 — Inverted Index Construction
Builds: term → {doc_id: term_frequency}
Also stores document lengths for normalization.
"""

from collections import defaultdict


def build_inverted_index(
    preprocessed_docs: dict[int, list[str]],
) -> tuple[dict[str, dict[int, int]], dict[int, int]]:
    """
    Build an inverted index from preprocessed documents.

    Args:
        preprocessed_docs: {doc_id: [stemmed_tokens]}

    Returns:
        inverted_index: {term: {doc_id: raw_count}}
        doc_lengths:    {doc_id: total_token_count}
    """
    inverted_index: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    doc_lengths: dict[int, int] = {}

    for doc_id, tokens in preprocessed_docs.items():
        doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            inverted_index[token][doc_id] += 1

    # Convert inner defaultdicts to plain dicts for safety
    inverted_index = {term: dict(postings) for term, postings in inverted_index.items()}

    vocab_size = len(inverted_index)
    avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0

    print(
        f"[Phase 2] Inverted index built | "
        f"Vocabulary size: {vocab_size:,} | "
        f"Avg doc length: {avg_doc_len:.1f} tokens"
    )

    return inverted_index, doc_lengths


def get_index_stats(inverted_index: dict[str, dict[int, int]]) -> dict:
    """Return useful statistics about the index."""
    df_values = [len(postings) for postings in inverted_index.values()]
    total_postings = sum(df_values)

    # Top 10 most frequent terms (highest document frequency)
    top_terms = sorted(
        inverted_index.items(), key=lambda x: len(x[1]), reverse=True
    )[:10]

    stats = {
        "vocabulary_size": len(inverted_index),
        "total_postings": total_postings,
        "max_df": max(df_values),
        "min_df": min(df_values),
        "avg_df": total_postings / len(inverted_index),
        "top_terms": [(term, len(postings)) for term, postings in top_terms],
    }
    return stats


if __name__ == "__main__":
    from phase0_parser import parse_documents
    from phase1_preprocessor import preprocess_corpus
    import os

    docs = parse_documents(os.path.join("data", "cran.all.1400"))
    preprocessed = preprocess_corpus(docs)
    index, lengths = build_inverted_index(preprocessed)

    stats = get_index_stats(index)
    print(f"\n[Phase 2] Index Statistics:")
    print(f"  Vocabulary size  : {stats['vocabulary_size']:,}")
    print(f"  Total postings   : {stats['total_postings']:,}")
    print(f"  Avg doc freq     : {stats['avg_df']:.2f}")
    print(f"  Top 10 terms by DF:")
    for term, df in stats["top_terms"]:
        print(f"    '{term}': appears in {df} docs")

    # Check a sample term
    sample_term = "lift"
    if sample_term in index:
        sample_postings = dict(list(index[sample_term].items())[:5])
        print(f"\n  Postings for '{sample_term}' (first 5 docs): {sample_postings}")
