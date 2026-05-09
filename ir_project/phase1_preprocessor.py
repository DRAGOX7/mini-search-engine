"""
Phase 1 — Text Preprocessing
Tokenization → Lowercasing → Stopword Removal → Porter Stemming
Same pipeline applied to BOTH documents and queries (mandatory).
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    packages = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
    ]
    for path, name in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[Phase 1] Downloading NLTK '{name}'...")
            nltk.download(name, quiet=True)


# Initialize once globally for performance
ensure_nltk_data()
_stemmer = PorterStemmer()
_stopwords = set(stopwords.words("english"))


def preprocess(text: str) -> list[str]:
    """
    Full preprocessing pipeline.

    Steps:
      1. Lowercase
      2. Tokenize (keep only alphabetic tokens, min length 2)
      3. Remove stopwords
      4. Porter stem

    Args:
        text: Raw text string.
    Returns:
        List of stemmed tokens.
    """
    if not text:
        return []

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Tokenize — only alpha words, length >= 2
    tokens = re.findall(r"\b[a-z]{2,}\b", text)

    # Step 3 + 4: Remove stopwords and stem in one pass
    processed = [
        _stemmer.stem(token)
        for token in tokens
        if token not in _stopwords
    ]

    return processed


def preprocess_corpus(documents: dict[int, dict]) -> dict[int, list[str]]:
    """
    Preprocess all documents.
    Args:
        documents: {doc_id: {"content": str, ...}}
    Returns:
        {doc_id: [stemmed_tokens]}
    """
    preprocessed = {}
    for doc_id, doc in documents.items():
        preprocessed[doc_id] = preprocess(doc["content"])

    total_tokens = sum(len(t) for t in preprocessed.values())
    print(
        f"[Phase 1] Preprocessed {len(preprocessed)} documents | "
        f"Total tokens: {total_tokens:,} | "
        f"Avg per doc: {total_tokens // len(preprocessed)}"
    )
    return preprocessed


def preprocess_queries(queries: dict[int, str]) -> dict[int, list[str]]:
    """
    Preprocess all queries — SAME pipeline as documents.
    Args:
        queries: {query_id: query_text}
    Returns:
        {query_id: [stemmed_tokens]}
    """
    preprocessed = {}
    for qid, text in queries.items():
        preprocessed[qid] = preprocess(text)

    print(f"[Phase 1] Preprocessed {len(preprocessed)} queries.")
    return preprocessed


if __name__ == "__main__":
    # Quick smoke test
    sample = "The aerodynamic efficiency of aircraft wings at high Reynolds numbers"
    result = preprocess(sample)
    print(f"Input:  {sample}")
    print(f"Output: {result}")

    sample2 = "What is the effect of turbulence on lift coefficient?"
    result2 = preprocess(sample2)
    print(f"\nInput:  {sample2}")
    print(f"Output: {result2}")
