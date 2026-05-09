"""
Phase 7 — RAG-Based Answer Generation
Uses retrieved top-k documents as context for Gemini to generate answers.

Set your API key:
  Windows : $env:GEMINI_API_KEY = "AIza..."
  Mac/Linux: export GEMINI_API_KEY="AIza..."

Install: pip install google-genai
"""

import os
import time
from google import genai


def build_rag_prompt(query: str, top_docs: list) -> str:
    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        title    = doc.get("title", "").strip()
        abstract = doc.get("abstract", "").strip()
        score    = doc.get("score", 0.0)
        context_parts.append(
            f"[Document {i} | ID: {doc['doc_id']} | Relevance: {score:.4f}]\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )

    context_block = "\n\n".join(context_parts)

    return f"""You are an expert information retrieval assistant. You have been given a user query and a set of relevant documents retrieved from a scientific corpus.

USER QUERY:
{query}

RETRIEVED DOCUMENTS:
{context_block}

INSTRUCTIONS:
- Answer the query using ONLY the information in the retrieved documents above.
- Synthesize information across documents where relevant.
- Cite document IDs when you use specific information (e.g., "[Doc 17]").
- Be concise and technical — this is an aerospace/engineering corpus.
- If the documents do not contain enough information, say so clearly.

ANSWER:"""


def generate_rag_answer(query, top_docs, client, model="gemini-2.5-flash"):
    prompt = build_rag_prompt(query, top_docs)
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"[RAG ERROR] API call failed: {e}"


def run_rag_for_queries(query_ids, raw_queries, retrieval_results, documents,
                        top_k_for_rag=5, delay_seconds=0.5):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "  Windows : $env:GEMINI_API_KEY = 'AIza...'\n"
            "  Mac/Linux: export GEMINI_API_KEY='AIza...'"
        )

    client = genai.Client(api_key=api_key)
    rag_outputs = {}

    print(f"\n[Phase 7] Generating RAG answers for {len(query_ids)} queries...")

    for i, qid in enumerate(query_ids, 1):
        query_text = raw_queries.get(qid, "")
        ranked     = retrieval_results.get(qid, [])[:top_k_for_rag]

        top_docs = []
        for doc_id, score in ranked:
            doc = documents.get(doc_id, {})
            top_docs.append({
                "doc_id"  : doc_id,
                "title"   : doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "score"   : score,
            })

        print(f"  [{i}/{len(query_ids)}] Query {qid}: {query_text[:60]}...")
        answer = generate_rag_answer(query_text, top_docs, client)
        rag_outputs[qid] = {"query": query_text, "top_docs": top_docs, "answer": answer}

        if i < len(query_ids):
            time.sleep(delay_seconds)

    print("[Phase 7] RAG generation complete.")
    return rag_outputs


def print_rag_output(rag_outputs, query_ids=None):
    ids_to_show = query_ids or list(rag_outputs.keys())
    for qid in ids_to_show:
        if qid not in rag_outputs:
            continue
        out = rag_outputs[qid]
        print(f"\n{'='*60}")
        print(f"Query {qid}: {out['query']}")
        print(f"{'-'*60}")
        print("Retrieved documents:")
        for j, doc in enumerate(out["top_docs"], 1):
            print(f"  {j}. [Doc {doc['doc_id']}] (score={doc['score']:.4f}) {doc['title'][:60]}")
        print(f"{'-'*60}")
        print(f"RAG Answer:\n{out['answer']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    from phase0_parser import parse_documents, parse_queries
    from phase1_preprocessor import preprocess_corpus, preprocess_queries
    from phase2_index import build_inverted_index
    from phase3_4_tfidf_retrieval import compute_tfidf_matrix, retrieve_all_queries

    docs    = parse_documents(os.path.join("data", "cran.all.1400"))
    queries = parse_queries(os.path.join("data", "cran.qry"))

    preprocessed_docs    = preprocess_corpus(docs)
    preprocessed_queries = preprocess_queries(queries)

    index, _  = build_inverted_index(preprocessed_docs)
    doc_ids   = sorted(docs.keys())
    N         = len(doc_ids)

    tfidf_matrix, vocab, _, idf = compute_tfidf_matrix(index, doc_ids, N)

    all_results = retrieve_all_queries(
        preprocessed_queries, tfidf_matrix, vocab, doc_ids, idf, index, k=20
    )

    sample_qids = [1, 2, 3]
    rag_outputs = run_rag_for_queries(
        query_ids=sample_qids, raw_queries=queries,
        retrieval_results=all_results, documents=docs, top_k_for_rag=5,
    )
    print_rag_output(rag_outputs, sample_qids)
