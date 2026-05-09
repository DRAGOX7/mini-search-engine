"""
Phase 0 — Cranfield Dataset Parser
Parses cran.all.1400, cran.qry, cran.qrel into clean Python structures.
"""

import os


def parse_documents(filepath: str) -> dict:
    documents = {}
    current_id = None
    current_section = None
    buffer = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                if current_id is not None:
                    if current_section == "W":
                        documents[current_id]["abstract"] = " ".join(buffer).strip()
                    elif current_section == "T":
                        documents[current_id]["title"] = " ".join(buffer).strip()
                    documents[current_id]["content"] = (
                        documents[current_id]["title"] + " " +
                        documents[current_id]["abstract"]
                    ).strip()
                current_id = int(line[3:].strip())
                documents[current_id] = {"title": "", "abstract": "", "content": ""}
                current_section = None
                buffer = []
            elif line.startswith(".T"):
                if current_section == "W":
                    documents[current_id]["abstract"] = " ".join(buffer).strip()
                elif current_section == "T":
                    documents[current_id]["title"] = " ".join(buffer).strip()
                current_section = "T"
                buffer = []
            elif line.startswith(".A"):
                if current_section == "T":
                    documents[current_id]["title"] = " ".join(buffer).strip()
                elif current_section == "W":
                    documents[current_id]["abstract"] = " ".join(buffer).strip()
                current_section = "A"
                buffer = []
            elif line.startswith(".B"):
                current_section = "B"
                buffer = []
            elif line.startswith(".W"):
                if current_section == "T":
                    documents[current_id]["title"] = " ".join(buffer).strip()
                current_section = "W"
                buffer = []
            else:
                if current_section in ("T", "W"):
                    buffer.append(line)

    if current_id is not None:
        if current_section == "W":
            documents[current_id]["abstract"] = " ".join(buffer).strip()
        elif current_section == "T":
            documents[current_id]["title"] = " ".join(buffer).strip()
        documents[current_id]["content"] = (
            documents[current_id]["title"] + " " +
            documents[current_id]["abstract"]
        ).strip()

    print(f"[Phase 0] Parsed {len(documents)} documents.")
    return documents


def parse_queries(filepath: str) -> dict:
    queries = {}
    current_id = None
    current_section = None
    buffer = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                if current_id is not None and current_section == "W":
                    queries[current_id] = " ".join(buffer).strip()
                current_id = int(line[3:].strip())
                current_section = None
                buffer = []
            elif line.startswith(".W"):
                current_section = "W"
                buffer = []
            else:
                if current_section == "W":
                    buffer.append(line)

    if current_id is not None and current_section == "W":
        queries[current_id] = " ".join(buffer).strip()

    print(f"[Phase 0] Parsed {len(queries)} queries.")
    return queries


def parse_qrels(filepath: str) -> dict:
    """
    Parse cran.qrel — handles both 3-column and 4-column formats.

    Cranfield relevance scale (REVERSED — 1 is BEST):
      1 = most relevant      → binary: relevant, graded gain: 4
      2 = relevant           → binary: relevant, graded gain: 3
      3 = marginally rel.    → binary: relevant, graded gain: 2
      4 = marginally irrel.  → binary: relevant, graded gain: 1
     -1 = irrelevant         → binary: not relevant, graded gain: 0

    NOTE: The standard Cranfield dataset has exactly 1,837 judgments.
          This is correct and expected.

    Returns: {(query_id, doc_id): relevance_score}
    """
    qrels = {}
    skipped = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                skipped += 1
                continue
            try:
                qid   = int(parts[0])
                did   = int(parts[1])
                score = int(parts[2])
                qrels[(qid, did)] = score
            except ValueError:
                skipped += 1
                continue

    query_ids_covered = set(qid for (qid, _) in qrels)
    doc_ids_covered   = set(did for (_, did) in qrels)

    print(f"[Phase 0] Parsed {len(qrels):,} relevance judgments "
          f"({skipped} lines skipped).")
    print(f"  Queries with judgments : {len(query_ids_covered)}")
    print(f"  Unique docs judged     : {len(doc_ids_covered)}")

    return qrels


if __name__ == "__main__":
    base = "data"
    docs    = parse_documents(os.path.join(base, "cran.all.1400"))
    queries = parse_queries(os.path.join(base, "cran.qry"))
    qrels   = parse_qrels(os.path.join(base, "cran.qrel"))

    rel_for_q1 = {did: s for (qid, did), s in qrels.items() if qid == 1}
    print(f"\nAll relevance judgments for query 1 ({len(rel_for_q1)} docs):")
    for did, score in sorted(rel_for_q1.items()):
        print(f"  Doc {did:4d} → score {score}")

    print(f"\nSample Doc 1 title: {docs[1]['title']}")
    print(f"Sample Query 1: {queries[1][:80]}")
