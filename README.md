<div align="center">

# 🔍 Mini Search Engine with RAG Extension

**AI356 — Information Retrieval · Spring 2025-2026**
**Jordan University of Science and Technology**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-154F5B?style=flat-square)](https://nltk.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.12+-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/gemini)
[![Dataset](https://img.shields.io/badge/Dataset-Cranfield_1400-orange?style=flat-square)](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)
[![MAP](https://img.shields.io/badge/MAP-0.2976-success?style=flat-square)](/)
[![NDCG@20](https://img.shields.io/badge/NDCG@20-0.3988-success?style=flat-square)](/)

<br/>

A complete **Information Retrieval system** built on the **Vector Space Model** with **TF-IDF weighting** and **cosine similarity ranking**, evaluated on the Cranfield benchmark, and extended with a **Retrieval-Augmented Generation (RAG)** pipeline powered by Google Gemini.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Phase Details](#-phase-details)
- [Sample Outputs](#-sample-outputs)
- [RAG Answers](#-rag-answers)
- [Dataset](#-dataset)
- [Evaluation](#-evaluation)
- [Dependencies](#-dependencies)

---

## 🧭 Overview

This project implements a two-part search system:

**Part 1 — Classical IR (Vector Space Model)**
> Build, index, and rank 1,400 aerospace documents using TF-IDF weighting and cosine similarity. Evaluate with MAP and NDCG on the Cranfield benchmark.

**Part 2 — RAG Extension**
> Feed the top-5 retrieved documents into Google Gemini 2.5 Flash to generate a cited, grounded natural-language answer for each query.

```
User Query  →  [TF-IDF Retrieval]  →  Top-k Documents  →  [Gemini LLM]  →  AI Answer
```

---

## 📊 Results

| Metric | Score | Description |
|--------|-------|-------------|
| **MAP** | **0.2976** | Mean Average Precision — binary relevance (scores 1–4 = relevant) |
| **NDCG@20** | **0.3988** | Normalized DCG at rank 20 — graded relevance |
| **NDCG@1400** | **0.5186** | Normalized DCG — full corpus ranking |

### Per-Query Highlights

```
Best queries:
  Query 9   → AP: 1.0000  (Perfect retrieval)
  Query 119 → AP: 1.0000  (Perfect retrieval)
  Query 172 → AP: 0.8542
  Query 3   → AP: 0.6592  (Heat conduction — excellent)

Worst queries:
  Query 13  → AP: 0.0000  (Query terms absent after stemming)
  Query 22  → AP: 0.0000
  Query 44  → AP: 0.0000
```

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE OVERVIEW                               │
├──────────┬──────────┬──────────┬──────────┬──────────┬────────┬────────┤
│ Phase 0  │ Phase 1  │ Phase 2  │ Phase 3  │ Phase 4  │ Phase5 │ Phase7 │
│          │          │          │          │          │        │        │
│  Parse   │ Preproc  │ Inverted │  TF-IDF  │  Cosine  │  MAP   │  RAG   │
│ Dataset  │   Text   │  Index   │  Matrix  │ Ranking  │  NDCG  │  Gen.  │
│          │          │          │          │          │        │        │
│ cran.*   │lowercase │term→docs │TF×IDF    │dot prod  │binary  │Gemini  │
│  files   │tokenize  │raw freqs │L2-norm   │top-k     │graded  │2.5Flash│
│          │stopwords │          │sparse    │          │        │        │
│          │ stemming │          │ matrix   │          │        │        │
└──────────┴──────────┴──────────┴──────────┴──────────┴────────┴────────┘
     ↓           ↓          ↓          ↓          ↓         ↓        ↓
  1400 docs   138,732   4,267 terms  83,094   MAP=0.2976  cited
  225 queries  tokens   vocabulary  non-zeros NDCG=0.3988 answers
  1,837 qrels
```

### Data Flow

```
cran.all.1400 ──→ parse_documents() ──→ {doc_id: {title, abstract, content}}
                                                        │
                                                        ▼
cran.qry ──────→ parse_queries() ──→ {query_id: text}  │
                                            │           │
                                            ▼           ▼
                                    preprocess()   preprocess()     ← SAME pipeline
                                            │           │
                                            ▼           ▼
                                    query_tokens   doc_tokens
                                            │           │
                                            │           ▼
                                            │    build_inverted_index()
                                            │           │
                                            │           ▼
                                            │    compute_tfidf_matrix()   → sparse (1400×4267)
                                            │           │
                                            └───────────┤
                                                        ▼
                                              vectorize_query()
                                                        │
                                                        ▼
                                            cosine_similarity()  → ranked docs
                                                        │
                              ┌─────────────────────────┤
                              │                         │
                              ▼                         ▼
                        evaluate()               top-5 docs → Gemini → answer
                      MAP + NDCG
```

---

## 📁 Project Structure

```
ir_project/
│
├── data/                          # Cranfield dataset files
│   ├── cran.all.1400              # 1,400 aerospace documents
│   ├── cran.qry                   # 225 user queries
│   ├── cran.qrel                  # 1,837 relevance judgments
│   └── cranqrel.readme            # Relevance scale documentation
│
├── results/                       # Auto-generated outputs
│   ├── evaluation_scores.json     # MAP, NDCG, per-query scores
│   ├── sample_retrieval.json      # Top-10 docs for sample queries
│   └── rag_answers.json           # AI-generated answers
│
├── phase0_parser.py               # Dataset parsing (docs, queries, qrels)
├── phase1_preprocessor.py         # Text preprocessing pipeline
├── phase2_index.py                # Inverted index construction
├── phase3_4_tfidf_retrieval.py    # TF-IDF matrix + cosine similarity
├── phase5_evaluation.py           # MAP and NDCG evaluation
├── phase7_rag.py                  # RAG with Google Gemini
├── main.py                        # Main orchestrator
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## ⚡ Quick Start

### 1. Clone & enter the project

```bash
git clone https://github.com/your-username/ir-search-engine.git
cd ir-search-engine
```

### 2. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Cranfield data files

Place the four Cranfield files inside the `data/` folder:
```
data/
├── cran.all.1400
├── cran.qry
├── cran.qrel
└── cranqrel.readme
```

### 5. Set your Gemini API key *(only needed for RAG)*

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."

# macOS / Linux
export GEMINI_API_KEY="AIza..."
```

### 6. Run

```bash
# Full pipeline (IR + RAG)
python main.py

# Skip RAG — no API key needed
python main.py --no-rag

# Custom options
python main.py --k 20 --rag-queries 1 2 3 5 10 --rag-top-k 5
```

### Expected output

```
============================================================
  Phase 0 — Parsing Cranfield Dataset
============================================================
[Phase 0] Parsed 1400 documents.
[Phase 0] Parsed 225 queries.
[Phase 0] Parsed 1,837 relevance judgments (0 lines skipped).

...

==================================================
  EVALUATION RESULTS (top-1400)
==================================================
  MAP  (binary relevance) : 0.2976
  NDCG@20 (graded relev.) : 0.3988
==================================================

  Total runtime : 43.4s
```

---

## 🔬 Phase Details

### Phase 0 — Dataset Parsing

Parses the three Cranfield files into Python dictionaries.

| File | Format | Output |
|------|--------|--------|
| `cran.all.1400` | `.I` `.T` `.A` `.B` `.W` markers | `{doc_id: {title, abstract, content}}` |
| `cran.qry` | `.I` `.W` markers | `{query_id: query_text}` |
| `cran.qrel` | `qid did score` (3 or 4 col) | `{(qid, did): score}` |

**Cranfield relevance scale (reversed — 1 = highest):**

| Score | Meaning | Binary (MAP) | Graded gain (NDCG) |
|-------|---------|:---:|:---:|
| 1 | Most relevant | ✓ | 4 |
| 2 | Relevant | ✓ | 3 |
| 3 | Marginally relevant | ✓ | 2 |
| 4 | Marginally irrelevant | ✓ | 1 |
| -1 | Irrelevant | ✗ | 0 |
| missing | Not judged | ✗ | 0 |

---

### Phase 1 — Text Preprocessing

The **same** 4-step pipeline is applied to both documents and queries:

```
Raw text
   │
   ▼  Step 1: Lowercase
"Aerodynamic Heating"  →  "aerodynamic heating"
   │
   ▼  Step 2: Tokenize  regex: \b[a-z]{2,}\b
["aerodynamic", "heating"]
   │
   ▼  Step 3: Remove stopwords  (NLTK English, 179 words)
["aerodynamic", "heating"]   ← "the", "of", "in" removed
   │
   ▼  Step 4: Porter stemming
["aerodynam", "heat"]
```

**Example — Query 1:**
```
Input:  "what similarity laws must be obeyed when constructing aeroelastic models of heat"
Output: ["similar", "law", "obey", "construct", "aeroelast", "model", "heat"]
```

**Statistics:**

| Metric | Value |
|--------|-------|
| Total tokens after preprocessing | 138,732 |
| Average tokens per document | 99.1 |
| Vocabulary size (unique stems) | 4,267 |

---

### Phase 2 — Inverted Index

Maps every term to the documents containing it:

```python
inverted_index = {
    "aerodynam": {1: 2, 12: 5, 51: 3, 184: 4, ...},  # doc_id: count
    "similar":   {51: 1, 486: 4, 573: 2, ...},
    "heat":      {5: 3, 6: 2, 144: 4, 399: 2, 485: 6, ...},
    # 4,267 terms total
}
```

**Top terms by document frequency:**

| Term | # Documents | % of corpus |
|------|:-----------:|:-----------:|
| flow | 730 | 52.1% |
| result | 691 | 49.4% |
| number | 570 | 40.7% |
| pressur | 551 | 39.4% |
| effect | 539 | 38.5% |

> High-frequency terms like "flow" get very low IDF weights — they appear everywhere and discriminate nothing.

---

### Phase 3 — TF-IDF Weighting

Builds a sparse document-term matrix **(1400 × 4267)** with L2-normalized rows.

**Formulas:**

```
TF(t, d)  =  1 + log₁₀(count(t, d))          log-normalized term frequency
IDF(t)    =  log₁₀(N / df(t))                standard IDF  (N = 1400)
w(t, d)   =  TF(t, d) × IDF(t)               TF-IDF weight
v̂(d)      =  v(d) / ‖v(d)‖₂                  L2 normalization
```

**Example calculation for "aeroelast" in Doc 486:**

```
count = 4 occurrences
df    = 35 documents contain "aeroelast"

TF    = 1 + log₁₀(4) = 1 + 0.602 = 1.602
IDF   = log₁₀(1400/35) = log₁₀(40) = 1.602
w     = 1.602 × 1.602 = 2.566   ← high weight, rare informative term
```

**Matrix statistics:**

```
Shape:      1400 docs × 4267 terms  =  5,973,800 cells total
Non-zeros:  83,094                  =  1.4% density
Storage:    Sparse CSR format (scipy)
```

---

### Phase 4 — Cosine Similarity Retrieval

After L2 normalization, cosine similarity reduces to a dot product:

```python
# Shape: (1400,) — one score per document
scores = tfidf_matrix.dot(query_vector)

# Full ranking for evaluation (k = 1400)
top_indices = np.argsort(scores)[::-1]
```

> **Key design decision:** Evaluation uses `k=1400` (all documents ranked).
> MAP requires finding ALL relevant documents — capping at k=20 caused MAP to drop from **0.2976 → 0.003** (100× lower) because relevant documents missed below rank 20 still count against the score.

---

### Phase 5 — Evaluation (MAP + NDCG)

**MAP — Mean Average Precision:**

```
For each query q:
  AP(q) = Σ [Precision@k × rel(k)] / |R_q|

  where:
    Precision@k = relevant found in top-k / k
    rel(k)      = 1 if doc at rank k is relevant, else 0
    |R_q|       = TOTAL relevant docs for q (including unfound ones)

MAP = mean(AP) over all 225 queries
```

**NDCG — Normalized Discounted Cumulative Gain:**

```
DCG@k   = Σᵢ₌₁ᵏ  gain(i) / log₂(i+1)
IDCG@k  = DCG of ideal (perfect) ranking
NDCG@k  = DCG@k / IDCG@k   ∈ [0, 1]

gain(i) uses converted Cranfield scores: 1→4, 2→3, 3→2, 4→1, -1→0
```

---

### Phase 7 — RAG Answer Generation

```python
# 1. Take top-5 retrieved documents
top_docs = retrieval_results[query_id][:5]

# 2. Build structured prompt
prompt = f"""
USER QUERY: {query}

RETRIEVED DOCUMENTS:
[Document 1 | ID: 51 | Score: 0.253]
Title: Theory of aircraft structural models...
Abstract: ...

INSTRUCTIONS:
- Answer using ONLY the retrieved documents
- Cite document IDs as [Doc N]
- Be technically precise
"""

# 3. Call Gemini API
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=prompt
)
```

---

## 📄 Sample Outputs

### Query 1 — Similarity Laws for Aeroelastic Models

```
Query: "what similarity laws must be obeyed when constructing aeroelastic
        models of heated high speed aircraft"

Rank  Doc    Score   Title
───────────────────────────────────────────────────────────────────────
 #1   51    0.2533   Theory of aircraft structural models subjected to
                     aerodynamic heating and external loads
 #2   878   0.2123   Experimental model techniques and equipment for
                     flutter investigations
 #3   486   0.2111   Similarity laws for aerothermoelastic testing
 #4   573   0.2053   Viscous hypersonic similitude
 #5   12    0.1951   Some structural and aeroelastic considerations of
                     high speed flight
```

### Query 3 — Heat Conduction in Composite Slabs  *(AP = 0.659)*

```
Query: "what problems of heat conduction in composite slabs have been solved so far"

Rank  Doc    Score   Title
───────────────────────────────────────────────────────────────────────
 #1   485   0.4776   Linear heat flow in a composite slab
 #2   399   0.3916   Conduction of heat in composite slabs
 #3   5     0.3700   One-dimensional transient heat conduction into a
                     double-layer slab
 #4   144   0.2951   Heat flow in composite slabs
 #5   181   0.2594   Some problems on heat conduction in stratiform bodies
```

---

## 🤖 RAG Answers

### Query 1

> **When constructing aeroelastic models of heated high-speed aircraft, the following similarity laws must be obeyed:**
>
> 1. **Derivation Basis** — Similarity laws are obtained by nondimensionalizing the governing equations for aerodynamic flow, internal heat conduction, and structural stress-deflection [Doc 3].
> 2. **Material and Thermal Similarity** — If constructed from the same materials and at the same temperature, the model will be thermally similar with respect to heat flow and external loads [Doc 1].
> 3. **Challenges with Scale Ratio** — Achieving full similitude is very difficult when scale ratio is other than unity due to conflicts between Mach number, Reynolds number, aeroelastic parameter, and thermal parameters [Doc 3].
> 4. **Mitigation Strategies** — Methods include incomplete aerothermoelastic testing and restricted-purpose models for specific phenomena [Doc 3].

### Query 10

> **Yes, real-gas transport properties for air are available over a wide range of enthalpies and densities:**
>
> - Viscosity and thermal conductivity tabulated from **500K to 15,000K** over pressures **0.0001–100 atm** [Doc 1]
> - Transport properties for partially ionized air at velocities **25,000–40,000 ft/sec** [Doc 4]
> - Charts for equilibrium flow up to **10,000 btu/lb stagnation enthalpy** and **1,000 atm** stagnation pressure [Doc 5]

---

## 📦 Dataset

The [Cranfield collection](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/) is the oldest and most-used IR benchmark dataset.

```
cran.all.1400   1,400 aeronautical research paper abstracts
cran.qry        225 natural language user queries
cran.qrel       1,837 human relevance judgments (query-document pairs)
cranqrel.readme Explanation of the relevance scale
```

**Relevance judgments distribution:**

```
225 queries × avg 8.2 relevant docs per query = 1,837 total judgments
Unique documents judged: 924 out of 1,400
Documents not in qrel  = treated as non-relevant (per Cranfield convention)
```

---

## 📐 Evaluation

**MAP result breakdown:**

```
Overall MAP: 0.2976

Queries with perfect AP = 1.0:  2  (queries 9 and 119)
Queries with AP > 0.5:         62
Queries with AP > 0.2:        118
Queries with AP = 0.0:          3  (queries 13, 22, 44)
```

**Why MAP = 0.2976 is a good result:**

| System | MAP | Notes |
|--------|:---:|-------|
| Random ranking | ~0.01 | No retrieval at all |
| Our TF-IDF | **0.2976** | This project |
| BM25 (probabilistic) | ~0.35 | Better but not VSM |
| Neural (BERT-based) | ~0.45+ | Out of scope |

---

## 📦 Dependencies

```txt
nltk>=3.8.1          # Tokenization, stopwords, Porter stemmer
numpy>=1.26.0        # Array operations
scipy>=1.12.0        # Sparse matrices (CSR format)
scikit-learn>=1.4.0  # Evaluation utilities
google-genai>=1.0.0  # Gemini API for RAG
```

Install all at once:

```bash
pip install -r requirements.txt
```

**NLTK data** (downloaded automatically on first run):
```python
nltk.download('stopwords')
nltk.download('punkt')
```

---

## ⚙️ CLI Reference

```
python main.py [OPTIONS]

Options:
  --data-dir DIR        Path to Cranfield data files  (default: data/)
  --k INT               Top-k documents for display and RAG  (default: 20)
  --no-rag              Skip Phase 7 RAG generation
  --rag-queries INT...  Query IDs to run RAG on  (default: 1 2 3 5 10)
  --rag-top-k INT       Documents fed to LLM per query  (default: 5)
  --output DIR          Output folder for JSON results  (default: results/)
```

**Examples:**

```bash
# Run everything with defaults
python main.py

# Only IR — no RAG, no API key needed
python main.py --no-rag

# RAG on 10 specific queries
python main.py --rag-queries 1 3 9 10 15 20 25 30 50 100

# Test individual phases
python phase0_parser.py
python phase1_preprocessor.py
python phase3_4_tfidf_retrieval.py
python phase5_evaluation.py
```

---

## 👥 Course Information

| | |
|--|--|
| **Course** | Information Retrieval (AI356) |
| **Academic Year** | Spring 2025-2026 |
| **Institution** | Jordan University of Science and Technology |
| **Instructor** | Dr. Abdullah Al-Amaren |
| **Dataset** | Cranfield Benchmark Collection |
| **Submission** | 08/05/2026 |

---

<div align="center">

*Built with Python · Evaluated on Cranfield · Powered by Gemini 2.5 Flash*

</div>
