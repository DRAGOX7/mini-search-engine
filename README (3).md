<div align="center">

# 🔍 Mini Search Engine with RAG Extension

**AI356 — Information Retrieval · Spring 2025-2026**  
**Jordan University of Science and Technology**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-154F5B?style=for-the-badge)](https://nltk.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.12+-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/gemini)
[![MAP Score](https://img.shields.io/badge/MAP-0.2976-2ea44f?style=for-the-badge)](/)
[![NDCG@20](https://img.shields.io/badge/NDCG@20-0.3988-2ea44f?style=for-the-badge)](/)
[![License](https://img.shields.io/badge/License-Academic-orange?style=for-the-badge)](/)

<br/>

> A complete **Information Retrieval system** built on the **Vector Space Model** with **TF-IDF weighting** and **cosine similarity ranking**, evaluated on the Cranfield benchmark, and extended with a **Retrieval-Augmented Generation (RAG)** pipeline powered by Google Gemini 2.5 Flash.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Evaluation Results](#-evaluation-results)
- [Pipeline Architecture](#-pipeline-architecture)
- [Data Flow](#-data-flow)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Phase Details](#-phase-details)
- [TF-IDF Formula Breakdown](#-tf-idf-formula-breakdown)
- [Sample Retrieval Results](#-sample-retrieval-results)
- [RAG Answers](#-rag-answers)
- [Dataset](#-dataset-cranfield-collection)
- [Dependencies](#-dependencies)
- [CLI Reference](#-cli-reference)

---

## 🧭 Overview

This project implements a full IR pipeline in two parts:

```mermaid
graph LR
    A([👤 User Query]) --> B[🔍 TF-IDF Retrieval]
    B --> C[(📄 Top-k Docs)]
    C --> D[🤖 Gemini LLM]
    D --> E([✅ AI Answer])

    style A fill:#4A90D9,stroke:#2E6DA4,color:#fff
    style B fill:#7B68EE,stroke:#5A4FCF,color:#fff
    style C fill:#20B2AA,stroke:#178F88,color:#fff
    style D fill:#FF8C00,stroke:#CC7000,color:#fff
    style E fill:#32CD32,stroke:#228B22,color:#fff
```

| Part | Description |
|------|-------------|
| **Part 1 — Classical IR** | Vector Space Model · TF-IDF · Cosine similarity · Inverted index |
| **Part 2 — RAG Extension** | Top-5 retrieved docs → Gemini 2.5 Flash → Cited natural-language answer |

---

## 📊 Evaluation Results

```mermaid
xychart-beta
    title "Evaluation Scores"
    x-axis ["MAP (Binary)", "NDCG@20 (Graded)", "NDCG@1400 (Full)"]
    y-axis "Score" 0 --> 1
    bar [0.2976, 0.3988, 0.5186]
```

| Metric | Score | Description |
|--------|:-----:|-------------|
| **MAP** | **`0.2976`** | Mean Average Precision — binary relevance (scores 1–4 = relevant) |
| **NDCG@20** | **`0.3988`** | Normalized DCG at rank 20 — graded relevance |
| **NDCG@1400** | **`0.5186`** | Normalized DCG — full 1,400-document ranking |

### Per-Query Performance

```mermaid
pie title Query Performance Distribution (225 queries)
    "AP > 0.5  (62 queries)" : 62
    "AP 0.2–0.5  (56 queries)" : 56
    "AP 0.0–0.2  (104 queries)" : 104
    "AP = 0.0  (3 queries)" : 3
```

| 🏆 Best Queries | AP | 💔 Worst Queries | AP |
|----------------|:--:|-----------------|:--:|
| Query 9 | `1.0000` — Perfect | Query 13 | `0.0000` |
| Query 119 | `1.0000` — Perfect | Query 22 | `0.0000` |
| Query 172 | `0.8542` | Query 44 | `0.0000` |
| Query 3 | `0.6592` | Query 31 | `0.0010` |
| Query 52 | `0.8125` | Query 216 | `0.0020` |

---

## 🏗️ Pipeline Architecture

```mermaid
flowchart TD
    A[📂 Cranfield Files\ncran.all.1400 · cran.qry · cran.qrel] --> B

    subgraph IR["🔵 Classical IR — Vector Space Model"]
        B[Phase 0\n📥 Parse Dataset\n1400 docs · 225 queries · 1837 qrels]
        B --> C[Phase 1\n✂️ Preprocess Text\nlowercase · tokenize · stopwords · stem]
        C --> D[Phase 2\n🗂️ Build Inverted Index\n4267 terms · 83094 postings]
        D --> E[Phase 3\n⚖️ TF-IDF Matrix\n1400×4267 sparse · L2-normalized]
        E --> F[Phase 4\n🎯 Cosine Similarity\nfull 1400-doc ranking]
    end

    subgraph EVAL["🟢 Evaluation"]
        F --> G[Phase 5\n📐 MAP + NDCG\nMAP=0.2976 · NDCG@20=0.3988]
    end

    subgraph RAG["🟠 RAG Extension"]
        F --> H[Phase 7\n🤖 Gemini 2.5 Flash\ntop-5 docs → cited answer]
    end

    style IR fill:#EEF2FF,stroke:#6366F1
    style EVAL fill:#F0FDF4,stroke:#22C55E
    style RAG fill:#FFF7ED,stroke:#F97316
```

---

## 🔄 Data Flow

```mermaid
flowchart LR
    subgraph INPUT["📥 Input"]
        D1[cran.all.1400]
        D2[cran.qry]
        D3[cran.qrel]
    end

    subgraph PARSE["Phase 0 · Parse"]
        P1["docs\n{id: title+abstract}"]
        P2["queries\n{id: text}"]
        P3["qrels\n{(qid,did): score}"]
    end

    subgraph PREPROCESS["Phase 1 · Preprocess"]
        PP1["doc tokens\n138,732 total"]
        PP2["query tokens\nsame pipeline"]
    end

    subgraph INDEX["Phase 2 · Index"]
        IX["inverted index\nterm→{doc:count}\n4,267 terms"]
    end

    subgraph TFIDF["Phase 3 · TF-IDF"]
        TF["sparse matrix\n1400 × 4267\n83,094 non-zeros"]
    end

    subgraph RETRIEVE["Phase 4 · Retrieve"]
        RT["ranked docs\ncosine similarity\ntop-20 display\ntop-1400 eval"]
    end

    subgraph OUTPUT["📤 Output"]
        O1["evaluation_scores.json\nMAP · NDCG"]
        O2["sample_retrieval.json\ntop-10 results"]
        O3["rag_answers.json\ncited answers"]
    end

    D1 --> P1
    D2 --> P2
    D3 --> P3
    P1 --> PP1
    P2 --> PP2
    PP1 --> IX
    IX --> TF
    TF --> RT
    PP2 --> RT
    P3 --> O1
    RT --> O1
    RT --> O2
    RT --> O3

    style INPUT fill:#DBEAFE,stroke:#3B82F6
    style OUTPUT fill:#DCFCE7,stroke:#16A34A
```

---

## 📁 Project Structure

```
ir_project/
│
├── 📂 data/                          # Cranfield dataset files
│   ├── cran.all.1400                 # 1,400 aerospace documents
│   ├── cran.qry                      # 225 user queries
│   ├── cran.qrel                     # 1,837 relevance judgments
│   └── cranqrel.readme               # Relevance scale documentation
│
├── 📂 results/                       # Auto-generated on run
│   ├── evaluation_scores.json        # MAP, NDCG@20, per-query scores
│   ├── sample_retrieval.json         # Top-10 docs for sample queries
│   └── rag_answers.json              # AI-generated answers + context
│
├── 🐍 phase0_parser.py               # Dataset parsing (docs, queries, qrels)
├── 🐍 phase1_preprocessor.py         # Text preprocessing pipeline
├── 🐍 phase2_index.py                # Inverted index construction
├── 🐍 phase3_4_tfidf_retrieval.py    # TF-IDF matrix + cosine similarity
├── 🐍 phase5_evaluation.py           # MAP and NDCG evaluation
├── 🐍 phase7_rag.py                  # RAG with Google Gemini
├── 🐍 main.py                        # Main pipeline orchestrator
├── 📄 requirements.txt               # Python dependencies
└── 📖 README.md                      # This file
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- The four Cranfield data files in `data/`
- A Google Gemini API key *(only for Phase 7 RAG)*

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ir-search-engine.git
cd ir-search-engine

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set Gemini API key (only needed for RAG)
# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."
# macOS / Linux
export GEMINI_API_KEY="AIza..."
```

### Run

```bash
# Full pipeline — IR + evaluation + RAG
python main.py

# IR + evaluation only — no API key needed
python main.py --no-rag

# Custom options
python main.py --k 20 --rag-queries 1 2 3 5 10 --rag-top-k 5

# Test individual phases
python phase0_parser.py
python phase5_evaluation.py
```

### Expected Terminal Output

```
============================================================
  Phase 0 — Parsing Cranfield Dataset
============================================================
[Phase 0] Parsed 1400 documents.
[Phase 0] Parsed 225 queries.
[Phase 0] Parsed 1,837 relevance judgments (0 lines skipped).

============================================================
  Phase 3 — Computing TF-IDF Matrix
============================================================
[Phase 3] Building TF-IDF matrix (1400 docs × 4267 terms)...
[Phase 3] TF-IDF matrix built and L2-normalized | Non-zeros: 83,094

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

```mermaid
flowchart LR
    F1["cran.all.1400\n.I .T .A .B .W\nmarkers"] -->|parse_documents| R1["{doc_id:\n  title,\n  abstract,\n  content}"]
    F2["cran.qry\n.I .W\nmarkers"] -->|parse_queries| R2["{query_id:\n  text}"]
    F3["cran.qrel\nqid did score\n3 or 4 col"] -->|parse_qrels| R3["{(qid, did):\n  score}"]

    style F1 fill:#DBEAFE,stroke:#3B82F6
    style F2 fill:#DBEAFE,stroke:#3B82F6
    style F3 fill:#DBEAFE,stroke:#3B82F6
    style R1 fill:#D1FAE5,stroke:#059669
    style R2 fill:#D1FAE5,stroke:#059669
    style R3 fill:#D1FAE5,stroke:#059669
```

**Relevance scale (reversed — 1 = highest relevance):**

| Score | Meaning | Binary MAP | Graded NDCG Gain |
|:-----:|---------|:----------:|:----------------:|
| `1` | Most relevant | ✅ relevant | `4` |
| `2` | Relevant | ✅ relevant | `3` |
| `3` | Marginally relevant | ✅ relevant | `2` |
| `4` | Marginally irrelevant | ✅ relevant | `1` |
| `-1` | Irrelevant | ❌ not relevant | `0` |
| `—` | Not judged | ❌ not relevant | `0` |

> ⚠️ The Cranfield scale is **reversed** — score `1` is the **best** relevance, not the worst.

---

### Phase 1 — Text Preprocessing

```mermaid
flowchart TD
    A["Raw text\n'What Similarity Laws Must Be Obeyed\nwhen Constructing Aeroelastic Models'"] 
    --> B["① Lowercase\n'what similarity laws must be obeyed\nwhen constructing aeroelastic models'"]
    --> C["② Tokenize  regex: \\b[a-z]{2,}\\b\n['what','similarity','laws','must','be',\n'obeyed','when','constructing','aeroelastic','models']"]
    --> D["③ Remove Stopwords  179 NLTK words\n['similarity','laws','obeyed',\n'constructing','aeroelastic','models']\n❌ removed: what · must · be · when"]
    --> E["④ Porter Stemming\n['similar','law','obey',\n'construct','aeroelast','model']"]

    style A fill:#FEF3C7,stroke:#D97706
    style B fill:#DBEAFE,stroke:#3B82F6
    style C fill:#EDE9FE,stroke:#7C3AED
    style D fill:#FEE2E2,stroke:#DC2626
    style E fill:#D1FAE5,stroke:#059669
```

> 🔑 The **exact same pipeline** is applied to both documents and queries. If you stem documents but not queries, no terms will ever match.

**Statistics:**

| Metric | Value |
|--------|------:|
| Total tokens after preprocessing | `138,732` |
| Average tokens per document | `99.1` |
| Vocabulary size (unique stems) | `4,267` |

---

### Phase 2 — Inverted Index

```mermaid
flowchart LR
    subgraph DOCS["Documents (after preprocessing)"]
        D1["Doc 51\n[aircraft, structural,\n model, aerodynam, heat]"]
        D2["Doc 486\n[similar, law, aeroelast,\n heat, test]"]
        D3["Doc 573\n[viscous, hyperson,\n similar, flow]"]
    end

    subgraph INDEX["Inverted Index"]
        I1["'similar'\n→ {51:1, 486:4, 573:2, ...}"]
        I2["'aeroelast'\n→ {51:2, 486:3, 875:1, ...}"]
        I3["'heat'\n→ {5:3, 51:2, 486:2, 144:4, ...}"]
    end

    D1 -->|index| I1
    D1 -->|index| I2
    D2 -->|index| I1
    D2 -->|index| I2
    D3 -->|index| I1

    style DOCS fill:#EEF2FF,stroke:#6366F1
    style INDEX fill:#FFF7ED,stroke:#F97316
```

**Top terms by document frequency:**

| Rank | Term | Documents | % Corpus | IDF Weight |
|:----:|------|:---------:|:--------:|:----------:|
| 1 | `flow` | 730 | 52.1% | `0.283` — very low |
| 2 | `result` | 691 | 49.4% | `0.307` — very low |
| 3 | `aerodynam` | ~120 | 8.6% | `1.067` — medium |
| 4 | `aeroelast` | ~35 | 2.5% | `1.602` — high |
| 5 | `similar` | ~15 | 1.1% | `1.970` — very high |

---

### Phase 3 — TF-IDF Weighting

```mermaid
flowchart LR
    A["raw count\ncount(t,d)"] -->|"TF = 1 + log₁₀(count)"| B["TF weight\nlog-normalized"]
    C["doc freq\ndf(t)"] -->|"IDF = log₁₀(N/df)"| D["IDF weight\nrare terms boosted"]
    B --> E(("×"))
    D --> E
    E -->|"TF × IDF"| F["raw weight\nw(t,d)"]
    F -->|"divide by ‖v‖₂"| G["normalized weight\nready for cosine"]

    style A fill:#DBEAFE,stroke:#3B82F6
    style C fill:#DBEAFE,stroke:#3B82F6
    style B fill:#EDE9FE,stroke:#7C3AED
    style D fill:#EDE9FE,stroke:#7C3AED
    style E fill:#FEF3C7,stroke:#D97706
    style F fill:#FEE2E2,stroke:#DC2626
    style G fill:#D1FAE5,stroke:#059669
```

**Worked example — "aeroelast" in Doc 486:**

```
count  =  4 occurrences in Doc 486
df     =  35 documents contain "aeroelast"
N      =  1400 documents total

TF     =  1 + log₁₀(4)        =  1 + 0.602  =  1.602
IDF    =  log₁₀(1400 / 35)    =  log₁₀(40)  =  1.602
w      =  1.602 × 1.602        =  2.566   ← high — rare, informative term
```

**Matrix statistics:**

| Property | Value |
|----------|------:|
| Shape | `1400 × 4267` |
| Total cells | `5,973,800` |
| Non-zero cells | `83,094` |
| Sparsity | `98.6%` empty |
| Storage format | SciPy CSR sparse |

---

### Phase 4 — Cosine Similarity Retrieval

```mermaid
flowchart TD
    Q["Query vector\n(L2-normalized, shape: 4267)"]
    M["TF-IDF matrix\n(L2-normalized, shape: 1400 × 4267)"]
    Q --> DOT(["dot product\n≡ cosine similarity\nafter L2-norm"])
    M --> DOT
    DOT --> S["scores\n(shape: 1400,)\none per document"]
    S --> SORT["argsort descending"]
    SORT --> R1["Rank 1 — most relevant"]
    SORT --> R2["Rank 2"]
    SORT --> RN["..."]
    SORT --> R1400["Rank 1400 — least relevant"]

    style Q fill:#DBEAFE,stroke:#3B82F6
    style M fill:#EDE9FE,stroke:#7C3AED
    style DOT fill:#FEF3C7,stroke:#D97706
    style S fill:#FEE2E2,stroke:#DC2626
    style R1 fill:#D1FAE5,stroke:#059669
```

> ⚠️ **Critical design decision:** Evaluation uses `k = 1400` (all documents ranked).  
> Using `k = 20` caused MAP to drop from **0.2976 → 0.003** (100× lower) because MAP divides by the total number of relevant documents — and cutting off at rank 20 means missing most of them.

---

### Phase 5 — MAP & NDCG Evaluation

```mermaid
flowchart LR
    subgraph MAP["MAP — Binary Relevance"]
        A1["Ranked list\nfor query q"] --> B1["Is doc relevant?\n(score 1–4 = yes)"]
        B1 --> C1["Precision@k\nat each hit"]
        C1 --> D1["AP(q) = Σ P@k / |R_q|"]
        D1 --> E1["MAP = mean AP\nover 225 queries"]
    end

    subgraph NDCG["NDCG — Graded Relevance"]
        A2["Ranked list\nfor query q"] --> B2["Convert score\n1→4, 2→3, 3→2, 4→1"]
        B2 --> C2["DCG@k = Σ gain/log₂(i+1)"]
        C2 --> D2["NDCG@k = DCG/IDCG"]
        D2 --> E2["Mean NDCG@k\nover 225 queries"]
    end

    style MAP fill:#EFF6FF,stroke:#3B82F6
    style NDCG fill:#F0FDF4,stroke:#22C55E
```

---

### Phase 7 — RAG Answer Generation

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant IR as 🔍 TF-IDF Engine
    participant P as 📝 Prompt Builder
    participant G as 🤖 Gemini 2.5 Flash
    participant O as 📄 Output

    U->>IR: Natural language query
    IR->>IR: Preprocess → vectorize → cosine sim
    IR->>P: Top-5 retrieved documents
    P->>P: Format titles + abstracts as context
    P->>G: Structured prompt with instructions
    Note over P,G: "Answer ONLY from documents. Cite [Doc N]."
    G->>O: Cited, grounded technical answer
    O->>U: Answer + source documents saved to JSON
```

---

## ⚗️ TF-IDF Formula Breakdown

```mermaid
graph TD
    A["📊 Term Frequency\nTF = 1 + log₁₀ count"] --> |"Measures local importance"| D
    B["📈 Inverse Doc Freq\nIDF = log₁₀ N/df"] --> |"Penalizes common terms"| D
    D(("✖️")) --> E["TF-IDF Weight\nw = TF × IDF"]
    E --> F["L2 Normalize\nv̂ = v / ‖v‖₂"]
    F --> G["Cosine Similarity\ncos(q,d) = q̂ · d̂"]

    style A fill:#DBEAFE,stroke:#3B82F6,color:#1E3A5F
    style B fill:#EDE9FE,stroke:#7C3AED,color:#3B0764
    style D fill:#FEF3C7,stroke:#D97706,color:#7C2D12
    style E fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D
    style F fill:#ECFDF5,stroke:#10B981,color:#064E3B
    style G fill:#F0F9FF,stroke:#0EA5E9,color:#0C4A6E
```

| Component | Formula | Why this formula |
|-----------|---------|-----------------|
| **TF** | `1 + log₁₀(count)` | Log compresses scale — 100 occurrences isn't 100× more important than 1 |
| **IDF** | `log₁₀(N / df)` | Rare terms get high weight; terms in every doc get near-zero weight |
| **TF-IDF** | `TF × IDF` | Local importance × global rarity = discriminative feature |
| **L2 Norm** | `v / ‖v‖₂` | Removes document length bias; enables cosine via dot product |

---

## 📄 Sample Retrieval Results

### Query 1 — Aeroelastic Similarity Laws

> *"What similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft?"*

| Rank | Doc | Score | Title |
|:----:|:---:|:-----:|-------|
| 🥇 1 | 51 | `0.253` | Theory of aircraft structural models subjected to aerodynamic heating and external loads |
| 🥈 2 | 878 | `0.212` | Experimental model techniques and equipment for flutter investigations |
| 🥉 3 | 486 | `0.211` | Similarity laws for aerothermoelastic testing |
| 4 | 573 | `0.205` | Viscous hypersonic similitude |
| 5 | 12 | `0.195` | Some structural and aeroelastic considerations of high speed flight |

---

### Query 3 — Heat Conduction in Composite Slabs *(AP = 0.659)*

> *"What problems of heat conduction in composite slabs have been solved so far?"*

| Rank | Doc | Score | Title |
|:----:|:---:|:-----:|-------|
| 🥇 1 | 485 | `0.478` | Linear heat flow in a composite slab |
| 🥈 2 | 399 | `0.392` | Conduction of heat in composite slabs |
| 🥉 3 | 5 | `0.370` | One-dimensional transient heat conduction into a double-layer slab |
| 4 | 144 | `0.295` | Heat flow in composite slabs |
| 5 | 181 | `0.259` | Some problems on heat conduction in stratiform bodies |

---

### Query 10 — Real-Gas Transport Properties *(AP = 0.250)*

> *"Are real-gas transport properties for air available over a wide range of enthalpies and densities?"*

| Rank | Doc | Score | Title |
|:----:|:---:|:-----:|-------|
| 🥇 1 | 302 | `0.269` | Approximations for the thermodynamic and transport properties of high temperature air |
| 🥈 2 | 493 | `0.248` | Real-gas laminar boundary layer skin friction and heat transfer |
| 🥉 3 | 1143 | `0.233` | A one-foot hypervelocity shock tunnel with high-enthalpy real gas flows |
| 4 | 524 | `0.219` | Stagnation point heat transfer in partially ionized air |
| 5 | 949 | `0.214` | Charts for equilibrium flow properties of air in hyper-velocity nozzles |

---

## 🤖 RAG Answers

### Query 1 — Similarity Laws

> **When constructing aeroelastic models of heated high-speed aircraft, the following similarity laws must be obeyed:**
>
> 1. **Derivation Basis** — Similarity laws are obtained by nondimensionalizing the governing equations for aerodynamic flow, internal heat conduction, and structural stress-deflection **[Doc 3]**.
> 2. **Material and Thermal Similarity** — If constructed from the same materials and at the same temperature, the model will be thermally similar with respect to heat flow and external loads **[Doc 1]**.
> 3. **Challenges with Scale Ratio** — Achieving full similitude is very difficult when scale ratio is other than unity due to conflicts between Mach number, Reynolds number, aeroelastic parameter, and thermal parameters **[Doc 3]**.
> 4. **Mitigation Strategies** — Methods include incomplete aerothermoelastic testing and restricted-purpose models for specific phenomena **[Doc 3]**.

### Query 3 — Heat Conduction

> **Solved problems in heat conduction in composite slabs include:**
>
> - **Linear heat flow** — determining temperature as a function of position and time where external surface temperatures are prescribed functions **[Doc 1]**
> - **Total heat quantity** — method incorporating surface resistance as an additional layer **[Doc 2]**
> - **One-dimensional transient conduction** — analytic solutions for double-layer slabs with triangular heat input **[Doc 3]**
> - **Rocket engine composite walls** — Fourier series solutions for refractory + metallic media under uncooled rocket conditions **[Doc 4]**

### Query 10 — Real-Gas Properties

> **Yes, real-gas transport properties for air are available over a wide range:**
>
> - Viscosity and thermal conductivity tabulated from **500K to 15,000K** at pressures **0.0001–100 atm** **[Doc 1]**
> - Transport properties for partially ionized air at **25,000–40,000 ft/sec** **[Doc 4]**
> - Charts for equilibrium flow up to **10,000 btu/lb** enthalpy and **1,000 atm** stagnation pressure **[Doc 5]**

---

## 📦 Dataset — Cranfield Collection

The [Cranfield collection](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/) is the oldest and most widely used IR benchmark dataset, established in the 1960s.

```mermaid
pie title Dataset Composition
    "Documents used for indexing" : 1400
    "Queries evaluated" : 225
    "Relevance judgments" : 1837
```

| File | Description | Count |
|------|-------------|------:|
| `cran.all.1400` | Aeronautical research abstracts | `1,400` documents |
| `cran.qry` | Natural language user queries | `225` queries |
| `cran.qrel` | Human relevance judgments | `1,837` pairs |

**Coverage:**
- Queries with judgments: **225 / 225** (100%)
- Unique documents judged: **924 / 1,400** (66%)
- Avg relevant docs per query: **~8.2**
- Documents NOT in qrel → treated as **non-relevant** (Cranfield convention)

---

## 📦 Dependencies

```txt
nltk>=3.8.1          # Tokenization · stopwords · Porter stemmer
numpy>=1.26.0        # Array operations
scipy>=1.12.0        # Sparse CSR matrices
scikit-learn>=1.4.0  # Evaluation utilities
google-genai>=1.0.0  # Gemini API for RAG phase
```

```bash
pip install -r requirements.txt
```

> NLTK data (`stopwords`, `punkt`) is downloaded automatically on first run.

---

## ⚙️ CLI Reference

```bash
python main.py [OPTIONS]

Options:
  --data-dir  DIR       Cranfield data folder       (default: data/)
  --k         INT       Top-k docs for display/RAG  (default: 20)
  --no-rag              Skip Gemini RAG generation
  --rag-queries INT...  Query IDs to run RAG on     (default: 1 2 3 5 10)
  --rag-top-k INT       Docs fed to LLM per query   (default: 5)
  --output    DIR       Output folder               (default: results/)
```

**Common usage patterns:**

```bash
# Run everything
python main.py

# IR evaluation only — no API key needed
python main.py --no-rag

# RAG on 10 specific queries
python main.py --rag-queries 1 3 9 15 25 50 75 100 150 200

# Test individual modules
python phase0_parser.py         # Check dataset parsing
python phase1_preprocessor.py  # Check tokenization/stemming
python phase3_4_tfidf_retrieval.py  # Check retrieval results
python phase5_evaluation.py    # Check MAP/NDCG scores
python phase7_rag.py           # Test RAG on 3 queries
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
| **Submission Date** | 08/05/2026 |

---

<div align="center">

**Built with Python · Evaluated on Cranfield · Powered by Gemini 2.5 Flash**

*Information Retrieval (AI356) · Spring 2025-2026*

</div>
