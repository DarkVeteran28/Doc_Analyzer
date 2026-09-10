# Doc Analyzer — Hybrid RAG Pipeline

Document Q&A pipeline combining **vector retrieval** (ChromaDB + MiniLM), **BM25
keyword retrieval**, and **Reciprocal Rank Fusion (RRF)** for hybrid search.
Answers are generated with **Ollama `qwen3:8b`** (local) or **Google Gemini** (cloud), selected via the `LLM_PROVIDER` environment variable.

## Architecture

```
PDF upload
    │
    ▼
extract_text_from_pdf  →  chunk_pages (600 words, 100 overlap)
    │
    ├─► MiniLM embeddings  →  ChromaDB (chroma_db/)
    │
    └─► BM25Index          →  JSON per document (bm25_index/)
                │
                ▼
         ask_question(question)
                │
    ┌───────────┴───────────┐
    │  retrieval_mode       │
    │  (default: hybrid)    │
    └───────────┬───────────┘
                │
    vector ─────┼───── bm25
                │
                ▼
      reciprocal_rank_fusion (RRF, k=60)
                │
                ▼
      generate_answer (Ollama qwen3:8b)
                │
                ▼
      build_sources (one source per page)
```

## Why Hybrid Retrieval?

| Component | Role |
|-----------|------|
| **Vector (MiniLM + ChromaDB)** | Semantic similarity; strong on paraphrased questions |
| **BM25** | Exact keywords, IDs, numbers, rare terms |
| **RRF fusion** | Combines both ranked lists without score normalization; deduplicates by stable `chunk_id` |

Hybrid is the **default** because checkpoint evaluations show BM25 and vector
rankings often diverge; RRF preserves both signals. On the expanded evaluation
set, BM25 and Hybrid tie as strongest; Vector trails slightly on semantic
Recall@1. See `rag/evaluation/ANALYSIS.md` for honest comparison.

## Public API

`rag_service.py` exposes two functions (unchanged signatures):

```python
from rag_service import process_document, ask_question

# Index a PDF
process_document(
    pdf_path="document.pdf",
    document_id="my_doc",
    persist_directory="chroma_db",  # optional
)

# Ask a question (hybrid retrieval by default)
response = ask_question(
    document_id="my_doc",
    question="What did the experiment show?",
    n_results=3,
    persist_directory="chroma_db",  # optional
    retrieval_mode="hybrid",        # optional: vector | bm25 | hybrid
)

# response = {"answer": str, "sources": [{"page": int, "text": str}, ...]}
```

## Retrieval Modes

| Mode | Description |
|------|-------------|
| `hybrid` | RRF fusion of vector + BM25 (default) |
| `vector` | ChromaDB cosine similarity only |
| `bm25` | rank-bm25 keyword search only |

If no BM25 index exists, hybrid falls back to vector-only results.

## Project Structure

```
rag/
  ingestion.py       PDF text extraction
  chunking.py          Page → overlapping word chunks
  embeddings.py        all-MiniLM-L6-v2 embeddings
  vector_store.py      ChromaDB persistence
  bm25.py              Document-scoped BM25 index (cached)
  fusion.py            Reciprocal Rank Fusion
  retrieval.py         Vector / BM25 / hybrid retrieval
  generation.py        Ollama prompt + answer generation
  sources.py           Page-level source citations
  rag_pipeline.py      process_document + ask_question orchestration
  evaluation/
    dataset.json       25-query evaluation set
    metrics.py         Recall@K, MRR
    run_evaluation.py  Run full mode comparison
    ANALYSIS.md        Honest Vector vs BM25 vs Hybrid analysis
  tests/               Unit and integration tests
rag_service.py         Public compatibility wrapper
requirements-rag.txt   rank-bm25 dependency
description.txt        Cumulative development log
```

## LLM Providers

| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama` / `gemini` | `ollama` |
| `GEMINI_API_KEY` | your Gemini API key | — (required when `gemini`) |
| `GEMINI_MODEL` | any Gemini model name | `gemini-2.5-flash` |

**Local development (Ollama):**

```bash
# default — no extra variables needed
LLM_PROVIDER=ollama
```

**Cloud deployment (Gemini):**

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your-secret-key>   # never commit this
GEMINI_MODEL=gemini-2.5-flash      # optional
```

Copy `.env.example` to `.env` and fill in your key. `.env` is git-ignored.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with `qwen3:8b` pulled
- Dependencies: `pypdf`, `sentence-transformers`, `chromadb`, `ollama`, `numpy`, `pytest`, `rank-bm25`

### Install

```bash
pip install pypdf sentence-transformers chromadb ollama numpy pytest
pip install -r requirements-rag.txt

ollama pull qwen3:8b
```

### Environment

```bash
export PYTHONPATH=.
```

## Running the Pipeline

```python
from rag_service import process_document, ask_question

result = process_document("rag/tests/sample.pdf", "sample_document")
print(result)  # {"document_id", "pages", "chunks"}

response = ask_question("sample_document", "Who is the favorite Formula 1 driver?")
print(response["answer"])
for source in response["sources"]:
    print(f"Page {source['page']}: {source['text'][:120]}...")
```

## Testing

### Deterministic tests (no LLM)

```bash
PYTHONPATH=. pytest rag/tests/ \
  --ignore=rag/tests/test_generation_llm.py \
  --ignore=rag/tests/test_rag_pipeline.py -q
```

### End-to-end test (requires Ollama)

```bash
PYTHONPATH=. pytest rag/tests/test_rag_pipeline.py -q -s
```

### Gemini generation test (requires GEMINI_API_KEY)

```bash
GEMINI_API_KEY=<key> LLM_PROVIDER=gemini \
    PYTHONPATH=. pytest rag/tests/test_generation_gemini.py -v -s
```

Skipped automatically when `GEMINI_API_KEY` is not set.


### Evaluation benchmark

```bash
PYTHONPATH=. pytest rag/tests/test_evaluation.py -q -s
```

Or programmatically:

```python
from rag.evaluation.run_evaluation import run_evaluation
report = run_evaluation()
print(report)
```

## Evaluation Methodology

- **Dataset:** 25 queries (21 scored + 4 irrelevant) over 10 synthetic chunks
- **Query types:** exact_keyword, semantic, mixed, multi_page, irrelevant
- **Metrics:** Recall@1, Recall@3, Recall@5, MRR
- **Irrelevant queries:** excluded from Recall/MRR (no labeled relevant pages)
- **Modes compared:** vector, bm25, hybrid

Latest aggregate results (see `description.txt` checkpoint 12):

| Mode   | Recall@1 | Recall@3 | MRR   |
|--------|----------|----------|-------|
| Vector | 0.802    | 0.976    | 0.976 |
| BM25   | 0.849    | 1.000    | 1.000 |
| Hybrid | 0.849    | 1.000    | 1.000 |

## Important Design Decisions

1. **Separate stores:** ChromaDB and BM25 persist independently, keyed per document.
2. **Stable chunk IDs:** `{document_id}_{page}_{chunk_number}` shared across stores for RRF dedup.
3. **No confidence gate:** Irrelevant queries may still retrieve in-document chunks; the LLM prompt instructs no-answer when context is insufficient.
4. **BM25 caching:** Indexes are cached in memory per document; invalidated on re-index.
5. **Generation unchanged:** Hybrid retrieval integrates before the existing Ollama layer.

## Branch

Active development: **`RAG`** branch (`origin/RAG`).

## License / Notes

See `description.txt` for full checkpoint history, test counts, and validation status.
