# Retrieval Evaluation Analysis

This analysis compares Vector, BM25, and Hybrid retrieval on the expanded
evaluation dataset (`rag/evaluation/dataset.json`).

**Dataset:** 25 queries (21 scored, 4 irrelevant), 10 chunks, single document  
**Metrics:** Recall@1, Recall@3, Recall@5, MRR  
**Irrelevant queries:** excluded from Recall/MRR (no labeled relevant pages)

## Overall Results

| Mode   | Recall@1 | Recall@3 | Recall@5 | MRR   |
|--------|----------|----------|----------|-------|
| Vector | 0.802    | 0.976    | 1.000    | 0.976 |
| BM25   | 0.849    | 1.000    | 1.000    | 1.000 |
| Hybrid | 0.849    | 1.000    | 1.000    | 1.000 |

## Where Each Method Performs Better

### BM25 and Hybrid vs Vector

- **BM25 and Hybrid outperform Vector on Recall@1** (0.849 vs 0.802).
- On **semantic queries**, Vector missed q09 (laboratory chemistry background)
  at rank 1; BM25 and Hybrid ranked page 2 first (Recall@1 = 1.0 for BM25/Hybrid
  vs 0.833 for Vector).
- At **Recall@3 and MRR**, BM25 and Hybrid reach perfect scores on this dataset;
  Vector remains slightly lower (Recall@3 = 0.976, MRR = 0.976).

### Vector

- **Exact-keyword queries:** Vector matches BM25/Hybrid (Recall@1 = 0.917).
- **Mixed queries:** Vector matches BM25/Hybrid except shared miss on q14
  (multi-chunk methods/calibration query).
- Vector is **competitive but not best** on this synthetic corpus; semantic
  paraphrase is the main area where it trails BM25/Hybrid at rank 1.

### BM25

- Strongest overall on this dataset, especially for **keyword-overlap and
  semantic queries** where salient terms still appear in the chunk text.
- **Does not outperform Hybrid** here; scores are identical across all metrics.

### Hybrid

- **Does not exceed BM25** on this evaluation set. RRF fusion matched BM25
  exactly on every aggregate metric.
- Hybrid **does improve over Vector** on Recall@1 and MRR by combining rankings,
  but on this small corpus BM25 alone already dominates.

## Where Methods Perform Similarly

- **Exact-keyword queries:** All three modes: Recall@1 = 0.917, Recall@3 = 1.0.
  Shared miss: q01 (sample ID 7781 spans pages 3 and 10; top-1 finds only one).
- **Mixed queries:** All three modes identical (Recall@1 = 0.867).
- **Multi-page queries:** All three modes identical at Recall@1 = 0.5 and
  Recall@3 = 0.875–1.0. Multi-page recall@1 is inherently limited when only
  one page appears in the top result.
- **Irrelevant queries:** All modes still return in-document chunks (expected
  behavior without a rejection threshold).

## Limitations

1. **Small synthetic corpus** (10 chunks, 1 document) — not representative of
   production PDF diversity.
2. **Hybrid did not demonstrate superiority** over BM25 on this benchmark; prior
   checkpoint-3 comparisons showed ranking differences, but aggregate metrics
   here favor BM25 and Hybrid equally.
3. **Multi-page queries** penalize Recall@1 by design when multiple pages are
   relevant but only one fits in the top slot.
4. **Irrelevant queries** are not scored; retrieval still returns document text
   because no confidence gate exists (by design).
5. **Embedding and BM25 tokenization** are simple; no stemming or domain tuning.
6. Results depend on **MiniLM-L6-v2** and **rank-bm25** defaults.

## Honest Conclusion

Hybrid retrieval is **justified as a robust default** because it matches the
best-performing mode (BM25) on this evaluation while preserving vector signal
for cases where rankings diverge (observed in checkpoint 3). However, **this
expanded evaluation does not prove Hybrid is universally superior to Vector or
BM25**. BM25 alone is strongest on this dataset; Vector lags slightly on
semantic rank-1 retrieval.
