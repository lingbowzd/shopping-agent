# WANDS shopping search, Standard RAG, and Agentic RAG

This repository prepares the official [Wayfair WANDS](https://github.com/wayfair/WANDS) catalog and searches it with BM25, English dense embeddings, or BM25+dense reciprocal rank fusion (RRF). Optional Cohere reranking works after any backend. Four selections share one catalog and retrieval service: Auto, Search Only, Standard RAG, and Agentic RAG. The active catalog contains all **42,994** products; the original 3,000-product quickstart sample is retained separately. The current specification is [DESIGN.md](DESIGN.md) (v1.5).

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
git clone --depth 1 https://github.com/wayfair/WANDS.git data/raw/WANDS
.venv/bin/python scripts/inspect_wands.py
.venv/bin/python scripts/prepare_data.py --output-dir data/processed/quickstart --seed 42 --target 3000
.venv/bin/python scripts/prepare_data.py --full
.venv/bin/python scripts/build_dense_index.py --batch-size 64
.venv/bin/python scripts/try_search.py 'modern coffee table' --top-k 3
.venv/bin/python scripts/compare_search_modes.py --query-id 1 --top-k 5
.venv/bin/python scripts/run_pipeline.py 'smart coffee table' --pipeline search_only --retriever hybrid --top-k 5
.venv/bin/python scripts/run_pipeline.py 'smart coffee table' --pipeline standard_rag --retriever hybrid --top-k 5
.venv/bin/python scripts/evaluate_retrievers.py --k 10
.venv/bin/python scripts/compare_search_modes.py --query-id 1 --top-k 5 --rerank
.venv/bin/python scripts/evaluate_retrievers.py --k 10 --cohere-queries 480
.venv/bin/python scripts/evaluate_bm25.py --k 10
.venv/bin/python -m unittest discover -s tests -v
```

The source clone and processed data are ignored by Git. `prepare_data.py --full` writes the active `data/processed/products.jsonl` and `data/processed/manifest.json`; the CLI searches this full catalog by default. The `--output-dir data/processed/quickstart` command retains the 3,000-product sample. Re-running that sample on the same source commit with the same seed gives the same selected product IDs. Each manifest records the source commit and product file hash, selected classes, counts, missing fields, and selected ID hash.

The dense build downloads the English [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model on its first run. It writes normalized 384-dimensional vectors, `products.faiss`, and an index manifest under `data/processed/indexes/minilm_l6/`. Later searches load the persisted index and reject it if the catalog text or product order changed. On this Mac, keeping `OMP_NUM_THREADS=1` avoids a PyTorch crash during batch encoding; the code sets that default. The embedding model cache is under `data/models/`.

After the first model download, `HF_HUB_OFFLINE=1` can be used with the build and search commands to use the cached MiniLM files. The index is specific to the current full catalog and must be rebuilt after changing `products.jsonl`.

## Observed source schema

All three `.csv` files are **tab-separated** and loaded with `pd.read_csv(..., sep="\t")`. At source commit `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`:

| File | Rows | Actual columns |
| --- | ---: | --- |
| `product.csv` | 42,994 | `product_id`, `product_name`, `product_class`, **`category hierarchy`**, `product_description`, `product_features`, `rating_count`, `average_rating`, `review_count` |
| `query.csv` | 480 | `query_id`, `query`, `query_class` |
| `label.csv` | 233,448 | `id`, `query_id`, `product_id`, `label` |

The raw `category hierarchy` name contains a space. The processed record also exposes `category_hierarchy` for convenient access. Original source fields remain in each record's `raw` object, including unmodified `product_features`. The parser splits feature segments on `|` and only the first `:`, retains duplicate keys as lists, and skips malformed segments without changing the raw string. Blank/NaN source fields become JSON `null`.

Full mode includes every source product, including rows with missing classes, descriptions, or ratings. The optional 3,000-product sample covers eleven actual classes: Coffee & Cocktail Tables, Accent Chairs, Desks, Bookcases, Table Lamps, End Tables, Dining Tables, Beds, Area Rugs, Sofas, and Chandeliers. Quotas are allocated evenly up to each class's available count. Within each class, a fixed-seed sample reserves up to 10% for rows missing a description or feature string; this keeps incomplete records visible for later grounding checks. Bookcases has 208 source rows, so all 208 are included. Other classes contribute 279 or 280 rows each.

## Data flow

`product.csv` → schema validation → full catalog or fixed-seed category sampling → conservative feature parsing → `products.jsonl` → one-document-per-product BM25 and dense indexes → CLI results.

`query.csv` is loaded and validated and supplies real query examples in the manifest. `label.csv` is loaded and validated for source relationships; relevance labels do **not** influence search. A search hit reports the stored WANDS product ID, title, class, backend score, and literal source-field excerpts. A query term absent from a hit is called out explicitly. For dense results, a literal excerpt is product evidence, not an explanation of the embedding similarity. The CLI builds its in-memory BM25 inverted index from the active JSONL file each time it starts; product records are streamed, and only top hits are read back for evidence.

## Retrieval modes

Use `scripts/try_search.py 'query text' --mode bm25|dense|hybrid|hybrid_rerank --top-k 5` or the shared `ProductSearchService.search(query, mode, top_k)` method.

- `bm25`: scores exact tokens in `search_text` and ranks by BM25.
- `dense`: encodes the English query with the same model and searches the normalized vectors using FAISS `IndexFlatIP`. Inner product equals cosine similarity for these unit vectors.
- `hybrid`: takes the top 30 from each method, sums `1/(60+rank)` for each product, and removes duplicate IDs. A product found by both methods receives both contributions.
- `hybrid_rerank`: legacy shorthand for hybrid with Cohere reranking. In the pipeline CLI or UI, enable the independent `--rerank` switch with BM25, dense or hybrid. Set `COHERE_API_KEY` in your environment; the code can also read `LLM_API_KEY` from a local `.env` when `LLM_BASE_URL` points to a Cohere API host. `COHERE_RERANK_MODEL` optionally changes the default `rerank-v5.0-fast`. Candidate count is controlled by `--candidate-k`.

BM25, cosine, RRF, and Cohere relevance scores have different meanings; compare ranks or labeled metrics, not raw scores across modes.

For the genuine WANDS query `query_id=1`, **smart coffee table**, BM25 ranks product **34862** (`sobro smart end table with built-in outlets`) first because its searchable text contains the exact terms. Dense ranks product **10995** (`perfectable cocktail coffee table`) first: its vector is close to the query, although the title does not say “smart.” Hybrid takes **10995** first and **34862** second because both are rank 1 in one input list, with numeric ID breaking the equal RRF score. Cohere reranking moves **33698** (`smart coffee table with storage`) to first place after examining the hybrid candidate texts. These are observed ranks, not explanations of the model internals.

The exact implementation points are `build_dense_index()` and `load_dense_index()` in `src/shopping_agent/dense.py`; `_bm25_ranking()`, `_dense_ranking()`, `rrf_fuse()`, `search()`, and `rank_candidates()` in `src/shopping_agent/retrieval.py`; `cohere_rerank()` in `src/shopping_agent/rerank.py`; and `evaluate_judged_candidates()` in `src/shopping_agent/evaluation.py`.

## Play with the pipelines

Install dependencies, prepare the full WANDS catalog and build the dense index with the Reproduce commands above. Add LLM_MODEL_ID, LLM_API_KEY and LLM_BASE_URL to a local .env for Standard RAG and Agentic RAG; set COHERE_API_KEY only if you enable reranking. Search Only with rule parsing needs no LLM key.

Launch the UI:

    .venv/bin/streamlit run app.py

The sidebar selects Auto, Search Only, Standard RAG or Agentic RAG. Retrieval backend (BM25, dense, hybrid), parsed metadata filtering, Cohere reranking, query parser, Top K and candidate pool are independent controls. Manual category, historical rating and material filters stay active when parsed filters are disabled. The Compare button reruns the same request with the same retrieval settings through the three manual pipelines. Expand the execution trace or download it as JSON.

Try these natural requests in the UI or CLI:

| Goal | Query | Suggested mode |
| --- | --- | --- |
| Find and filter | solid wood coffee table at least 4 stars | Auto or Search Only |
| Inspect unavailable data | glass coffee table under $300 | Search Only |
| Compare known records | Compare product ID 37944 and product ID 25759 by material and rating | Standard RAG |
| Ask for recommendations | Which accent chair would suit a small room? | Standard RAG |
| Coordinate a pair | I'm furnishing my first apartment. I like mid-century modern furniture. Can you help me pick a coffee table and matching end tables? | Agentic RAG |
| Coordinate three categories | Help me find a set of dining table, chairs and sofa and the style matches | Auto or Agentic RAG |

CLI examples:

    .venv/bin/python scripts/run_pipeline.py 'solid wood coffee table at least 4 stars' --pipeline auto --retriever bm25 --top-k 5
    .venv/bin/python scripts/run_pipeline.py 'Which accent chair would suit a small room?' --pipeline standard_rag --retriever hybrid
    .venv/bin/python scripts/run_pipeline.py "I'm furnishing my first apartment. I like mid-century modern furniture. Can you help me pick a coffee table and matching end tables?" --pipeline agentic_rag --retriever hybrid
    .venv/bin/python scripts/run_pipeline.py 'glass coffee table under $300' --pipeline search_only --retriever dense --no-filter --rerank --candidate-k 30
    .venv/bin/python scripts/run_pipeline.py 'Compare product ID 37944 and product ID 25759 by material and rating' --pipeline standard_rag --retriever hybrid

Search Only performs one retrieval and returns actual WANDS IDs, ranked cards, evidence and unavailable requirements without answer generation. Standard RAG performs exactly one retrieval or one batch catalog lookup for explicit IDs, then one LLM generation from a context bounded to eight products and 8,000 characters. It releases at most three recommendations whose IDs and quoted evidence are checked against the retrieved records. The optional LLM query parser is a separate call and its use appears in the trace.

Agentic RAG runs a bounded LangGraph StateGraph with a ToolNode exposing one search_products tool over the same ProductSearchService. A model chooses tool arguments; it can stop after one search or use observed attributes from first-search products to form a second search for a complementary item. Its default limits are six LLM calls, five tool calls and three searches. The trace records actual arguments, results, observed cue, token use, latency and budget use. Auto uses a rule-first router and shows its route reason; a manual mode overrides it.

For the coordinated WANDS example, one observed real run first searched for a mid-century modern coffee table and retrieved product 37944 with a recorded basecolor of acacia. The next search used the observed term in 'acacia end table' and retrieved product 25759. The final answer validator checked both IDs and corrected the second product's cited description field after finding its exact quote in that field. This is one observed trace, not a guaranteed model path. A simple single-product request routes to Search Only.

For the dining-table/chair/sofa request, Auto now selects Agentic RAG rather than applying a Dining Tables-only filter. A real run searched dining table, then used the first result's recorded traditional style in traditional dining chairs and traditional sofa searches. It returned source-backed IDs 15202 (dining table set), 14288 (dining chair), and 2453 (sofa). The products share a WANDS traditional style tag; visual compatibility remains unverified. In one run the final LLM returned no text at its token limit, so the trace recorded a deterministic catalog-evidence fallback instead of claiming an LLM-authored recommendation.

Rules parsing extracts recognized WANDS classes, explicit material and minimum historical rating. The metadata filter reduces eligible IDs before ranking; dense uses precomputed vectors for filtered candidates and FAISS for the unrestricted search. The reranker then reorders the chosen candidate pool when enabled. LLM parsing uses a Pydantic-validated plan and falls back to rules on invalid output. Agentic mode uses its tool arguments and does not make a redundant parser call.

Each run appends one JSON trace to data/processed/logs/pipeline.jsonl. LLM API cost is estimated only if LLM_INPUT_COST_PER_1M and LLM_OUTPUT_COST_PER_1M are set; unknown cost is shown as N/A. WANDS has historical ratings but no verified live price, stock, shipping, photos or purchase links. Dimensions and units that cannot be reliably normalized are flagged as unverified, not guessed.

## Code map

| File | Main responsibility |
| --- | --- |
| src/shopping_agent/retrieval.py | Shared ProductSearchService, metadata candidate selection, BM25/dense/hybrid ranking, optional rerank, explicit-ID batch lookup |
| src/shopping_agent/dense.py | MiniLM embedding and persisted FAISS index |
| src/shopping_agent/query_understanding.py and llm_query_understanding.py | Rules and Pydantic-validated LLM intent plans |
| src/shopping_agent/rag.py | Bounded context, one-pass generation and deterministic source validation |
| src/shopping_agent/agent.py | LangGraph model/tool loop, conditional follow-up search and budgets |
| src/shopping_agent/router.py | Explainable Auto/manual route decisions |
| src/shopping_agent/modes.py and schemas.py | Shared configuration, dispatch and JSONL trace |
| src/shopping_agent/tracing.py | Token, latency and optional API-cost summary |
| scripts/run_pipeline.py and app.py | CLI and interactive Streamlit UI |

## Example output

For `modern coffee table`, the active full catalog returns these real WANDS records:

```text
1. ID 29435 | celentano pedestal coffee table | Coffee & Cocktail Tables | BM25 10.604
   Evidence (description): ...this ultra-modern coffee table...
2. ID 20331 | crotzer coffee table | Coffee & Cocktail Tables | BM25 10.496
   Evidence (description): ...the chrome finish of this table adds to its modern style...
3. ID 38250 | seese coffee table | Coffee & Cocktail Tables | BM25 10.474
   Evidence (description): ...clean lines , and low profile add to its ultra modern feel...
```

Run the CLI for the full, literal excerpts. WANDS contains historical aggregate ratings, but has no live price, stock, shipping, or purchase links.

## Label-based ranking evaluation

Run `.venv/bin/python scripts/evaluate_bm25.py --k 10` to write a per-query report to `data/processed/evaluations/bm25_judged.json`. This is **judged-candidate reranking**: for each WANDS query, BM25 ranks only the products with a non-conflicting label for that query. The BM25 index and inverse document frequencies are still computed over all 42,994 catalog products. Candidate membership comes from the labels, but label values are used only after ranking for scoring. This measures ranking among judged candidates, **not** open-catalog Top-10 retrieval.

The evaluator collapses identical duplicate query-product labels and excludes conflicting pairs. In the current source, 14 pairs conflict. Unjudged products never become `Irrelevant`. Products with zero BM25 score stay in the judged candidate ranking, tied by numeric product ID.

- **nDCG@10:** `Irrelevant=0`, `Partial=1`, `Exact=2`; gain is `2^relevance - 1`, discounted by `log2(rank + 1)`, then divided by the ideal DCG of that query's judged candidates. Queries with no positive judged item are excluded from the mean.
- **MRR@10:** only `Exact` counts as a hit. A query scores `1 / rank` for its first `Exact` in the top 10, or 0 if an `Exact` exists in the candidate set but is below rank 10. Queries with no `Exact` label are excluded from the mean.

Running `scripts/evaluate_retrievers.py --k 10 --cohere-queries 480` on all 480 WANDS queries gives:

| Mode | nDCG@10 (479 eligible) | MRR@10 (379 eligible) |
| --- | ---: | ---: |
| BM25 | 0.7898 | 0.7416 |
| MiniLM + FAISS | 0.7710 | 0.6479 |
| BM25 + dense RRF | 0.7968 | 0.7282 |
| Hybrid + Cohere Rerank | **0.8602** | **0.8270** |

These are judged-candidate ranking metrics under the same label policy. Hybrid improves nDCG slightly over BM25, while BM25 has the higher MRR; Cohere reranking improves both metrics over the other modes in this run. The results do not measure open-catalog recall or production shopping relevance.

The evaluator writes each completed Cohere result to `data/processed/evaluations/cohere_judged_480_k10.checkpoint.json`; rerunning the same command skips completed query IDs. Once complete, it writes all four reports and per-query results to `data/processed/evaluations/retrievers_judged.json`. The checkpoint records query IDs, ranked WANDS product IDs, labels, Cohere scores, and per-query metrics without storing API credentials. For a shorter API test, use `--cohere-queries 20` to select the same 20 seeded query IDs for all four modes.
