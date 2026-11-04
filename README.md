# Adaptive Shopping Assistant

**One shopping request, three ways to handle it:** direct product search, grounded recommendations, or a multi-step agent. A rule-based router chooses the simplest pipeline that fits the task. You can also select a pipeline manually and compare their traces in the Streamlit UI.

## When does it use each pipeline?

| Request | Route | What happens |
| --- | --- | --- |
| “Find a solid wood coffee table rated at least 4 stars.” | **Search Only** | One retrieval returns ranked product cards and source evidence. No answer-generation call. |
| “Which accent chair would suit a small room?” | **Standard RAG** | One retrieval builds a fixed candidate set; an LLM recommends from those records, with product IDs and evidence checked against the catalog. |
| “Find a coffee table and matching end tables.” | **Agentic RAG** | A bounded agent can search for a coffee table, inspect its recorded attributes, then use them to search for a complementary item. |

`auto` applies an explainable, rule-first router. More filters alone do not trigger an agent. Recommendation or comparison within one candidate set goes to Standard RAG; coordinating products across categories can go to Agentic RAG when a later search may depend on an earlier result. The UI shows the selected route and reason, and lets you override it.

All three pipelines share the same product catalog and retrieval service. Retrieval can use BM25, MiniLM embeddings with FAISS, or a hybrid of both; metadata filtering and optional Cohere reranking are separate settings. The UI records actual searches, tool calls, latency, and token usage so the extra steps are visible.

## Run it locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
git clone --depth 1 https://github.com/wayfair/WANDS.git data/raw/WANDS
.venv/bin/python scripts/prepare_data.py --full
.venv/bin/python scripts/build_dense_index.py --batch-size 64
.venv/bin/streamlit run app.py
```

The dense-index step downloads the embedding model on its first run. For Standard RAG or Agentic RAG, copy `.env.example` to `.env` and set `LLM_MODEL_ID`, `LLM_API_KEY`, and `LLM_BASE_URL`. Search Only with rule-based parsing needs no LLM key. Cohere reranking requires `COHERE_API_KEY` if enabled.

The same routes are available from the CLI:

```bash
.venv/bin/python scripts/run_pipeline.py 'solid wood coffee table at least 4 stars' --pipeline auto --retriever bm25
.venv/bin/python scripts/run_pipeline.py 'Which accent chair would suit a small room?' --pipeline standard_rag --retriever hybrid
.venv/bin/python scripts/run_pipeline.py 'Find a coffee table and matching end tables' --pipeline agentic_rag --retriever hybrid
```

## Data and limits

The catalog comes from [Wayfair WANDS](https://github.com/wayfair/WANDS), a public product-search dataset with 42,994 products, queries, and relevance labels. The source data, processed catalog, and model files are generated locally and excluded from this repository. WANDS provides historical product descriptions and ratings, but no verified live prices, stock, shipping, photos, or purchase links. The assistant flags requests for those details instead of inventing them.

## Where to look

- `src/shopping_agent/router.py` — route rules and reasons
- `src/shopping_agent/modes.py` — shared pipeline dispatch and traces
- `src/shopping_agent/retrieval.py` — search, filtering, and reranking
- `src/shopping_agent/rag.py` and `agent.py` — grounded answer and agent flows
- `app.py` — interactive UI with manual mode selection and comparison

For implementation details, dataset preparation, and evaluation notes, see [DESIGN.md](DESIGN.md). Run the test suite with `.venv/bin/python -m unittest discover -s tests -v`.
