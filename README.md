# Development of Advanced AI Applications using RAG and LLM APIs

This project is a self-contained summer-internship-level implementation of an advanced AI application that combines Retrieval-Augmented Generation (RAG), vector search, document ingestion, prompt orchestration, and LLM APIs. The goal is to build a practical assistant that answers questions from a controlled knowledge base while exposing the engineering tradeoffs behind retrieval quality, latency, grounding, and evaluation.

Repository: <https://github.com/archit2501/advanced-rag-llm-app>

## Quickstart

1. Clone the repository, then create and activate a Python environment.

   ```bash
   git clone https://github.com/archit2501/advanced-rag-llm-app.git
   cd advanced-rag-llm-app
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install project dependencies.

   ```bash
   pip install -r requirements-dev.txt
   make browser-install
   ```

3. Configure provider credentials.

   ```bash
   cp .env.example .env
   ```

   The default `LLM_PROVIDER=offline` mode needs no API key. Set `LLM_PROVIDER=openai`
   with `OPENAI_API_KEY` and (optionally) `OPENAI_MODEL=gpt-4o-mini` for the
   OpenAI Responses API, or set `LLM_PROVIDER=ollama` for a local provider.

   Embeddings are configured separately via `EMBEDDING_PROVIDER`. The default
   `hashing` provider is offline, deterministic, and zero-dependency. To swap in
   learned semantic embeddings behind the same interface, set
   `EMBEDDING_PROVIDER=sentence-transformers` (optionally overriding
   `EMBEDDING_MODEL`, default `sentence-transformers/all-MiniLM-L6-v2`) and
   `pip install sentence-transformers`.

4. Ingest documents into the linear-scan vector store (brute-force cosine over a SQLite-backed chunk store).

   ```bash
   python - <<'PY'
   from pathlib import Path
   from backend.app import RAGService

   service = RAGService()
   service.ingest_directory(Path("sample_docs"))
   print(service.list_documents())
   PY
   ```

5. Start the application.

   ```bash
   make dev
   ```

   Open `http://127.0.0.1:8000` and use the upload, sample-ingest, chat, sources,
   and evaluation controls.

6. Run the sample evaluation set.

   ```bash
   make eval
   ```

   The included corpus has 13 synthetic internship documents and 41 labeled cases,
   including ambiguous, conflicting, and unanswerable questions. Compare retrieval
   strategies with:

   ```bash
   make compare
   ```

   The latest recorded comparison is written to `evals/comparison_results.json`.

7. Run tests.

   ```bash
   make test
   ```

   With the development server running, execute the browser workflow in a second terminal:

   ```bash
   make test-ui
   ```

   Browser screenshots are written to `artifacts/browser/` for visual inspection.

## Architecture

The application follows a modular RAG architecture:

- **Document ingestion**: Loads source files, normalizes text, splits documents into chunks, attaches metadata, and prepares embeddings.
- **Embedding and indexing**: Converts chunks into deterministic local vectors and stores them in SQLite.
- **Retriever**: Accepts a user query, embeds it, combines vector similarity with BM25-style keyword scoring, and returns top-k chunks.
- **Prompt builder**: Combines the user question, retrieved context, citation metadata, and system instructions into a grounded LLM prompt.
- **LLM generation layer**: Uses an offline extractive generator by default, with optional OpenAI and Ollama generation providers.
- **Application API/UI**: Provides a user-facing interface for asking questions, viewing cited sources, and inspecting answer confidence.
- **Evaluation harness**: Runs fixed questions against the RAG pipeline and records retrieval and generation metrics.

See [docs/architecture.md](docs/architecture.md) for a more detailed component view.

## Project Timeline

| Phase | Focus | Deliverables |
| --- | --- | --- |
| Week 1 | Requirements and setup | Scope, repository structure, environment setup, initial docs |
| Week 2 | Ingestion pipeline | Loaders, chunking strategy, metadata schema |
| Week 3 | Retrieval baseline | Embeddings, linear-scan vector store (brute-force cosine), top-k retrieval, retrieval diagnostics |
| Week 4 | LLM answer generation | Prompt templates, grounded responses, citation formatting |
| Week 5 | Evaluation | Sample dataset, metrics, manual review rubric, error analysis |
| Week 6 | Application polish | UI/API integration, configuration, limitations, final report |

## Features

- RAG-based question answering over a controlled document collection.
- Pluggable LLM and embedding provider design.
- Configurable chunk size, overlap, top-k retrieval, and temperature.
- Source-aware responses with citations or metadata references.
- Evaluation dataset for repeatable quality checks.
- FastAPI backend with static web UI.
- SQLite-backed local index for a zero-service demo.
- Documentation for architecture, metrics, risks, and project reporting.

## Implemented Module Map

- `backend/app/text_processing.py`: text/PDF extraction, normalization, and chunking.
- `backend/app/embeddings.py`: deterministic offline embedding baseline.
- `backend/app/store.py`: SQLite document and chunk storage.
- `backend/app/retrieval.py`: hybrid vector and keyword retrieval.
- `backend/app/llm.py`: offline, OpenAI, and Ollama generation adapters.
- `backend/app/rag.py`: ingestion and answer orchestration service.
- `backend/app/api.py`: FastAPI endpoints and static frontend hosting.
- `backend/app/evaluation.py`: JSONL evaluation runner.
- `backend/app/comparison.py`: vector-only, keyword-only, and hybrid retrieval comparison.
- `frontend/`: document upload, chat, source panel, and evaluation UI.
- `scripts/browser_smoke.py`: Playwright workflow covering upload, retrieval, evaluation, and responsive layout.
- `tests/`: core unit tests for chunking, ingestion, retrieval, and evaluation.
- `deliverables/`: final internship report and presentation.

## Evaluation Metrics

The project should evaluate both retrieval and generation:

- **Retrieval precision@k**: Whether retrieved chunks contain the evidence needed to answer the question.
- **Context recall**: Whether all required facts are present in the retrieved context.
- **Answer correctness**: Whether the generated answer matches the expected facts.
- **Groundedness**: Whether the answer is supported by retrieved source text.
- **Citation accuracy**: Whether cited documents actually support the answer.
- **Faithfulness failure rate**: Frequency of hallucinated, contradicted, or unsupported claims.
- **Latency**: End-to-end response time for ingestion, retrieval, and answer generation.
- **Cost per query**: Approximate API cost from embeddings and LLM calls.

See [docs/evaluation.md](docs/evaluation.md) for the suggested rubric and JSONL format.

The current local benchmark records 0.8241 average required-fact recall, 1.0000
citation-or-refusal coverage, 0.8000 unanswerable-refusal accuracy, and a 0.9126
combined score for hybrid retrieval. These values are development results on the
included synthetic corpus, not production claims.

## Limitations

- RAG quality depends heavily on document quality, chunking, and retrieval settings.
- LLM responses may still contain unsupported claims if prompts and validation are weak.
- Small evaluation datasets are useful for development but not enough for production confidence.
- API-based LLMs introduce external latency, cost, rate-limit, and availability constraints.
- Sensitive documents require careful access control, logging rules, and secret handling.
- The demo API exposes unauthenticated destructive endpoints (`DELETE /api/documents` and the ingest reset that replaces the corpus); these are intended for local single-user use only and must not be exposed on an untrusted network.
- The baseline project is designed for learning and demonstration, not regulated or safety-critical deployment.
