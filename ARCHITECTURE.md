# Tilon AI Chatbot Architecture

## Overview

Tilon AI Chatbot is a document-first RAG backend for English/Korean PDFs and images.

This architecture should be understood as two connected systems:

1. a document RAG inference pipeline
2. a QLoRA fine-tuning pipeline for the final answer model

The current repo is much stronger on the RAG side than the QLoRA side. The next project milestone is not another large parser rewrite; it is to stabilize RAG enough that evaluation and fine-tuning can begin systematically.

The product supports two document sources:

- `data/library/`
  Persistent team documents that form the long-term knowledge base

- `data/uploads/`
  Chat-uploaded files that are ingested for the current user flow and scoped to chat

The system is optimized for:
- summarization
- question answering
- exact information lookup
- text extraction from images/scanned pages
- section/structure-aware retrieval

## High-Level Flow

### 1. Library documents
1. Team adds files to `data/library/`
2. Startup ingest or file watcher detects them
3. Parser extracts text from PDF/image
4. Chunker creates semantic chunks
5. Enricher prepends contextual headers
6. Embeddings are stored in ChromaDB
7. Keyword index is rebuilt/updated for hybrid retrieval

### 2. Chat uploads
1. User uploads a file in `/ui` or `/chat-with-file`
2. File is saved to `data/uploads/`
3. It is parsed, chunked, enriched, and stored
4. Chat keeps an `active_source` so follow-up questions stay scoped to that uploaded file
5. Temporary chat uploads are not treated as permanent library docs by the watcher

## Bigger-Picture Architecture

The full product flow is:

1. document input
2. extraction / parsing
3. semantic chunking
4. contextual enrichment
5. embeddings
6. vector + keyword storage
7. hybrid retrieval
8. reranking
9. confidence gating
10. prompt construction
11. answer generation
12. future QLoRA-enhanced answer generation

Important distinction:
- RAG is responsible for retrieving correct evidence
- QLoRA is responsible for improving how the model uses that evidence

QLoRA is not a substitute for:
- extraction quality
- chunking quality
- retrieval precision
- confidence gating

## Storage Layout

```text
data/
├── library/   # persistent corpus
├── uploads/   # chat-uploaded files
└── temp/      # optional intermediate files
```

This split avoids mixing permanent documents with one-off chat uploads and prevents unexpected re-ingestion after restart.

## Core Components

### API Layer
- [main.py](/home/tilon/chatbot-karbi/main.py)
- [app/api/routes.py](/home/tilon/chatbot-karbi/app/api/routes.py)
- [app/api/upload_ui.py](/home/tilon/chatbot-karbi/app/api/upload_ui.py)
- [app/api/openai_compat.py](/home/tilon/chatbot-karbi/app/api/openai_compat.py)

Responsibilities:
- HTTP endpoints
- built-in chat/upload UI
- OpenAI-compatible endpoints
- upload handling and folder routing

### Parsing / Ingestion Layer
- [app/pipeline/parser.py](/home/tilon/chatbot-karbi/app/pipeline/parser.py)
- [app/pipeline/chunker.py](/home/tilon/chatbot-karbi/app/pipeline/chunker.py)
- [app/pipeline/enricher.py](/home/tilon/chatbot-karbi/app/pipeline/enricher.py)
- [app/pipeline/ingest.py](/home/tilon/chatbot-karbi/app/pipeline/ingest.py)

Responsibilities:
- PDF/image text extraction
- semantic chunking
- contextual enrichment
- ingestion orchestration

### Retrieval Layer
- [app/core/vectorstore.py](/home/tilon/chatbot-karbi/app/core/vectorstore.py)
- [app/retrieval/retriever.py](/home/tilon/chatbot-karbi/app/retrieval/retriever.py)
- [app/retrieval/keyword_index.py](/home/tilon/chatbot-karbi/app/retrieval/keyword_index.py)
- [app/retrieval/reranker.py](/home/tilon/chatbot-karbi/app/retrieval/reranker.py)

Responsibilities:
- vector search
- BM25-style keyword retrieval
- reciprocal rank fusion
- optional reranking
- confidence-aware scoped retrieval

### Chat Layer
- [app/chat/handlers.py](/home/tilon/chatbot-karbi/app/chat/handlers.py)
- [app/chat/prompts.py](/home/tilon/chatbot-karbi/app/chat/prompts.py)

Responsibilities:
- build final prompt
- unify chat behavior
- direct extraction responses for image/text requests
- scoped document fallback when retrieval confidence is low

## Extraction Strategy

The parser uses a layered approach:

1. `marker_single`
   Best for structured digital PDFs

2. `PyMuPDF`
   Fast text-layer extraction for born-digital pages

3. `qwen2.5vl`
   Vision fallback for scanned or image-heavy pages

4. `tesseract`
   OCR fallback

Recent improvements:
- per-page routing
- page type classification: `digital`, `hybrid`, `scanned`
- quality gates for low text yield / garbled text
- richer page metadata such as heading hints and layout counts

## Retrieval Strategy

The current retrieval path is hybrid:

1. vector retrieval from ChromaDB
2. keyword retrieval from in-memory index
3. reciprocal rank fusion
4. optional reranking
5. confidence gating for scoped document chat

For scoped uploaded-file chat:
- specific question -> top-k scoped retrieval
- full-document tasks like summary/analysis -> full-document scoped context
- direct OCR/transcription requests -> return extracted text directly

## Maturity By Layer

### Strong / mostly ready
- upload and storage split (`library` vs `uploads`)
- parser routing and fallback extraction
- semantic chunking
- contextual enrichment
- document-scoped chat
- hybrid retrieval foundation

### Working but still being tuned
- reranker runtime strategy
- confidence thresholds
- screenshot/image upload behavior in broader chat
- document registry behavior

### Not started or not complete
- benchmark-driven evaluation
- formal QLoRA dataset pipeline
- multi-document comparison
- richer block-level artifacts for tables/citations/versioning

## UI Behavior

The built-in `/ui` is the primary frontend.

Behavior:
- uploaded file is stored in `data/uploads/`
- the chat remembers `active_source`
- follow-up messages stay scoped to the uploaded file
- user can clear scope and return to global retrieval

## Current Limits

Still not fully implemented:
- full document registry lifecycle management
- document versioning
- multi-document comparison workflow
- richer block-level artifact storage beyond page metadata
- polished production UI design
- formal evaluation benchmark
- QLoRA training/evaluation pipeline

## Why The Project Can Feel “Stuck”

The repo has moved beyond the early prototype stage, but progress can still feel slow because most work is happening one example at a time:
- one screenshot
- one OCR case
- one threshold
- one reranker issue

That is useful engineering work, but it does not yet create a measurable product-quality loop.

What is missing is:
- a benchmark set of real Tilon documents
- a representative query set
- baseline metrics
- a QLoRA training/evaluation workflow

Without these, it is hard to tell whether the project is improving globally or only locally.

## Recommended Next Steps

### Phase 1: Stabilize RAG Core
1. Finish tuning retrieval and confidence behavior on real Tilon docs
2. Finalize document registry behavior and upload lifecycle
3. Keep prompt/context format stable enough for training reuse

### Phase 2: Build Evaluation Benchmark
1. Collect representative Tilon manuals, guides, and internal docs
2. Build question sets for:
   - exact lookup
   - summary
   - OCR/image text
   - section understanding
   - negative “not found” cases
   - Korean/English mixed queries
3. Record baseline system behavior

### Phase 3: Start QLoRA Properly
1. Build training data using the same live prompt/context format
2. Fine-tune the answer model, not the retriever
3. Compare:
   - base model
   - RAG + base
   - RAG + QLoRA

## QLoRA Start Conditions

QLoRA should begin after:
- retrieval quality is stable enough on real Tilon docs
- the prompt/context format is frozen
- benchmark questions exist
- baseline metrics are recorded

That is the point where fine-tuning becomes measurable instead of speculative.
