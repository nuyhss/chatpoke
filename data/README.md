# Data Layout

This project uses a document-first storage layout under `data/`.

## Directories

- `data/library/`
  Permanent shared documents for the knowledge base. Startup ingest and the file watcher target this folder.

- `data/uploads/`
  Runtime chat uploads from `/ui`, `/upload`, and `/chat-with-file`. These are intentionally not committed to git.

- `data/temp/`
  Temporary processing files. This is also intentionally not committed to git.

- `data/pdf/`
  Optional place for sample or legacy PDFs used during development and experiments.

## Git Policy

- The folder structure is tracked so teammates can see the intended layout after cloning.
- Runtime upload files, temp files, and the local document registry are ignored.
- If you want a document to be part of the shared benchmark/library corpus, place it in `data/library/`.
