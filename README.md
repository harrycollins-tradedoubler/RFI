# Gemini Embedding 2 RAG Agent (Vertex AI)

This repo now contains a minimal, local RAG pipeline using:
- `gemini-embedding-2-preview` for document/query embeddings
- local NumPy vector index for retrieval
- Gemini generation model for final answers

## 1) Prerequisites

- Python 3.10+
- A Google Cloud project with Vertex AI enabled
- ADC auth configured (one of):
  - `gcloud auth application-default login`
  - service account via `GOOGLE_APPLICATION_CREDENTIALS`

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables (`.env.example`):

```bash
set GOOGLE_CLOUD_PROJECT=your-project-id
set GOOGLE_CLOUD_LOCATION=us-central1
```

## 2) Build the index

Put your docs in a folder (for example `docs/`) and run:

```bash
python rag_agent.py ingest --input docs --index-dir index
```

Supported file types:
- `.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.json`, `.csv`, `.pdf`

## 3) Ask one question

```bash
python rag_agent.py query --index-dir index --question "What is Gemini Embedding 2 used for?"
```

## 4) Start interactive chat

```bash
python rag_agent.py chat --index-dir index
```

Type `exit` to quit.

## 5) Upload Ready rows to Supabase (pgvector)

Use the generated CSV:
- `output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv`

Set env vars:

```bash
set SUPABASE_DB_URL=postgresql://postgres:<PASSWORD>@db.<PROJECT-REF>.supabase.co:5432/postgres?sslmode=require
set GOOGLE_CLOUD_PROJECT=your-project-id
set GOOGLE_CLOUD_LOCATION=us-central1
```

Run uploader:

```bash
python scripts/upload_ready_to_supabase.py --csv output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv --dimension 1536
```

What this does:
- enables pgvector extension
- creates `public.rag_docs` table
- embeds each Ready row with `gemini-embedding-2-preview`
- upserts rows with vectors
- creates `public.match_rag_docs(...)` SQL function for similarity search

## 6) Upload Ready rows to Pinecone (simpler setup)

This path avoids Google Cloud setup by using Pinecone integrated embeddings.

Set API key:

```bash
set PINECONE_API_KEY=your-pinecone-api-key
```

Run uploader:

```bash
python scripts/upload_ready_to_pinecone.py --csv output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv --index-name rfi-ready-rag --namespace __default__
```

What this does:
- creates Pinecone integrated-embedding index if missing
- uploads only `Ready` rows from CSV
- stores searchable text in `chunk_text` and metadata fields
- runs a small verification search

To force a clean reload (avoid duplicate counts), add:

```bash
python scripts/upload_ready_to_pinecone.py --csv output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv --index-name rfi-ready-rag --namespace __default__ --reset-namespace
```

## 7) Upload to Pinecone with OpenAI vectors (for n8n embedding node compatibility)

Use this when your retriever query vectors come from OpenAI embeddings.

Set keys:

```bash
set PINECONE_API_KEY=your-pinecone-api-key
set OPENAI_API_KEY=your-openai-api-key
```

Create/load index and upload rows (OpenAI `text-embedding-3-large`, 1024 dims):

```bash
python scripts/upload_ready_to_pinecone_openai.py --csv output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv --index-name rfi-ready-rag-openai-1024 --namespace __default__ --dimension 1024 --reset-namespace
```

This creates a standard Pinecone dense index (cosine) and upserts vectors directly, so n8n OpenAI embeddings can query it without model mismatch.

Freshness logic (default):
- legacy and unknown-year approved rows are kept during upload
- each row gets metadata fields for `source_year`, `source_kind`, `freshness_tier`, and `retrieval_priority`
- retrieval can still prefer newer evidence without removing older approved attachments from scope

Useful overrides:

```bash
# keep all legacy sources explicitly
python scripts/upload_ready_to_pinecone_openai.py --csv output/dataset/unified_ready_dataset.csv --index-name rfi-ready-rag-openai-1024 --namespace __default__ --dimension 1024 --drop-stale-sources none --drop-unknown-year-sources none --min-source-year 2024

# optional stricter mode: only drop stale attachments (keep older QB rows)
python scripts/upload_ready_to_pinecone_openai.py --csv output/dataset/unified_ready_dataset.csv --index-name rfi-ready-rag-openai-1024 --namespace __default__ --dimension 1024 --drop-stale-sources attachments --drop-unknown-year-sources attachments --min-source-year 2024
```

## 8) Build one dataset from mixed attachments (PDF, DOCX, XLSX)

```bash
python scripts/build_attached_ready_dataset.py \
  --input tmp/bulk_ingest \
  --input "C:/Users/harcol/Downloads/GDPR FAQ and GDPR Info Guide_2020-07-09.pdf" \
  --input "C:/Users/harcol/Downloads/Requests from data subjects_manual_PP 2021-03-29.pdf" \
  --output output/dataset/attached_sources_ready_dataset.csv
```

Output:
- `output/dataset/attached_sources_ready_dataset.csv`

## Useful flags

- `--dimension 1536` (default; max is 3072)
- `--top-k 5`
- `--chunk-size 1200`
- `--overlap 200`
- `--generation-model gemini-2.5-flash`

## Notes

- The index is stored in:
  - `index/vectors.npy`
  - `index/chunks.jsonl`
  - `index/index_meta.json`
- Re-run `ingest` whenever source docs change.
- Model/region availability can change. Check:
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/embedding-2
  - https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/
