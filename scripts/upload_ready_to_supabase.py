#!/usr/bin/env python3
"""
Upload Ready rows from CSV into Supabase pgvector table with Gemini embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import psycopg
from google import genai


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embedding_literal(values: list[float]) -> str:
    # pgvector accepts a string literal like: [0.1,0.2,...]
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def ensure_schema(conn: psycopg.Connection, dimension: int) -> None:
    if dimension < 1 or dimension > 3072:
        raise ValueError("dimension must be between 1 and 3072")

    ddl = f"""
    create extension if not exists vector with schema extensions;

    create table if not exists public.rag_docs (
      id bigserial primary key,
      question_id text not null unique,
      content text not null,
      canonical_question text,
      starter_answer text,
      category text,
      subcategory text,
      response_type text,
      item_type text,
      language text,
      evidence_reference text,
      source_files text,
      source_sections text,
      readiness_status text,
      metadata_json jsonb default '{{}}'::jsonb,
      embedding extensions.vector({dimension}),
      created_at timestamptz not null default now(),
      updated_at timestamptz not null default now()
    );

    create index if not exists rag_docs_metadata_json_idx
      on public.rag_docs using gin (metadata_json);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
        # HNSW index with vector currently supports up to 2000 dims in pgvector.
        if dimension <= 2000:
            cur.execute(
                """
                create index if not exists rag_docs_embedding_hnsw_idx
                  on public.rag_docs using hnsw (embedding vector_cosine_ops);
                """
            )
        cur.execute(
            f"""
            create or replace function public.match_rag_docs(
              query_embedding extensions.vector({dimension}),
              match_count int default 5,
              filter jsonb default '{{}}'::jsonb
            )
            returns table (
              id bigint,
              question_id text,
              content text,
              metadata_json jsonb,
              similarity float
            )
            language sql
            stable
            as $$
              select
                d.id,
                d.question_id,
                d.content,
                d.metadata_json,
                1 - (d.embedding <=> query_embedding) as similarity
              from public.rag_docs d
              where d.metadata_json @> filter
              order by d.embedding <=> query_embedding
              limit match_count;
            $$;
            """
        )
    conn.commit()


def infer_source_year(row: dict) -> int | None:
    candidates = [
        row.get("source_files") or "",
        row.get("evidence_reference") or "",
        row.get("source_sections") or "",
        row.get("metadata_json") or "",
    ]
    years: list[int] = []
    for text in candidates:
        for match in YEAR_RE.finditer(str(text)):
            year = int(match.group(1))
            if 1990 <= year <= 2100:
                years.append(year)
    return max(years) if years else None


def classify_source_kind(qid: str) -> str:
    q = qid.strip().upper()
    if q.startswith("QB-"):
        return "question_bank"
    if q.startswith("ATTACH-"):
        return "attachment_corpus"
    return "other"


def has_curated_answer(row: dict) -> bool:
    starter = (row.get("starter_answer") or "").strip()
    item_type = (row.get("item_type") or "").strip().lower()
    return bool(starter) and item_type in {"question", "prompt"}


def compute_retrieval_priority(row: dict, source_year: int | None) -> int:
    qid = (row.get("question_id") or "").strip()
    item_type = (row.get("item_type") or "").strip().lower()
    source_kind = classify_source_kind(qid)

    score = 0
    if source_kind == "question_bank":
        score += 40
    elif source_kind == "attachment_corpus":
        score -= 10
    if has_curated_answer(row):
        score += 35
    if item_type in {"question", "prompt"}:
        score += 10
    elif item_type == "chunk":
        score -= 5
    if source_year is not None:
        score += max(-20, min(30, source_year - 2010))
        if source_year < 2023:
            score -= 20
    return score


def should_drop_as_stale(
    source_kind: str,
    source_year: int | None,
    min_source_year: int | None,
    drop_stale_sources: str,
) -> bool:
    if min_source_year is None or source_year is None:
        return False
    if source_year >= min_source_year:
        return False
    if drop_stale_sources == "all":
        return True
    if drop_stale_sources == "attachments" and source_kind == "attachment_corpus":
        return True
    return False


def should_drop_unknown_year(
    source_kind: str,
    source_year: int | None,
    drop_unknown_year_sources: str,
) -> bool:
    if source_year is not None:
        return False
    if drop_unknown_year_sources == "all":
        return True
    if drop_unknown_year_sources == "attachments" and source_kind == "attachment_corpus":
        return True
    return False


def better_row(new_row: dict, old_row: dict) -> bool:
    new_rank = (
        int(new_row.get("_retrieval_priority", 0)),
        int(new_row.get("_source_year", 0) or 0),
        len((new_row.get("starter_answer") or "").strip()),
        len((new_row.get("evidence_reference") or "").strip()),
    )
    old_rank = (
        int(old_row.get("_retrieval_priority", 0)),
        int(old_row.get("_source_year", 0) or 0),
        len((old_row.get("starter_answer") or "").strip()),
        len((old_row.get("evidence_reference") or "").strip()),
    )
    return new_rank > old_rank


def enrich_row(row: dict) -> dict:
    source_year = infer_source_year(row)
    source_kind = classify_source_kind((row.get("question_id") or "").strip())
    curated_answer = has_curated_answer(row)
    retrieval_priority = compute_retrieval_priority(row, source_year)
    out = dict(row)
    out["_source_year"] = source_year
    out["_source_kind"] = source_kind
    out["_curated_answer"] = curated_answer
    out["_retrieval_priority"] = retrieval_priority
    out["_freshness_tier"] = (
        "current" if source_year and source_year >= 2024 else ("legacy" if source_year else "unknown")
    )
    return out


def read_ready_rows(
    csv_path: Path,
    min_source_year: int | None,
    drop_stale_sources: str,
    drop_unknown_year_sources: str,
) -> list[dict]:
    deduped: dict[str, dict] = {}
    skipped_missing_id = 0
    skipped_stale = 0
    skipped_unknown_year = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("readiness_status") or "").strip().lower()
            if status != "ready":
                continue
            qid = (row.get("question_id") or "").strip()
            if not qid:
                skipped_missing_id += 1
                continue
            enriched = enrich_row(row)
            if should_drop_as_stale(
                source_kind=str(enriched.get("_source_kind") or ""),
                source_year=enriched.get("_source_year"),
                min_source_year=min_source_year,
                drop_stale_sources=drop_stale_sources,
            ):
                skipped_stale += 1
                continue
            if should_drop_unknown_year(
                source_kind=str(enriched.get("_source_kind") or ""),
                source_year=enriched.get("_source_year"),
                drop_unknown_year_sources=drop_unknown_year_sources,
            ):
                skipped_unknown_year += 1
                continue
            existing = deduped.get(qid)
            if not existing or better_row(enriched, existing):
                deduped[qid] = enriched
    if skipped_missing_id:
        print(f"Skipped {skipped_missing_id} row(s) with empty question_id.")
    if skipped_stale:
        print(
            f"Skipped {skipped_stale} stale row(s) "
            f"(min_source_year={min_source_year}, drop_stale_sources={drop_stale_sources})."
        )
    if skipped_unknown_year:
        print(
            f"Skipped {skipped_unknown_year} unknown-year row(s) "
            f"(drop_unknown_year_sources={drop_unknown_year_sources})."
        )
    return list(deduped.values())


def parse_metadata(value: str | None) -> dict:
    if not value:
        return {}
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else {"raw": obj}
    except Exception:
        return {"raw_metadata": value}


def upsert_batch(
    conn: psycopg.Connection,
    rows: list[dict],
    vectors: list[list[float]],
) -> None:
    sql = """
    insert into public.rag_docs (
      question_id, content, canonical_question, starter_answer,
      category, subcategory, response_type, item_type, language,
      evidence_reference, source_files, source_sections, readiness_status,
      metadata_json, embedding, updated_at
    ) values (
      %(question_id)s, %(content)s, %(canonical_question)s, %(starter_answer)s,
      %(category)s, %(subcategory)s, %(response_type)s, %(item_type)s, %(language)s,
      %(evidence_reference)s, %(source_files)s, %(source_sections)s, %(readiness_status)s,
      %(metadata_json)s::jsonb, cast(%(embedding)s as extensions.vector), now()
    )
    on conflict (question_id)
    do update set
      content = excluded.content,
      canonical_question = excluded.canonical_question,
      starter_answer = excluded.starter_answer,
      category = excluded.category,
      subcategory = excluded.subcategory,
      response_type = excluded.response_type,
      item_type = excluded.item_type,
      language = excluded.language,
      evidence_reference = excluded.evidence_reference,
      source_files = excluded.source_files,
      source_sections = excluded.source_sections,
      readiness_status = excluded.readiness_status,
      metadata_json = excluded.metadata_json,
      embedding = excluded.embedding,
      updated_at = now();
    """

    payloads = []
    for row, vector in zip(rows, vectors):
        source_year = row.get("_source_year")
        source_kind = row.get("_source_kind") or ""
        curated_answer = bool(row.get("_curated_answer"))
        retrieval_priority = int(row.get("_retrieval_priority") or 0)
        freshness_tier = row.get("_freshness_tier") or "unknown"
        metadata = parse_metadata(row.get("metadata_json"))
        metadata.update(
            {
                "source_year": source_year,
                "source_kind": source_kind,
                "curated_answer": curated_answer,
                "retrieval_priority": retrieval_priority,
                "freshness_tier": freshness_tier,
            }
        )
        payloads.append(
            {
                "question_id": row.get("question_id") or "",
                "content": row.get("content") or "",
                "canonical_question": row.get("canonical_question") or "",
                "starter_answer": row.get("starter_answer") or "",
                "category": row.get("category") or "",
                "subcategory": row.get("subcategory") or "",
                "response_type": row.get("response_type") or "",
                "item_type": row.get("item_type") or "",
                "language": row.get("language") or "",
                "evidence_reference": row.get("evidence_reference") or "",
                "source_files": row.get("source_files") or "",
                "source_sections": row.get("source_sections") or "",
                "readiness_status": row.get("readiness_status") or "",
                "metadata_json": json.dumps(metadata),
                "embedding": embedding_literal(vector),
            }
        )

    with conn.cursor() as cur:
        cur.executemany(sql, payloads)
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload Ready rows from CSV to Supabase with Gemini embeddings."
    )
    parser.add_argument(
        "--csv",
        default="output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv",
        help="Path to CSV file.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="Gemini output_dimensionality (1-3072).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--embedding-model",
        default="gemini-embedding-2-preview",
        help="Embedding model id.",
    )
    parser.add_argument(
        "--min-source-year",
        type=int,
        default=2024,
        help="Minimum source year to treat as current. Older rows may be dropped per --drop-stale-sources.",
    )
    parser.add_argument(
        "--drop-stale-sources",
        choices=["none", "attachments", "all"],
        default="all",
        help="Drop stale rows older than --min-source-year for attachment corpus only, all rows, or none.",
    )
    parser.add_argument(
        "--drop-unknown-year-sources",
        choices=["none", "attachments", "all"],
        default="attachments",
        help="Drop rows that do not have a detectable source year for attachment corpus only, all rows, or none.",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        help="Vertex location.",
    )
    args = parser.parse_args()

    db_url = os.environ.get("SUPABASE_DB_URL", "").strip()
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not db_url:
        print("Missing SUPABASE_DB_URL env var.", file=sys.stderr)
        return 2
    if not gcp_project:
        print("Missing GOOGLE_CLOUD_PROJECT env var.", file=sys.stderr)
        return 2

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    rows = read_ready_rows(
        csv_path=csv_path,
        min_source_year=args.min_source_year,
        drop_stale_sources=args.drop_stale_sources,
        drop_unknown_year_sources=args.drop_unknown_year_sources,
    )
    if not rows:
        print("No Ready rows found in CSV.", file=sys.stderr)
        return 1

    client = genai.Client(vertexai=True, project=gcp_project, location=args.location)
    embed_types = __import__("google.genai.types", fromlist=["EmbedContentConfig"])
    embed_cfg_cls = embed_types.EmbedContentConfig

    with psycopg.connect(db_url) as conn:
        # Avoid prepared statements so this works with Supavisor transaction poolers.
        conn.prepare_threshold = None
        ensure_schema(conn, args.dimension)

        total = len(rows)
        done = 0
        for batch in batched(rows, args.batch_size):
            texts = [(r.get("content") or "").strip() for r in batch]
            resp = client.models.embed_content(
                model=args.embedding_model,
                contents=texts,
                config=embed_cfg_cls(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=args.dimension,
                ),
            )
            vectors = [e.values for e in resp.embeddings]
            upsert_batch(conn, batch, vectors)
            done += len(batch)
            print(f"Uploaded {done}/{total}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
