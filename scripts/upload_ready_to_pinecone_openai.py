#!/usr/bin/env python3
"""
Upload Ready rows to a standard Pinecone index using OpenAI embeddings.

This is the correct path when your retriever also uses OpenAI embeddings in n8n.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from pinecone import Pinecone, ServerlessSpec


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def require_openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openai package is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc
    return OpenAI(api_key=api_key)


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


def load_ready_rows(
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


def parse_metadata(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"raw": obj}
    except Exception:
        return {"raw_metadata": raw}


def compact_metadata(data: dict) -> dict:
    return {k: v for k, v in data.items() if v is not None}


def batched(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ensure_index(
    pc: Pinecone,
    index_name: str,
    cloud: str,
    region: str,
    dimension: int,
    metric: str,
) -> None:
    if pc.has_index(index_name):
        desc = pc.describe_index(index_name)
        current_dim = int(desc.dimension)
        current_metric = str(desc.metric)
        if current_dim != dimension:
            raise RuntimeError(
                f"Existing index `{index_name}` has dimension {current_dim}, expected {dimension}."
            )
        if current_metric != metric:
            raise RuntimeError(
                f"Existing index `{index_name}` has metric {current_metric}, expected {metric}."
            )
    else:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
            deletion_protection="disabled",
            vector_type="dense",
        )

    for _ in range(60):
        desc = pc.describe_index(index_name)
        status = getattr(desc, "status", None)
        ready = False
        if isinstance(status, dict):
            ready = bool(status.get("ready"))
        else:
            ready = bool(getattr(status, "ready", False))
        if ready:
            return
        time.sleep(2)
    raise RuntimeError(f"Index `{index_name}` did not become ready in time.")


def embed_texts(openai_client, model: str, dimension: int, texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=model,
        input=texts,
        dimensions=dimension,
    )
    return [item.embedding for item in response.data]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload Ready rows CSV to Pinecone (OpenAI embeddings)."
    )
    parser.add_argument(
        "--csv",
        default="output/spreadsheet/RFI_RFP_Qs_ready_supabase_import.csv",
        help="CSV path",
    )
    parser.add_argument(
        "--index-name",
        default="rfi-ready-rag-openai-1024",
        help="Pinecone index name",
    )
    parser.add_argument("--namespace", default="__default__", help="Namespace")
    parser.add_argument("--cloud", default="aws", help="Pinecone cloud")
    parser.add_argument("--region", default="us-east-1", help="Pinecone region")
    parser.add_argument("--metric", default="cosine", help="Similarity metric")
    parser.add_argument("--dimension", type=int, default=1024, help="Embedding dimension")
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/upsert batch size")
    parser.add_argument(
        "--min-source-year",
        type=int,
        default=2024,
        help="Minimum source year to treat as current. Older rows may be dropped per --drop-stale-sources.",
    )
    parser.add_argument(
        "--drop-stale-sources",
        choices=["none", "attachments", "all"],
        default="none",
        help="Drop stale rows older than --min-source-year for attachment corpus only, all rows, or none. Default keeps legacy approved evidence and relies on metadata for prioritization.",
    )
    parser.add_argument(
        "--drop-unknown-year-sources",
        choices=["none", "attachments", "all"],
        default="none",
        help="Drop rows that do not have a detectable source year for attachment corpus only, all rows, or none. Default keeps unknown-year approved evidence.",
    )
    parser.add_argument(
        "--reset-namespace",
        action="store_true",
        help="Delete all vectors in namespace before upload.",
    )
    args = parser.parse_args()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not pinecone_api_key:
        print("Missing PINECONE_API_KEY env var.", file=sys.stderr)
        return 2
    if not openai_api_key:
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2
    if args.dimension < 1 or args.dimension > 3072:
        print("--dimension must be between 1 and 3072.", file=sys.stderr)
        return 2

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    rows = load_ready_rows(
        csv_path=csv_path,
        min_source_year=args.min_source_year,
        drop_stale_sources=args.drop_stale_sources,
        drop_unknown_year_sources=args.drop_unknown_year_sources,
    )
    if not rows:
        print("No Ready rows found in CSV.", file=sys.stderr)
        return 1

    openai_client = require_openai_client(openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)

    ensure_index(
        pc=pc,
        index_name=args.index_name,
        cloud=args.cloud,
        region=args.region,
        dimension=args.dimension,
        metric=args.metric,
    )

    index = pc.Index(args.index_name)
    if args.reset_namespace:
        print(f"Resetting namespace `{args.namespace}`...")
        try:
            index.delete(delete_all=True, namespace=args.namespace)
        except Exception as exc:
            if "Namespace not found" in str(exc):
                print("Namespace did not exist yet; continuing.")
            else:
                raise

    total = len(rows)
    done = 0

    for batch in batched(rows, args.batch_size):
        texts = [(r.get("content") or "").strip() for r in batch]
        vectors = embed_texts(openai_client, args.embed_model, args.dimension, texts)

        upserts = []
        for row, vector in zip(batch, vectors):
            qid = (row.get("question_id") or "").strip()
            parsed_meta = parse_metadata(row.get("metadata_json"))
            source_year = row.get("_source_year")
            source_kind = row.get("_source_kind") or ""
            curated_answer = bool(row.get("_curated_answer"))
            retrieval_priority = int(row.get("_retrieval_priority") or 0)
            freshness_tier = row.get("_freshness_tier") or "unknown"
            parsed_meta = compact_metadata(
                {
                    **parsed_meta,
                    "source_year": source_year,
                    "source_kind": source_kind,
                    "curated_answer": curated_answer,
                    "retrieval_priority": retrieval_priority,
                    "freshness_tier": freshness_tier,
                }
            )
            metadata = compact_metadata({
                "question_id": qid,
                "canonical_question": row.get("canonical_question") or "",
                "starter_answer": row.get("starter_answer") or "",
                "category": row.get("category") or "",
                "subcategory": row.get("subcategory") or "",
                "response_type": row.get("response_type") or "",
                "item_type": row.get("item_type") or "",
                "language": row.get("language") or "",
                "readiness_status": row.get("readiness_status") or "",
                "source_files": row.get("source_files") or "",
                "source_sections": row.get("source_sections") or "",
                "evidence_reference": row.get("evidence_reference") or "",
                "chunk_text": row.get("content") or "",
                "source_year": source_year,
                "source_kind": source_kind,
                "curated_answer": curated_answer,
                "retrieval_priority": retrieval_priority,
                "freshness_tier": freshness_tier,
                "metadata_json": json.dumps(parsed_meta, ensure_ascii=False),
            })
            upserts.append(
                {
                    "id": qid,
                    "values": vector,
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=upserts, namespace=args.namespace, show_progress=False)
        done += len(batch)
        print(f"Uploaded {done}/{total}")

    try:
        query_vector = embed_texts(
            openai_client,
            args.embed_model,
            args.dimension,
            ["Are the Administrators connecting with MFA?"],
        )[0]
        result = index.query(
            namespace=args.namespace,
            vector=query_vector,
            top_k=3,
            include_metadata=True,
        )
        matches = getattr(result, "matches", []) or []
        print(f"Verification query matches: {len(matches)}")
        for match in matches:
            meta = getattr(match, "metadata", {}) or {}
            print(
                f"- {meta.get('question_id', '')} | {meta.get('canonical_question', '')} | score={getattr(match, 'score', 0):.4f}"
            )
    except Exception:
        print("Upload succeeded (verification query skipped).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
