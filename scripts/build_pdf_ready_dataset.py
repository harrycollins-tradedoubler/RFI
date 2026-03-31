#!/usr/bin/env python3
"""
Build a Ready-only RAG dataset CSV from a PDF.

Output schema matches the existing upload CSV used by Pinecone/Supabase scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    merged = " ".join(lines)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        if end < n:
            floor = start + int(chunk_size * 0.65)
            sent_break = text.rfind(". ", floor, end)
            if sent_break > start:
                end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_rows(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
    id_prefix: str,
    skip_pages: set[int] | None = None,
) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    rows: list[dict] = []
    global_idx = 1
    skip_pages = skip_pages or set()

    for page_num, page in enumerate(reader.pages, start=1):
        if page_num in skip_pages:
            continue
        raw = page.extract_text() or ""
        clean = normalize_text(raw)
        if not clean:
            continue
        page_chunks = chunk_text(clean, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(page_chunks, start=1):
            qid = f"{id_prefix}-{global_idx:04d}"
            metadata = {
                "source_file": pdf_path.name,
                "doc_title": pdf_path.stem,
                "page": page_num,
                "chunk_on_page": i,
                "chunks_on_page": len(page_chunks),
            }
            rows.append(
                {
                    "question_id": qid,
                    "content": chunk,
                    "canonical_question": "",
                    "starter_answer": "",
                    "category": "Technical and Organizational Measures",
                    "subcategory": f"Page {page_num}",
                    "response_type": "document_chunk",
                    "item_type": "chunk",
                    "language": "en",
                    "evidence_reference": f"{pdf_path.name} - Page {page_num}",
                    "source_files": pdf_path.name,
                    "source_sections": f"Page {page_num}",
                    "readiness_status": "Ready",
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                }
            )
            global_idx += 1

    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "question_id",
        "content",
        "canonical_question",
        "starter_answer",
        "category",
        "subcategory",
        "response_type",
        "item_type",
        "language",
        "evidence_reference",
        "source_files",
        "source_sections",
        "readiness_status",
        "metadata_json",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Ready dataset CSV from a PDF.")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument(
        "--output",
        default="output/dataset/tradedoubler_tom_june_2024_ready_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument("--chunk-size", type=int, default=1100, help="Chunk size chars")
    parser.add_argument("--overlap", type=int, default=180, help="Chunk overlap chars")
    parser.add_argument("--id-prefix", default="TD-TOM24", help="Record ID prefix")
    parser.add_argument(
        "--skip-pages",
        default="1",
        help="Comma-separated 1-based page numbers to skip (default: 1 to skip TOC page). Use empty string to include all.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    skip_pages: set[int] = set()
    if args.skip_pages.strip():
        for token in args.skip_pages.split(","):
            token = token.strip()
            if token:
                skip_pages.add(int(token))

    rows = build_rows(
        pdf_path=pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        id_prefix=args.id_prefix,
        skip_pages=skip_pages,
    )
    if not rows:
        raise SystemExit("No text chunks extracted from PDF.")

    output_csv = Path(args.output).resolve()
    write_csv(rows, output_csv)

    print(f"Source PDF: {pdf_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
