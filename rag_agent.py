#!/usr/bin/env python3
"""
Minimal RAG agent using Gemini Embedding 2 on Vertex AI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log", ".json", ".csv"}


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return np


@dataclass
class ChunkRecord:
    chunk_id: int
    source: str
    chunk_index: int
    text: str


class GeminiRAGClient:
    def __init__(
        self,
        project: str,
        location: str,
        embedding_model: str,
        generation_model: str,
    ) -> None:
        from google import genai

        self._embed_config_cls = __import__(
            "google.genai.types", fromlist=["EmbedContentConfig"]
        ).EmbedContentConfig
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    def embed(
        self,
        texts: List[str],
        task_type: str,
        output_dimensionality: int,
        title: str | None = None,
    ) -> np.ndarray:
        np = require_numpy()
        config_kwargs = {
            "task_type": task_type,
            "output_dimensionality": output_dimensionality,
        }
        if title:
            config_kwargs["title"] = title

        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
            config=self._embed_config_cls(**config_kwargs),
        )

        rows = []
        for emb in response.embeddings:
            rows.append(np.array(emb.values, dtype=np.float32))

        if not rows:
            raise RuntimeError("Embedding API returned no vectors.")

        return np.vstack(rows)

    def answer(self, question: str, context_blocks: List[str]) -> str:
        prompt = (
            "You are a retrieval-augmented assistant.\n"
            "Use ONLY the provided context to answer.\n"
            "If context is insufficient, say what is missing.\n"
            "Always include source citations in square brackets like [source].\n\n"
            f"Question:\n{question}\n\n"
            "Context:\n"
            + "\n\n".join(context_blocks)
        )
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
        )
        if hasattr(response, "text") and response.text:
            return response.text
        return str(response)


def discover_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files: List[Path] = []
    for path in input_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS or path.suffix.lower() == ".pdf":
            files.append(path)
    return sorted(files)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError(
                "pypdf is not installed. Install dependencies from requirements.txt."
            )
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    clean = normalize_text(text)
    if not clean:
        return []

    chunks: List[str] = []
    start = 0
    length = len(clean)

    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            search_floor = start + int(chunk_size * 0.6)
            paragraph_break = clean.rfind("\n\n", search_floor, end)
            sentence_break = clean.rfind(". ", search_floor, end)
            best_break = max(paragraph_break, sentence_break)
            if best_break > start:
                end = best_break + 1

        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def batched(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def save_index(
    index_dir: Path,
    vectors: np.ndarray,
    records: List[ChunkRecord],
    metadata: dict | None = None,
) -> None:
    np = require_numpy()
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", vectors)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=True) + "\n")
    if metadata is not None:
        (index_dir / "index_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


def load_index(index_dir: Path) -> tuple[np.ndarray, List[ChunkRecord]]:
    np = require_numpy()
    vectors_path = index_dir / "vectors.npy"
    chunks_path = index_dir / "chunks.jsonl"
    if not vectors_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run `ingest` first."
        )

    vectors = np.load(vectors_path)
    records: List[ChunkRecord] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(ChunkRecord(**row))

    if len(vectors) != len(records):
        raise RuntimeError(
            "Index files are inconsistent: vector count != chunk count."
        )
    return vectors, records


def cosine_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    np = require_numpy()
    query = query.reshape(-1)
    doc_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise RuntimeError("Query embedding norm is zero.")

    scores = (matrix @ query) / (doc_norms * query_norm + 1e-12)
    k = min(k, len(scores))
    top_indices = np.argsort(-scores)[:k]
    top_scores = scores[top_indices]
    return top_indices, top_scores


def build_records(
    files: List[Path], chunk_size: int, overlap: int, root: Path
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    next_id = 0
    for file in files:
        text = read_document(file)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        rel = str(file.relative_to(root)) if file.is_relative_to(root) else str(file)
        for idx, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    chunk_id=next_id,
                    source=rel,
                    chunk_index=idx,
                    text=chunk,
                )
            )
            next_id += 1
    return records


def run_ingest(args: argparse.Namespace) -> None:
    np = require_numpy()
    input_path = Path(args.input).resolve()
    index_dir = Path(args.index_dir).resolve()
    root = input_path if input_path.is_dir() else input_path.parent

    files = discover_files(input_path)
    if not files:
        raise RuntimeError(f"No supported files found at: {input_path}")

    records = build_records(files, args.chunk_size, args.overlap, root=root)
    if not records:
        raise RuntimeError("No chunks generated from input files.")

    client = GeminiRAGClient(
        project=args.project,
        location=args.location,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
    )

    print(f"Embedding {len(records)} chunks from {len(files)} file(s)...")
    vectors_list: List[np.ndarray] = []
    for batch in batched([r.text for r in records], args.batch_size):
        vectors = client.embed(
            texts=batch,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=args.dimension,
        )
        vectors_list.append(vectors)

    matrix = np.vstack(vectors_list).astype(np.float32)
    save_index(
        index_dir=index_dir,
        vectors=matrix,
        records=records,
        metadata={
            "embedding_model": args.embedding_model,
            "dimension": int(matrix.shape[1]),
            "source_input": str(input_path),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "files": len(files),
            "chunks": len(records),
        },
    )
    print(f"Index saved to {index_dir}")
    print(f"Vector shape: {matrix.shape}")


def retrieve(
    client: GeminiRAGClient,
    question: str,
    index_dir: Path,
    top_k: int,
    dimension: int,
) -> tuple[List[ChunkRecord], np.ndarray]:
    vectors, records = load_index(index_dir)
    if vectors.shape[1] != dimension:
        raise RuntimeError(
            f"Dimension mismatch: index has {vectors.shape[1]} but --dimension={dimension}."
        )

    query_vec = client.embed(
        texts=[question],
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=dimension,
    )[0]

    indices, scores = cosine_top_k(vectors, query_vec, k=top_k)
    hits = [records[i] for i in indices]
    return hits, scores


def run_query(args: argparse.Namespace) -> None:
    client = GeminiRAGClient(
        project=args.project,
        location=args.location,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
    )

    hits, scores = retrieve(
        client=client,
        question=args.question,
        index_dir=Path(args.index_dir).resolve(),
        top_k=args.top_k,
        dimension=args.dimension,
    )

    context_blocks = []
    for rec, score in zip(hits, scores):
        context_blocks.append(
            f"[{rec.source}#chunk{rec.chunk_index} score={score:.4f}]\n{rec.text}"
        )

    answer = client.answer(args.question, context_blocks)
    print(answer)


def run_chat(args: argparse.Namespace) -> None:
    client = GeminiRAGClient(
        project=args.project,
        location=args.location,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
    )
    index_dir = Path(args.index_dir).resolve()

    print("RAG chat started. Type `exit` to quit.")
    while True:
        question = input("\nYou> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        hits, scores = retrieve(
            client=client,
            question=question,
            index_dir=index_dir,
            top_k=args.top_k,
            dimension=args.dimension,
        )
        context_blocks = []
        for rec, score in zip(hits, scores):
            context_blocks.append(
                f"[{rec.source}#chunk{rec.chunk_index} score={score:.4f}]\n{rec.text}"
            )

        answer = client.answer(question, context_blocks)
        print(f"\nAgent> {answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini Embedding 2 RAG agent (Vertex AI)")
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        help="Google Cloud project ID (or set GOOGLE_CLOUD_PROJECT).",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        help="Vertex AI region. Gemini Embedding 2 preview supports us-central1.",
    )
    parser.add_argument(
        "--embedding-model",
        default="gemini-embedding-2-preview",
        help="Embedding model ID.",
    )
    parser.add_argument(
        "--generation-model",
        default="gemini-2.5-flash",
        help="Generation model for final answers.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="Embedding output dimensionality (max 3072).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Chunk docs and build local vector index.")
    ingest.add_argument("--input", required=True, help="Input file or folder.")
    ingest.add_argument("--index-dir", default="index", help="Where to save index files.")
    ingest.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters.")
    ingest.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters.")
    ingest.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    ingest.set_defaults(func=run_ingest)

    query = sub.add_parser("query", help="One-shot RAG query.")
    query.add_argument("--index-dir", default="index", help="Index directory.")
    query.add_argument("--question", required=True, help="Question to answer.")
    query.add_argument("--top-k", type=int, default=5, help="Top matches to retrieve.")
    query.set_defaults(func=run_query)

    chat = sub.add_parser("chat", help="Interactive RAG chat.")
    chat.add_argument("--index-dir", default="index", help="Index directory.")
    chat.add_argument("--top-k", type=int, default=5, help="Top matches to retrieve.")
    chat.set_defaults(func=run_chat)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.project:
        print(
            "Missing project ID. Set --project or GOOGLE_CLOUD_PROJECT.",
            file=sys.stderr,
        )
        return 2
    if args.dimension < 1 or args.dimension > 3072:
        print("--dimension must be between 1 and 3072.", file=sys.stderr)
        return 2

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
