"""Ingestion entrypoint: PDFs in /data -> chunks -> embeddings -> Chroma.

Usage:
    python -m src.ingestion.ingest [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import time

from src import config
from src.ingestion.chunker import Chunk, chunk_pages
from src.ingestion.doc_classifier import classify_document
from src.ingestion.loader import PageText, load_pdfs
from src.llm.gemini_client import embed_texts_in_batches
from src.vectorstore.chroma_store import ChromaStore


def _chunk_all_pages(pages: list[PageText]) -> list[Chunk]:
    """Groups pages by source file, classifies each file, and chunks it with
    the chunk size/overlap appropriate to its doc_type."""
    all_chunks: list[Chunk] = []
    # load_pdfs() emits pages in-order per file, so this grouping is safe.
    for source_file, file_pages in itertools.groupby(pages, key=lambda p: p.source_file):
        doc_type, client_name = classify_document(source_file)
        if doc_type == "policy":
            chunk_size = config.POLICY_CHUNK_SIZE_CHARS
            overlap = config.POLICY_CHUNK_OVERLAP_CHARS
        else:
            chunk_size = config.TRANSCRIPT_CHUNK_SIZE_CHARS
            overlap = config.TRANSCRIPT_CHUNK_OVERLAP_CHARS

        file_chunks = chunk_pages(list(file_pages), chunk_size, overlap)
        for chunk in file_chunks:
            chunk.doc_type = doc_type
            chunk.client_name = client_name
        print(
            f"  {source_file}: doc_type={doc_type}"
            f"{f', client_name={client_name}' if client_name else ''}"
            f" -> {len(file_chunks)} chunk(s) (size={chunk_size}, overlap={overlap})"
        )
        all_chunks.extend(file_chunks)
    return all_chunks


async def run_ingestion(reset: bool) -> None:
    start = time.perf_counter()

    print(f"Loading PDFs from {config.DATA_DIR} ...")
    pages = load_pdfs(config.DATA_DIR)
    print(f"  extracted text from {len(pages)} page(s)")

    chunks = _chunk_all_pages(pages)
    print(f"  split into {len(chunks)} chunk(s) total")

    print(f"Embedding {len(chunks)} chunk(s) with {config.EMBEDDING_MODEL} ...")
    texts = [c.text for c in chunks]
    embeddings = await embed_texts_in_batches(
        texts, task_type="RETRIEVAL_DOCUMENT", batch_size=config.EMBEDDING_BATCH_SIZE
    )

    store = ChromaStore()
    if reset:
        print("Resetting existing collection ...")
        store.reset()

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "doc_type": c.doc_type,
            "client_name": c.client_name,
        }
        for c in chunks
    ]
    store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    elapsed = time.perf_counter() - start
    print(
        f"Done. Collection '{config.CHROMA_COLLECTION_NAME}' now has "
        f"{store.count()} chunk(s). ({elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before ingesting (re-index from scratch).",
    )
    args = parser.parse_args()
    asyncio.run(run_ingestion(reset=args.reset))


if __name__ == "__main__":
    main()
