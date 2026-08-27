"""Wrapper around a local persistent Chroma collection.

Chroma stores the embedding, chunk text, and metadata (source file, page,
chunk index) together, which is exactly what the Retriever Agent needs to
produce citations without a separate lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src import config


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    score: float  # similarity in [0, 1] roughly; higher is more relevant
    doc_type: str = ""
    client_name: str = ""


class ChromaStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(config.STORAGE_DIR))
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(config.CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            # cosine distance -> similarity
            similarity = 1 - distance
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=metadata["source_file"],
                    page_number=metadata["page_number"],
                    score=similarity,
                    doc_type=metadata.get("doc_type", ""),
                    client_name=metadata.get("client_name", ""),
                )
            )
        return chunks
