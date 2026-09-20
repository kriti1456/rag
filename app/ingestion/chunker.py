import re
from typing import List

import numpy as np

from langchain_core.documents import Document

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from app.models.document import ChunkingStrategy
from app.services.embedding_service import embedding_service


class Chunker:
    """
    Handles different document chunking strategies.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # --------------------------------------------------
    # Fixed Chunking
    # --------------------------------------------------

    def fixed_chunk(
        self,
        documents: List[Document]
    ) -> List[Document]:

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return splitter.split_documents(documents)

    # --------------------------------------------------
    # Recursive Chunking
    # --------------------------------------------------

    def recursive_chunk(
        self,
        documents: List[Document]
    ) -> List[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        return splitter.split_documents(documents)

    # --------------------------------------------------
    # Semantic Chunking
    # --------------------------------------------------

    def semantic_chunk(
        self,
        documents: List[Document],
        similarity_threshold: float = 0.65,
    ) -> List[Document]:
        """
        Groups consecutive sentences based on semantic similarity.

        Sentences with similar meanings remain in the same chunk.
        When similarity drops below the threshold, a new chunk starts.
        """

        semantic_chunks = []

        for document in documents:

            text = document.page_content.strip()

            if not text:
                continue

            # Split document into sentences.
            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

            if not sentences:
                continue

            # If there is only one sentence,
            # no semantic comparison is necessary.
            if len(sentences) == 1:

                semantic_chunks.append(
                    Document(
                        page_content=sentences[0],
                        metadata=document.metadata.copy(),
                    )
                )

                continue

            # Create embeddings for all sentences.
            embeddings = embedding_service.embed_documents(
                sentences
            )

            current_chunk = [sentences[0]]

            for i in range(1, len(sentences)):

                previous_embedding = np.array(
                    embeddings[i - 1]
                )

                current_embedding = np.array(
                    embeddings[i]
                )

                # Cosine similarity
                similarity = np.dot(
                    previous_embedding,
                    current_embedding,
                ) / (
                    np.linalg.norm(previous_embedding)
                    * np.linalg.norm(current_embedding)
                )

                if similarity >= similarity_threshold:

                    # Similar meaning → keep together.
                    current_chunk.append(
                        sentences[i]
                    )

                else:

                    # Meaning changed → start new chunk.
                    semantic_chunks.append(
                        Document(
                            page_content=" ".join(
                                current_chunk
                            ),
                            metadata=document.metadata.copy(),
                        )
                    )

                    current_chunk = [
                        sentences[i]
                    ]

            # Add final chunk.
            if current_chunk:

                semantic_chunks.append(
                    Document(
                        page_content=" ".join(
                            current_chunk
                        ),
                        metadata=document.metadata.copy(),
                    )
                )

        return semantic_chunks

    # --------------------------------------------------
    # Strategy Dispatcher
    # --------------------------------------------------

    def chunk(
        self,
        documents: List[Document],
        strategy: ChunkingStrategy,
    ) -> List[Document]:

        if strategy == ChunkingStrategy.FIXED:

            return self.fixed_chunk(documents)

        elif strategy == ChunkingStrategy.RECURSIVE:

            return self.recursive_chunk(documents)

        elif strategy == ChunkingStrategy.SEMANTIC:

            return self.semantic_chunk(documents)

        raise ValueError(
            f"Unsupported chunking strategy: {strategy}"
        )