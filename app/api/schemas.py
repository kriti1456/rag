from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request model for asking questions.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="User question."
    )

class SourceResponse(BaseModel):
    id: str
    title: str
    score: float | None = None
    rerank_score: float | None = None
    retriever: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_index: int | None = None
    strategy: str | None = None
    char_count: int | None = None


class AskResponse(BaseModel):

    answer: str

    confidence: float | None

    model: str | None

    latency_ms: float | None

    retrieved_chunks: int

    citations: int

    sources: list[SourceResponse]


class IngestRequest(BaseModel):
    """
    Request model for document ingestion.
    """

    document_path: str

    strategy: str = "recursive"