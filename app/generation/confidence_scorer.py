from app.models.citation import CitationResult
from app.models.retrieval import RetrievalResult


class ConfidenceScorer:
    """
    Computes an overall confidence score
    for the generated answer.
    """

    RETRIEVAL_WEIGHT = 0.5
    CITATION_WEIGHT = 0.5

    def score(
        self,
        retrieved_chunks: list[RetrievalResult],
        citation_results: list[CitationResult],
    ) -> float:

        if not retrieved_chunks:
            return 0.0

        scores = [
            chunk.rerank_score
            if chunk.rerank_score is not None
            else chunk.score
            for chunk in retrieved_chunks
        ]

        min_score = min(scores)
        max_score = max(scores)

        # Normalize retrieval scores relative to the
        # retrieved results.
        if max_score == min_score:
            retrieval_score = 1.0
        else:
            normalized_scores = [
                (score - min_score)
                / (max_score - min_score)
                for score in scores
            ]

            retrieval_score = (
                sum(normalized_scores)
                / len(normalized_scores)
            )

        if citation_results:

            citation_score = (
                sum(
                    result.supported
                    for result in citation_results
                )
                / len(citation_results)
            )

        else:

            citation_score = 0.0

        confidence = (
            self.RETRIEVAL_WEIGHT
            * retrieval_score
            +
            self.CITATION_WEIGHT
            * citation_score
        )

        return round(
            max(0.0, min(confidence, 1.0)),
            3,
        )