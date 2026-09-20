
## RAG Pipeline

```text
Documents
    ↓
Document Loading
    ↓
Chunking
    ↓
Embeddings
    ↓
┌───────────────────────┐
│                       │
Dense Retrieval     BM25 Retrieval
│                       │
└───────────┬───────────┘
            ↓
   Weighted RRF Fusion
            ↓
    Cross-Encoder Reranking
            ↓
         Top-K 
            ↓
      Prompt Building
            ↓
          LLM
            ↓
     Generated Answer
            ↓
 Citation Verification
            ↓
   Confidence Scoring
            ↓
       Final Response
