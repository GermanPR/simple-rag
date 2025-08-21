# RAG System Architecture Flow

This diagram shows the high-level flow of the ingest and query endpoints in the RAG system.

```mermaid
flowchart TD
    %% Ingest Flow
    subgraph "Document Ingestion Pipeline"
        A[Upload PDF Documents] --> B[Extract Text from PDFs]
        B --> C[Split Text into Chunks]
        C --> D[Create TF-IDF Index]
        C --> E[Generate Semantic Embeddings]
        D --> F[Store in Database]
        E --> F
        F --> G[Documents Ready for Search]
    end
    
    %% Query Flow
    subgraph "Query Processing Pipeline"
        H[User Query] --> I[Analyze Intent & Rewrite Query]
        I --> J{Query Type?}
        
        J -->|Small Talk| K[Return Conversational Response]
        J -->|Question| L[Hybrid Search]
        
        L --> M[Keyword Search<br/>TF-IDF]
        L --> N[Semantic Search<br/>Vector Similarity]
        
        M --> O[Combine & Rank Results]
        N --> O
        O --> P[Diversify Results<br/>MMR Algorithm]
        P --> Q{Sufficient Evidence?}
        
        Q -->|No| R[Return Insufficient Evidence]
        Q -->|Yes| S[Generate Answer with LLM]
        S --> T[Extract Citations]
        T --> U[Safety Checks]
        U --> U1[Hallucination Detection]
        U --> U2[PII Detection]
        U1 --> V[Return Final Answer with Citations]
        U2 --> V
    end
    
    %% Data Storage
    subgraph "Knowledge Base"
        W[(Document Metadata)]
        X[(Text Chunks)]
        Y[(Vector Embeddings)]
        Z[(Search Indices)]
    end
    
    %% External Services
    AA[Mistral AI API<br/>Embeddings & Chat]
    
    %% Connections
    E -.-> AA
    S -.-> AA
    U1 -.-> AA
    U2 -.-> AA
    
    F -.-> W
    F -.-> X
    F -.-> Y
    F -.-> Z
    
    M -.-> Z
    N -.-> Y
    T -.-> X
```

## Flow Description

### Document Ingestion Pipeline
1. **Upload PDF Documents**: Users upload PDF files to the system
2. **Extract Text**: Text is extracted from PDFs page by page
3. **Split into Chunks**: Text is divided into overlapping chunks for processing
4. **Create TF-IDF Index**: Keyword search indices are built for each chunk
5. **Generate Embeddings**: Semantic embeddings are created using Mistral AI
6. **Store in Database**: All data is persisted in SQLite database

### Query Processing Pipeline
1. **User Query**: User submits a question to the system
2. **Intent Analysis**: LLM analyzes query intent and rewrites if needed
3. **Query Routing**: Small talk gets conversational responses, questions proceed to search
4. **Hybrid Search**: Combines keyword (TF-IDF) and semantic (vector) search
5. **Result Fusion**: Merges and ranks results using alpha weighting
6. **Diversification**: Applies MMR algorithm to reduce redundancy
7. **Evidence Check**: Validates if sufficient evidence exists to answer
8. **Answer Generation**: LLM generates response based on retrieved context
9. **Citation Extraction**: Identifies and formats source citations
10. **Safety Validation**: Checks for hallucinations and personally identifiable information
11. **Final Response**: Returns answer with citations to user