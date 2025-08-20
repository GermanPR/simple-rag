#!/usr/bin/env python3
"""Debug script to test retrieval components independently."""

from app.core.config import config
from app.llm.mistral_client import EmbeddingManager, MistralClient
from app.retriever.index import DatabaseManager
from app.retriever.semantic import SemanticSearcher
from app.retriever.keyword import KeywordSearcher
from app.retriever.fusion import HybridRetriever

def main():
    print("Testing retrieval components...")
    
    # Initialize components
    db_manager = DatabaseManager(config.DB_PATH)
    mistral_client = MistralClient()
    embedding_manager = EmbeddingManager(db_manager, mistral_client)
    
    query = "What is artificial intelligence?"
    threshold = 0.18
    
    print(f"Query: {query}")
    print(f"Threshold: {threshold}")
    
    # Test semantic search
    print("\n1. Testing semantic search...")
    semantic_searcher = SemanticSearcher(db_manager)
    query_embedding = embedding_manager.embed_query(query)
    
    semantic_results = semantic_searcher.search(query_embedding, top_k=10, threshold=threshold)
    print(f"Semantic results count: {len(semantic_results)}")
    for i, (chunk_id, score) in enumerate(semantic_results[:5]):
        print(f"  {i+1}. Chunk {chunk_id}: {score:.4f}")
    
    # Test keyword search
    print("\n2. Testing keyword search...")
    keyword_searcher = KeywordSearcher(db_manager)
    keyword_results = keyword_searcher.search(query, top_k=10)
    print(f"Keyword results count: {len(keyword_results)}")
    for i, (chunk_id, score) in enumerate(keyword_results[:5]):
        print(f"  {i+1}. Chunk {chunk_id}: {score:.4f}")
    
    # Test hybrid fusion
    print("\n3. Testing hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        db_manager=db_manager,
        keyword_searcher=keyword_searcher,
        semantic_searcher=semantic_searcher,
        alpha=0.7,
        lambda_param=0.7,
    )
    
    hybrid_results = hybrid_retriever.retrieve(
        query=query,
        query_embedding=query_embedding,
        top_k=5,
        rerank_k=15,
        threshold=threshold,
        use_mmr=True,
    )
    
    print(f"Hybrid results count: {len(hybrid_results)}")
    for i, (chunk_id, score, chunk_info) in enumerate(hybrid_results):
        filename = chunk_info.get('filename', 'unknown')
        page = chunk_info.get('page', '?')
        print(f"  {i+1}. Chunk {chunk_id}: {score:.4f} [{filename} p.{page}]")

if __name__ == "__main__":
    main()