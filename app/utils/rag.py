from app.utils.vector_store import get_vector_store


def search(query: str, k: int = 5) -> str:
    """Search documents and return relevant context.
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        Combined context from top-k results
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

