class Reranker:
    def __init__(self, api_key):
        import cohere         # Local import to avoid unnecessary dependency for other routes
        self.client = cohere.Client(api_key)

    def rerank(self, query, documents, top_k=3):
        response = self.client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[{"text": d} for d in documents],
            top_n=top_k,
        )

        ranked = []
        for r in response.results:
            ranked.append({
                "text": documents[r.index],
                "score": r.relevance_score
            })

        return ranked