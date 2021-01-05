class QA:
    def __init__(self, Reader, Ranker):
        self.Reader = Reader
        self.Ranker = Ranker
    
    def query(self, query, num_results=10):
        top_passages = self.Ranker.query(query)
        results = self.Reader.predict(query, top_passages)

        return results[:num_results]
