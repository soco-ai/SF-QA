import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QA:
    def __init__(self, Reader, Ranker):
        self.Reader = Reader
        self.Ranker = Ranker
    
    def query(self, query, num_results=10):
        logger.info('Start ranking...')
        top_passages = self.Ranker.query(query)
        logger.info('Start reading...')
        results = self.Reader.predict(query, top_passages)

        return results[:num_results]
