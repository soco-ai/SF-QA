import os
import requests
from requests.auth import HTTPBasicAuth

host_url = "https://api.soco.ai/v1/sfqa/query"

INDEX_MAP = {
    'sparta-en-wiki-2016':  {'lang': 'en', 'index': 'wiki-frame-2016'},
    'sparta-zh-wiki-2020':  {'lang': 'zh', 'index': 'zh-wiki-frame-2020'},
    'bm25-en-wiki-2016':    {'lang': 'en', 'index': 'bm25_wiki_2016_en'},
}

class Ranker:
    def __init__(self, index):
        if index not in INDEX_MAP:
            raise ValueError('{} not existed, try one from {}'.format(index, INDEX_MAP.keys()))
        self.index = INDEX_MAP[index]
        

    def query(self, query):
        """
        query api and get ranker results

        :param query: a string of natural language question
        :type query: str
        :return: a list of dictionaries containing topn retrieved answers
        :rtype: list
        """
        headers = {"Accept": "application/json", "Authorization": 'soco_research'}
        json_body = {   
            "lang": self.index['lang'],
            "index": self.index['index'],
            "model_id": "",
            "query": query,
            "params": {
                "top_k": 50,
                "n_best": 50,
                "ranker_only":True
            }
        }

        r = requests.post(url=host_url, json=json_body, headers=headers)
        r.raise_for_status()
        res = r.json()['result']
        # clean res
        for ans in res:
            ans['answer'] = ans['answer']['context']
        
        return res


if __name__ == '__main__':
    ranker = Ranker('sparta-en-wiki-2016')
    res = ranker.query('when was microsoft founded?')
    print(res)