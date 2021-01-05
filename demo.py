from soco_openqa.demo import Reader, Ranker, QA, display

# reader = Reader(model='squad-ds-context-global-norm-2016sparta-from-pt')
reader = Reader(model='cmrc2018-ds-context-global-norm-2018sparta-from-pt-v1')

# ranker = Ranker(index='sparta-en-wiki-2016')
ranker = Ranker(index='sparta-zh-wiki-2020')

qa = QA(reader, ranker)

while True:
    q = input('Enter a query: ')
    results = qa.query(q, num_results=20)
    display(results)
