from soco_openqa.demo import Reader, Ranker, QA, display

reader = Reader(model='squad-ds-context-global-norm-2016sparta-from-pt')

ranker = Ranker(index='sparta-en-wiki-2016')

qa = QA(reader, ranker)

while True:
    q = input('Enter a query: ')
    results = qa.query(q, num_results=3)
    display(results)
