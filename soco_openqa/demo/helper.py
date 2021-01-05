import re
from rich.console import Console
from rich.table import Table


def _normalize_text(text):
    return re.sub('\s+', ' ', text)

def display(results):

    console = Console()
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("doc_id",style="dim",width=12)
    table.add_column("passage")
    table.add_column("answer",width=30)
    table.add_column("score",style="dim",width=15)

    for r in results:
        doc_id = str(r['source']['doc_id'])
        passage = str(r['source']['context'])
        passage = _normalize_text(passage)
        answer_span = r['answer_span']
        answer = str(r['value'])
        score = '{:.4f}'.format(r['score'])
        passage_with_ans = '{}[red]{}[/red]{}'.format(passage[:answer_span[0]], passage[answer_span[0]:answer_span[1]], passage[answer_span[1]:])
        
        table.add_row(
            doc_id, passage_with_ans, answer, score,
        )
    
    console.print(table)
