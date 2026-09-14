"""Render a chapter with a separate, hidden Word instance and inspect pagination."""
from pathlib import Path
import argparse
import json
import win32com.client
import pymupdf as fitz


def main():
    parser=argparse.ArgumentParser();parser.add_argument('document');args=parser.parse_args()
    path=Path(args.document).resolve();pdf=path.with_suffix('.pdf')
    app=win32com.client.DispatchEx('Word.Application')
    app.Visible=False;app.DisplayAlerts=0
    document=None
    try:
        document=app.Documents.Open(str(path),ReadOnly=True,AddToRecentFiles=False,Visible=False)
        document.Repaginate()
        document.ExportAsFixedFormat(str(pdf),17,OpenAfterExport=False)
    finally:
        if document is not None:document.Close(False)
        app.Quit()
    doc=fitz.open(pdf)
    record={'pdf':str(pdf),'pages':len(doc),'page_details':[]}
    for i,p in enumerate(doc):
        text=p.get_text()
        record['page_details'].append({'page':i+1,'words':len(text.split()),
          'first_lines':text.splitlines()[:3],'last_lines':text.splitlines()[-3:]})
    path.with_suffix('.pagination.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
