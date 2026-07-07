# Python RAG

App requires Python 3.12

add txt, PDF files in `dokumentacija/` folder.

bring up QDRANT vector base

```sh
docker compose up
```
install and run pythom app 

create environment `python -m venv .venv`
activate it ` .venv\Scripts\activate `
run `python .\rag_pipeline_local.py`