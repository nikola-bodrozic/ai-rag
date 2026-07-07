# Python RAG

App requires Python 3.12

add txt, PDF files in `dokumentacija/` folder.

bring up QDRANT vector base

```sh
docker compose up
```
## Install and run pythom app 

```sh
ollama list
NAME                       ID              SIZE      MODIFIED    
nomic-embed-text:latest    0a109f422b47    274 MB    3 days ago     
gemma2:2b                  8ccf136fdd52    1.6 GB    13 days ago  
```
create environment `python -m venv .venv`

activate it ` .venv\Scripts\activate `

install deps. `pip install -r requirements.txt --no-build-isolation`

run `python .\rag_pipeline_local.py` and `python .\rag_pipeline_groq.py`