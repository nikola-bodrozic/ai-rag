# Python RAG

Training and inference form files in `documents/` folder.

## Resources 

App requires Python 3.12

put GROQ_API_KEY in `.env`

add txt, PDF files in `documents/` folder.

```sh
ollama list
NAME                       ID              SIZE      MODIFIED    
nomic-embed-text:latest    0a109f422b47    274 MB    3 days ago     
gemma2:2b                  8ccf136fdd52    1.6 GB    13 days ago  
```

bring up Qdrant vector base

```sh
docker compose up
```
## Install and run pythom app 

create environment `python -m venv .venv`

activate it ` .venv\Scripts\activate `

install deps. `pip install -r requirements.txt`

run `python .\rag_pipeline_local.py` and `python .\rag_pipeline_groq.py`