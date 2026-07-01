# Run Docker Stack

add txt, md, HTML and PDF files in `dokumentacija/` folder.

```sh
docker compose up
```

visit <http://localhost:8000/docs#/default/query_rag_query_post> for API call

note \u017 unicode character for serbian letters 

```sh
curl -X POST http://localhost:8000/query -H "Content-Type: application/json; charset=utf-8" -d $'{"question": "Kako se ja zovem i koji jezik najvi\u0171e volim?", "session_id": "sesija_999"}'

{"question":"Kako se ja zovem i koji jezik najviue volim?","answer":"Tvoje ime je Nikola, a najviše voliš Python.","session_id":"sesija_999"}

```