# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Uvoz RAG i Memory komponenti
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, BSHTMLLoader
from qdrant_client import QdrantClient

# 🔥 NOVI UVOZI ZA MEMORIJU
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Učitavanje .env fajla
load_dotenv()

# ==========================================
# 1. LIFESPAN MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Aplikacija se pokreće i inicijalizuje RAG...")
    try:
        initialize_rag_pipeline()
    except Exception as e:
        print(f"❌ Fatalna greška pri inicijalizaciji RAG-a: {e}")
    
    yield
    print("🛑 Aplikacija se gasi...")

app = FastAPI(
    title="AI RAG Pipeline API",
    description="API za RAG sa Groq, Qdrant i Memorijom",
    version="1.1.0",
    lifespan=lifespan
)

# ==========================================
# 2. GLOBALNE PROMENLJIVE & MEMORIJA
# ==========================================
rag_chain = None
embedding_model = None
vectorstore = None

# 🔥 Skladište za istoriju razgovora (In-Memory)
sessions_db = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Pomaže sistemu da pronađe ili kreira istoriju za određeni session_id"""
    if session_id not in sessions_db:
        sessions_db[session_id] = ChatMessageHistory()
    return sessions_db[session_id]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"⚠️ Greška pri računanju heša za {file_path}: {e}")
        return None

def initialize_rag_pipeline():
    """Inicijalizuje RAG pipeline, proverava MD5 i osvežava Qdrant"""
    global rag_chain, embedding_model, vectorstore
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ Greška: GROQ_API_KEY nije pronađen u okruženju!")
        raise Exception("Missing GROQ_API_KEY")
    
    FOLDER_SA_DOKUMENTIMA = "./dokumentacija"
    HASH_DATABASE_PATH = "./indexed_files.json"
    
    if not os.path.exists(FOLDER_SA_DOKUMENTIMA):
        os.makedirs(FOLDER_SA_DOKUMENTIMA)
    
    stari_hes_podaci = {}
    if os.path.exists(HASH_DATABASE_PATH):
        try:
            with open(HASH_DATABASE_PATH, "r", encoding="utf-8") as f:
                stari_hes_podaci = json.load(f)
        except Exception:
            stari_hes_podaci = {}

    loader_txt = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    loader_md = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    loader_pdf = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_html = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.html", loader_cls=BSHTMLLoader)
    
    svi_dokumenti = loader_txt.load() + loader_md.load() + loader_pdf.load() + loader_html.load()
    
    novi_ili_izmenjeni_dokumenti = []
    novi_hes_podaci = {}

    for doc in svi_dokumenti:
        izvorna_putanja = doc.metadata.get('source')
        if not izvorna_putanja:
            continue
            
        trenutni_hes = get_file_md5(izvorna_putanja)
        if not trenutni_hes:
            continue
            
        novi_hes_podaci[izvorna_putanja] = trenutni_hes
        
        if stari_hes_podaci.get(izvorna_putanja) != trenutni_hes:
            novi_ili_izmenjeni_dokumenti.append(doc)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    COLLECTION_NAME = "dokumentacija"

    if novi_ili_izmenjeni_dokumenti:
        print(f"♻️ Detektovano {len(novi_ili_izmenjeni_dokumenti)} dokumenata za osvežavanje.")
        
        vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)
        from qdrant_client.http import models as rest_models
        
        izmenjeni_fajlovi = set(doc.metadata.get('source') for doc in novi_ili_izmenjeni_dokumenti if doc.metadata.get('source'))
        for stari_fajl in izmenjeni_fajlovi:
            print(f"🗑️ Čistim stare zapise iz Qdrant baze za: {stari_fajl}")
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=rest_models.Filter(
                    must=[rest_models.FieldCondition(key="metadata.source", match=rest_models.MatchValue(value=stari_fajl))]
                )
            )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        izrezani_delovi = text_splitter.split_documents(novi_ili_izmenjeni_dokumenti)
        
        print(f"✂️ Tekst isečen na {len(izrezani_delovi)} chunk-ova. Indeksiram u Qdrant...")
        vectorstore.add_documents(izrezani_delovi)
            
        with open(HASH_DATABASE_PATH, "w", encoding="utf-8") as f:
            json.dump(novi_hes_podaci, f, indent=4, ensure_ascii=False)
        print("💾 Lista MD5 heševa je uspešno ažurirana.")
    else:
        print("🔒 Svi fajlovi su već indeksirani (MD5 se poklapa). Preskačem slanje u Qdrant.")
        vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_model)

    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.2)
    
    # 🎯 STABILAN PROMPT: Kombinuje kontekst iz baze i istoriju poruka
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ti si koristan asistent. Odgovori na pitanje korisnika koristeći priloženi kontekst (ukoliko je relevantan za pitanje).\n\nKontekst iz dokumenata:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # Lanac koji prima čist rečnik sa gotovim 'context' i 'question' stringovima
    osnovni_rag_chain = prompt | llm | StrOutputParser()
    
    # Povezujemo sa istorijom razgovora
    rag_chain = RunnableWithMessageHistory(
        osnovni_rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    print("✅ RAG pipeline sa MEMORIJOM je uspešno konfigurisan!")

# ==========================================
# 3. PYDANTIC MODELI SA SESSION_ID
# ==========================================
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"  # 🔥 Dodat parametar za praćenje sesije

class QueryResponse(BaseModel):
    question: str
    answer: str
    session_id: str

# ==========================================
# 4. API ENDPOINTI
# ==========================================
@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI RAG Pipeline API sa Memorijom je pokrenut"}

# ==========================================
# 4. API ENDPOINTI (Ažuriran sa stabilnim retriever pozivom)
# ==========================================
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if rag_chain is None or vectorstore is None:
        raise HTTPException(status_code=503, detail="RAG pipeline nije spreman.")
    
    try:
        # 1. Ručno izvlačimo relevantne dokumente iz Qdranta za trenutno pitanje
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        pronadjeni_docs = retriever.invoke(request.question)
        kontekst_str = format_docs(pronadjeni_docs)
        
        # 2. Prosleđujemo čist string u RunnableWithMessageHistory
        answer = rag_chain.invoke(
            {
                "question": request.question,
                "context": kontekst_str
            },
            config={"configurable": {"session_id": request.session_id}}
        )
        return QueryResponse(question=request.question, answer=answer, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)