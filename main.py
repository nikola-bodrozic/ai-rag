# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Uvoz RAG pipeline komponenti
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, BSHTMLLoader
from qdrant_client import QdrantClient
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
    
    yield  # Ovde aplikacija živi i prima zahteve
    
    print("🛑 Aplikacija se gasi...")

# 🎯 KLJUČNO: Inicijalizacija FastAPI objekta koji je nedostajao!
app = FastAPI(
    title="AI RAG Pipeline API",
    description="API za Retrieval-Augmented Generation sa Groq i Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# 2. GLOBALNE PROMENLJIVE & PIPELINE
# ==========================================
rag_chain = None
embedding_model = None
vectorstore = None

def format_docs(docs):
    """Pomoćna funkcija koja spaja izvučene dokumente u jedan string"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_file_md5(file_path):
    """Pomoćna funkcija koja računa MD5 heš za zadati fajl"""
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
    """Inicijalizuje RAG pipeline i indeksira SAMO nove ili izmenjene fajlove"""
    global rag_chain, embedding_model, vectorstore
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ Greška: GROQ_API_KEY nije pronađen u okruženju!")
        raise Exception("Missing GROQ_API_KEY")
    
    FOLDER_SA_DOKUMENTIMA = "./dokumentacija"
    HASH_DATABASE_PATH = "./indexed_files.json"
    
    print("📂 Skeniram folder sa dokumentima...")
    if not os.path.exists(FOLDER_SA_DOKUMENTIMA):
        os.makedirs(FOLDER_SA_DOKUMENTIMA)
    
    # Učitavamo staru bazu heševa ako postoji
    stari_hes_podaci = {}
    if os.path.exists(HASH_DATABASE_PATH):
        try:
            with open(HASH_DATABASE_PATH, "r", encoding="utf-8") as f:
                stari_hes_podaci = json.load(f)
        except Exception:
            print("⚠️ Fajl sa heševima je oštećen, biće ponovo kreiran.")
            stari_hes_podaci = {}

    # --- DEFINIŠEMO SVE LOADERE ---
    loader_txt = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    loader_md = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    loader_pdf = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_html = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.html", loader_cls=BSHTMLLoader)
    
    print("⏳ Proveravam strukturu dokumenata...")
    svi_dokumenti = loader_txt.load() + loader_md.load() + loader_pdf.load() + loader_html.load()
    
    novi_ili_izmenjeni_dokumenti = []
    novi_hes_podaci = {}

    # Filtriranje fajlova na osnovu MD5 heša
    for doc in svi_dokumenti:
        izvorna_putanja = doc.metadata.get('source')
        if not izvorna_putanja:
            continue
            
        trenutni_hes = get_file_md5(izvorna_putanja)
        if not trenutni_hes:
            continue
            
        novi_hes_podaci[izvorna_putanja] = trenutni_hes
        
        # Ako je fajl nov ili mu se heš razlikuje, ide na indeksiranje
        if stari_hes_podaci.get(izvorna_putanja) != trenutni_hes:
            novi_ili_izmenjeni_dokumenti.append(doc)

    # Inicijalizacija embedding modela
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    # 🎯 FIX: Koristimo zvanični QdrantClient za stabilnu vezu sa serverom
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    COLLECTION_NAME = "dokumentacija"

    # Ako ima promena, šaljemo ih u Qdrant
    if novi_ili_izmenjeni_dokumenti:
        print(f"♻️ Detektovano {len(novi_ili_izmenjeni_dokumenti)} dokumenata za osvežavanje.")
        
        # Inicijalizujemo vectorstore preko klijenta na samom početku
        vectorstore = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embedding_model
        )
        
        # 🔥 POBOLJŠANJE: Brišemo stare chunk-ove iz Qdrant-a za fajlove koji su izmenjeni
        from qdrant_client.http import models as rest_models
        
        # Izvlačimo jedinstvene putanje fajlova koji se ponovo indeksiraju
        izmenjeni_fajlovi = set(doc.metadata.get('source') for doc in novi_ili_izmenjeni_dokumenti if doc.metadata.get('source'))
        
        for stari_fajl in izmenjeni_fajlovi:
            print(f"🗑️ Čistim stare zapise iz Qdrant baze za: {stari_fajl}")
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="metadata.source",
                            match=rest_models.MatchValue(value=stari_fajl),
                        )
                    ]
                )
            )

        # Sada bezbedno sečemo i dodajemo novu verziju dokumenata
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        izrezani_delovi = text_splitter.split_documents(novi_ili_izmenjeni_dokumenti)
        
        print(f"✂️ Tekst isečen na {len(izrezani_delovi)} chunk-ova. Indeksiram u Qdrant...")
        vectorstore.add_documents(izrezani_delovi)
            
        # Čuvamo heševe u fajl
        with open(HASH_DATABASE_PATH, "w", encoding="utf-8") as f:
            json.dump(novi_hes_podaci, f, indent=4, ensure_ascii=False)
        print("💾 Lista MD5 heševa je uspešno ažurirana.")
    else:
        print("🔒 Svi fajlovi su već indeksirani (MD5 se poklapa). Preskačem slanje u Qdrant.")
        # Povezujemo se na postojeću kolekciju stabilno preko klijenta
        vectorstore = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embedding_model
        )

    # Sklapanje LCEL lanca sa Groq API-jem
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.2)
    template = """Na osnovu sledećeg konteksta, odgovori na pitanje:\n\nKontekst:\n{context}\n\nPitanje: {question}\n\nOdgovor:"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG pipeline je uspešno konfigurisan!")
    
# ==========================================
# 3. PYDANTIC MODELI ZA ZAHTEVE/ODGOVORE
# ==========================================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

# ==========================================
# 4. API ENDPOINTI
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "AI RAG Pipeline API je pokrenut",
        "endpoints": {"query": "/query", "health": "/health", "docs": "/docs"}
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_pipeline_ready": rag_chain is not None
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline nije spreman.")
    
    try:
        answer = rag_chain.invoke(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)