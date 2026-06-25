# -*- coding: utf-8 -*-
import os
import sys
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

def initialize_rag_pipeline():
    """Inicijalizuje RAG pipeline pri startanju aplikacije za sve formate"""
    global rag_chain, embedding_model, vectorstore
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ Greška: GROQ_API_KEY nije pronađen u okruženju!")
        raise Exception("Missing GROQ_API_KEY")
    
    FOLDER_SA_DOKUMENTIMA = "./dokumentacija"
    
    print("📂 Skeniram folder sa dokumentima...")
    if not os.path.exists(FOLDER_SA_DOKUMENTIMA):
        os.makedirs(FOLDER_SA_DOKUMENTIMA)
    
    # --- OPTIMIZACIJA: Definišemo sve loadere na jednom mestu ---
    loader_txt = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    loader_md = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    loader_pdf = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.pdf", loader_cls=PyPDFLoader)
    loader_html = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.html", loader_cls=BSHTMLLoader)
    
    # Jedan čist zbirni poziv za sve formate bez gubljenja podataka
    print("⏳ Učitavam sve .txt, .md, .pdf i .html fajlove...")
    dokumenti = loader_txt.load() + loader_md.load() \
                 + loader_pdf.load() \
                 + loader_html.load()
    
    if dokumenti:
        # Zlatna sredina za chunk_size (700 karaktera sa preklapanjem)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        izrezani_delovi = text_splitter.split_documents(dokumenti)
        
        print(f"✂️ Dokumentacija uspešno isečena na {len(izrezani_delovi)} delova.")
        print("🧠 Indeksiranje podataka u Qdrant...")
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        
        vectorstore = Qdrant.from_documents(
            izrezani_delovi,
            embedding_model,
            url=qdrant_url,
            collection_name="dokumentacija"
        )
        
        # Inicijalizacija LLM-a
        llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.2)
        
        template = """Na osnovu sledećeg konteksta, odgovori na pitanje:

Kontekst:
{context}

Pitanje: {question}

Odgovor:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ RAG pipeline je uspešno inicijalizovan sa svim formatima!")
    else:
        print("⚠️ Nema nikakvih dokumenata za indeksiranje u folderu 'dokumentacija'.")

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
        raise HTTPException(status_code=503, detail="RAG pipeline nije inicijalizovan ili nema dokumenata u bazi.")
    
    try:
        answer = rag_chain.invoke(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)