# -*- coding: utf-8 -*-
import sys
import os
from dotenv import load_dotenv

# Uvozi koji su ti neophodni za rad celog lanca
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Učitavanje .env fajla
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("⚠️ Greška: GROQ_API_KEY nije pronađen u okruženju!")
    sys.exit()

FOLDER_SA_DOKUMENTIMA = "./dokumentacija"

# ==========================================
# 1. UČITAVANJE I CEPKANJE FAJLOVA
# ==========================================
print("📂 Učitavam dokumente iz foldera...")
if not os.path.exists(FOLDER_SA_DOKUMENTIMA):
    os.makedirs(FOLDER_SA_DOKUMENTIMA)

loader_txt = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
loader_md = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
dokumenti = loader_txt.load() + loader_md.load()

if not dokumenti:
    print(f"⚠️ Folder '{FOLDER_SA_DOKUMENTIMA}' je prazan. Ubaci fajlove pa pokreni ponovo.")
    sys.exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
izrezani_delovi = text_splitter.split_documents(dokumenti)

# ==========================================
# 2. VEKTORSKA BAZA (Povezivanje na Qdrant u Dockeru)
# ==========================================
print("🧠 Indeksiranje podataka u Qdrant vektorsku bazu...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# QDRANT_HOST se vuče iz docker-compose.yml, lokalno se vraća na localhost ako testiraš van Dockera
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
client = QdrantClient(host=qdrant_host, port=6333)

# Kreiramo indeks direktno iz dokumenata unutar Qdrant-a
vektorska_baza = Qdrant.from_documents(
    documents=izrezani_delovi,
    embedding=embedding_model,
    host=qdrant_host,
    port=6333,
    collection_name="moj_rag_collection"
)
retriever = vektorska_baza.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 3. POVEZIVANJE SA GROQ-OM
# ==========================================
print("⚡ Povezujem se na Groq API...")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)

# ==========================================
# 4. RAG PROMPT I LCEL LANAC
# ==========================================
sistemski_prompt = (
    "Ti si koristan asistent. Odgovori na pitanje isključivo koristeći priloženi kontekst ispod.\n"
    "Ako u kontekstu nema odgovora, reci 'Ne znam odgovor na to pitanje na osnovu internih dokumenata.'\n\n"
    "Kontekst:\n{context}\n\n"
    "Pitanje: {question}"
)

prompt = ChatPromptTemplate.from_template(sistemski_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_lanac = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. INTERAKTIVNI RAD
# ==========================================
print("\n🚀 RAG sistem sa Groq-om i Qdrant-om je spreman!")
print("Upisi 'izlaz' za kraj programa.\n")

while True:
    pitanje = input("🙋 Postavi pitanje Groqu: ")
    if pitanje.lower() in ['izlaz', 'exit', 'quit']:
        print("Doviđenja!")
        break
        
    if not pitanje.strip():
        continue

    print("⚡ Groq razmišlja...")
    odgovor = rag_lanac.invoke(pitanje)
    print(f"\n🤖 Odgovor:\n{odgovor}\n")
    print("-" * 50)