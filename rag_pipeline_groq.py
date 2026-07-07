import sys
import os
import hashlib
import json
from dotenv import load_dotenv

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env file!")
    print("Please add: GROQ_API_KEY=your_api_key_here")
    sys.exit(1)

# ==========================================
# IMPORTS - Using standalone GROQ package
# ==========================================
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq  # ← NEW: standalone package!
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance

# Document loaders - still needed for PDF, etc.
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# ==========================================
# CONFIGURATION
# ==========================================
FOLDER_SA_DOKUMENTIMA = "./dokumentacija"
INDEX_FILE = "indexed_files_groq.json"
COLLECTION_NAME = "rag_collection_groq"

# ==========================================
# 0. MD5 UTIL FUNKCIJE
# ==========================================
def md5_file(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_index(index):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

def ensure_collection_exists(client, collection_name, vector_size=768):
    """Create collection if it doesn't exist"""
    try:
        collections = client.get_collections()
        exists = any(c.name == collection_name for c in collections.collections)
        
        if not exists:
            print(f"📦 Creating collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{collection_name}' created!")
        else:
            print(f"✅ Collection '{collection_name}' already exists")
        return True
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

def check_and_reindex(client, embedding_model):
    current_index = load_index()
    new_index = {}
    changed_files = []

    for root, _, files in os.walk(FOLDER_SA_DOKUMENTIMA):
        for file in files:
            path = os.path.join(root, file)
            file_hash = md5_file(path)
            new_index[path] = file_hash
            if path not in current_index or current_index[path] != file_hash:
                changed_files.append(path)

    if changed_files:
        print(f"🔄 Promenjeni fajlovi: {changed_files}")
        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for path in changed_files:
            if path.endswith(".txt") or path.endswith(".md"):
                loader = TextLoader(path, encoding="utf-8")
            elif path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                print(f"⚠️ Preskačem nepodržan fajl: {path}")
                continue

            loaded = loader.load()
            docs.extend(splitter.split_documents(loaded))

            try:
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[FieldCondition(key="source", match=MatchValue(value=path))]
                    )
                )
                print(f"🗑️  Deleted old documents from: {path}")
            except Exception as e:
                print(f"ℹ️  No old documents to delete for {path}")

        if docs:
            ensure_collection_exists(client, COLLECTION_NAME, vector_size=768)
            
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
            qdrant_store.add_documents(docs)
            print(f"✅ Dodato {len(docs)} chunkova u bazu")
        else:
            print("⚠️ Nema dokumenata za indeksiranje")

        save_index(new_index)
        print("✅ Reindeksiranje završeno.")
    else:
        print("✅ Nema promena u fajlovima.")

# ==========================================
# 1. PROVERA FOLDERA
# ==========================================
if not os.path.exists(FOLDER_SA_DOKUMENTIMA):
    os.makedirs(FOLDER_SA_DOKUMENTIMA)
    print(f"📁 Napravljen je folder '{FOLDER_SA_DOKUMENTIMA}'. Ubaci fajlove pa pokreni skriptu ponovo!")
    sys.exit()

# ==========================================
# 2. VEKTORIZACIJA - LOCAL OLLAMA
# ==========================================
print("🧠 Pokretanje embedding modela (nomic-embed-text - LOCAL)...")
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

print("🔌 Povezivanje sa Qdrant bazom...")
client = QdrantClient(url="http://localhost:6333")

if os.path.exists(INDEX_FILE):
    os.remove(INDEX_FILE)
    print("🗑️  Old index file removed - forcing full reindex!")

check_and_reindex(client, embedding_model)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 3. LLM - GROQ API (CLOUD)
# ==========================================
print("\n" + "="*50)
print("🤖 KORISTI SE GROQ API (CLOUD LLM)")
print("="*50)

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",  # Recommended replacement!
    max_tokens=1024,
)

print(f"📊 Model: mixtral-8x7b-32768")
print(f"🌐 Tip: Cloud API (Internet required)")
print(f"💰 Trošak: Besplatno (trenutno)")
print(f"⚡ Brzina: Vrlo brzo (hardverski optimizovano)")
print("="*50 + "\n")

# ==========================================
# 4. RAG PROMPT I LANAC
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
print("\n🚀 RAG sistem je spreman (GROQ verzija)!")
print("="*50)
print(f"🔹 Embedding model: nomic-embed-text (274 MB - LOCAL)")
print(f"🔹 LLM model: mixtral-8x7b-32768 (GROQ - CLOUD)")
print(f"🔹 Kolekcija: {COLLECTION_NAME}")
print(f"🔹 API Key: {'✅ Pronađen' if GROQ_API_KEY else '❌ Nije pronađen'}")
print("="*50)
print("💡 Savet: Poredi odgovore sa lokalnom (Ollama) verzijom!")
print("Upisi 'izlaz' za kraj programa.\n")

while True:
    pitanje = input("🙋 Postavi pitanje: ")
    if pitanje.lower() in ['izlaz', 'exit', 'quit']:
        print("Doviđenja!")
        break

    if not pitanje.strip():
        continue

    print("🤖 Razmišljam (GROQ cloud)...")
    try:
        odgovor = rag_lanac.invoke(pitanje)
        print(f"\n🤖 Odgovor (GROQ):\n{odgovor}\n")
    except Exception as e:
        print(f"❌ Greška: {e}")
    print("-" * 50)