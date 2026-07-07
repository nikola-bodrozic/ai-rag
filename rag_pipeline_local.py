import sys
import os
import hashlib
import json

# ✅ Novi paketi
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ⚠️ Loaderi su još u community dok ne budu migrirani
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader

# Qdrant klijent
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

FOLDER_SA_DOKUMENTIMA = "./dokumentacija"
INDEX_FILE = "indexed_files.json"

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

            # Brišemo stare dokumente iz kolekcije
            

            client.delete(
                collection_name="rag_collection",
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=path))]
                )
            )


        # Dodaj nove dokumente
        qdrant_store = Qdrant(
            client=client,
            collection_name="rag_collection",
            embeddings=embedding_model
        )
        qdrant_store.add_documents(docs)

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
# 2. VEKTORIZACIJA I QDRANT BAZA
# ==========================================
print("🧠 Pokretanje embedding modela i indeksiranje podataka...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")

check_and_reindex(client, embedding_model)

retriever = Qdrant(
    client=client,
    collection_name="rag_collection",
    embeddings=embedding_model
).as_retriever(search_kwargs={"k": 3})

# ==========================================
# 3. POVEZIVANJE SA OLLAMOM
# ==========================================
llm = OllamaLLM(model="gemma2:2b")

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
print("\n🚀 RAG sistem je spreman za tvoje fajlove!")
print("Upisi 'izlaz' za kraj programa.\n")

while True:
    pitanje = input("🙋 Postavi pitanje: ")
    if pitanje.lower() in ['izlaz', 'exit', 'quit']:
        print("Doviđenja!")
        break

    if not pitanje.strip():
        continue

    print("🤖 Razmišljam...")
    odgovor = rag_lanac.invoke(pitanje)
    print(f"\n🤖 Odgovor:\n{odgovor}\n")
    print("-" * 50)
