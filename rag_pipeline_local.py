import sys
import os
import hashlib
import json
import requests
import datetime

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_qdrant import QdrantVectorStore

from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from qdrant_client import QdrantClient

# ==========================================
# CONFIGURATION
# ==========================================
DOCS_FOLDER = "./documents"
INDEX_FILE = "indexed_files.json"
COLLECTION_NAME = "rag_collection"

EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"
LLM_MODEL_NAME = "gemma2:2b"

# ==========================================
# MD5 HASH FUNCTIONS
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

# ==========================================
# OLLAMA & QDRANT CONNECTION
# ==========================================
def check_services():
    services = {
        "Ollama": "http://localhost:11434/api/tags",
        "Qdrant": "http://localhost:6333/healthz"
    }
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                print(f"❌ {name} is not responding correctly.")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to {name}. Is it running?")
            sys.exit(1)
    print("✅ All services reachable.")

# ==========================================
# DATABASE FUNCTIONS
# ==========================================
def ensure_collection_exists(client, collection_name, vector_size=768):
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
        return True
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

def delete_docs_by_source(client, collection_name, source_path):
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_path))]
            )
        )
        return True
    except Exception as e:
        return False

# ==========================================
# DOCUMENT LOADING
# ==========================================
def load_document(path):
    loader = None
    if path.endswith(".txt") or path.endswith(".md"):
        loader = TextLoader(path, encoding="utf-8")
    elif path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(path)
    
    if loader:
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": path,
                "file_type": os.path.splitext(path)[1],
                "indexed_at": datetime.datetime.now().isoformat()
            })
        return docs
    return None

# ==========================================
# MAIN INDEXING FUNCTION
# ==========================================
def check_and_reindex(client, embedding_model):
    stored_index = load_index()
    current_hashes = {}
    for root, _, files in os.walk(DOCS_FOLDER):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith((".txt", ".md", ".pdf", ".docx")):
                try:
                    current_hashes[path] = md5_file(path)
                except Exception as e:
                    print(f"⚠️ Error hashing {path}: {e}")
    
    added = [p for p in current_hashes if p not in stored_index]
    modified = [p for p in current_hashes if p in stored_index and stored_index[p] != current_hashes[p]]
    deleted = [p for p in stored_index if p not in current_hashes]
    
    for path in deleted:
        delete_docs_by_source(client, COLLECTION_NAME, path)
    
    files_to_index = added + modified
    if not files_to_index:
        print("✅ No files need reindexing.")
        save_index(current_hashes)
        return

    print(f"\n🔄 Indexing {len(files_to_index)} file(s)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []
    
    for path in files_to_index:
        delete_docs_by_source(client, COLLECTION_NAME, path)
        try:
            loaded = load_document(path)
            if loaded:
                all_docs.extend(splitter.split_documents(loaded))
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    
    if all_docs:
        qdrant_store = QdrantVectorStore.from_documents(
            all_docs,
            embedding_model,
            client=client,
            collection_name=COLLECTION_NAME
        )
        print(f"✅ Added {len(all_docs)} chunks to database")
    
    save_index(current_hashes)

# ==========================================
# MAIN PROGRAM
# ==========================================
def main():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"📁 Created folder '{DOCS_FOLDER}'. Add files and run again!")
        return
    
    print(f"🧠 Loading embedding model...")
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    print("🔌 Connecting to Qdrant...")
    client = QdrantClient(url="http://localhost:6333")
    ensure_collection_exists(client, COLLECTION_NAME)
    check_and_reindex(client, embedding_model)
    
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer using this context:\n{context}\n\nQuestion: {question}"
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n🚀 RAG SYSTEM READY. Type 'exit' to quit.\n")
    while True:
        question = input("🙋 Question: ").strip()
        if question.lower() in ['exit', 'quit']: break
        if not question: continue
        
        print("🤖 Thinking...", end="", flush=True)
        try:
            print("\r🤖 Answer: ", end="", flush=True)
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    check_services()
    main()