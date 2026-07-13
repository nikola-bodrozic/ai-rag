import os
import hashlib
import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_qdrant import QdrantVectorStore

# --- SHARED CONFIGURATION ---
DOCS_FOLDER = "./documents"
INDEX_FILE = "indexed_files.json"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "nomic-embed-text:latest"

def md5_file(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_collection(client):
    collections = client.get_collections()
    if not any(c.name == COLLECTION_NAME for c in collections.collections):
        print(f"📦 Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

def sync_documents(client, embedding_model):
    """Handles hashing, deletions, and indexing of new/modified files."""
    if not os.path.exists(INDEX_FILE):
        stored_index = {}
    else:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            stored_index = json.load(f)
            
    current_hashes = {os.path.join(r, f): md5_file(os.path.join(r, f)) 
                      for r, _, files in os.walk(DOCS_FOLDER) 
                      for f in files if f.endswith((".txt", ".md", ".pdf", ".docx"))}
    
    # Identify changes
    to_delete = [p for p in stored_index if p not in current_hashes]
    to_index = [p for p in current_hashes if p not in stored_index or stored_index[p] != current_hashes[p]]
    
    # Clean up deleted files
    for path in to_delete:
        client.delete(COLLECTION_NAME, Filter(must=[FieldCondition(key="source", match=MatchValue(value=path))]))
        
    # Index new/modified files
    if to_index:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_docs = []
        for path in to_index:
            client.delete(COLLECTION_NAME, Filter(must=[FieldCondition(key="source", match=MatchValue(value=path))]))
            
            if path.endswith(".pdf"): loader = PyPDFLoader(path)
            elif path.endswith(".docx"): loader = UnstructuredWordDocumentLoader(path)
            else: loader = TextLoader(path, encoding="utf-8")
            
            docs = loader.load()
            for doc in docs: doc.metadata.update({"source": path})
            all_docs.extend(splitter.split_documents(docs))
            
        if all_docs:
            print(f"🔄 Indexing {len(all_docs)} new chunks...")
            QdrantVectorStore.from_documents(all_docs, embedding_model, client=client, collection_name=COLLECTION_NAME)
            
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(current_hashes, f, indent=4)