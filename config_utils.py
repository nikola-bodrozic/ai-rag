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
COLLECTION_NAME = "rag_shared_collection"
EMBEDDING_MODEL = "nomic-embed-text:latest"
BATCH_SIZE = 100  # Number of chunks to index at once

def md5_file(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_collection(client):
    """Ensures the Qdrant collection exists, creates it if not."""
    try:
        collections = client.get_collections()
        if not any(c.name == COLLECTION_NAME for c in collections.collections):
            print(f"📦 Creating shared collection: {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        return True
    except Exception as e:
        print(f"❌ Error ensuring collection exists: {e}")
        return False

def collection_exists(client):
    """Checks if the collection exists without raising exceptions."""
    try:
        collections = client.get_collections()
        return any(c.name == COLLECTION_NAME for c in collections.collections)
    except Exception:
        return False

def delete_points_for_path(client, path):
    """Safely deletes points associated with a specific file path."""
    try:
        # First check if collection exists before attempting delete
        if not collection_exists(client):
            print(f"⚠️ Collection {COLLECTION_NAME} doesn't exist, skipping delete for {path}")
            return True
            
        client.delete(
            COLLECTION_NAME, 
            Filter(must=[FieldCondition(key="source", match=MatchValue(value=path))])
        )
        print(f"🗑️ Deleted existing points for: {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"⚠️ Could not delete points for {path}: {e}")
        return False

def sync_documents(client, embedding_model):
    """Handles hashing, deletions, and indexing of new/modified files with batching."""
    if not os.path.exists(DOCS_FOLDER):
        print(f"⚠️ Documents folder '{DOCS_FOLDER}' not found. Creating it...")
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        return
        
    # Load stored index
    if not os.path.exists(INDEX_FILE):
        stored_index = {}
    else:
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                stored_index = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Error reading index file: {e}. Starting fresh.")
            stored_index = {}
            
    # Calculate current hashes
    current_hashes = {}
    for root, _, files in os.walk(DOCS_FOLDER):
        for file in files:
            if file.endswith((".txt", ".md", ".pdf", ".docx")):
                path = os.path.join(root, file)
                try:
                    current_hashes[path] = md5_file(path)
                except Exception as e:
                    print(f"⚠️ Error hashing {path}: {e}")
    
    # Identify changes
    to_delete = [p for p in stored_index if p not in current_hashes]
    to_index = [p for p in current_hashes if p not in stored_index or stored_index[p] != current_hashes[p]]
    
    # Clean up deleted files
    if to_delete:
        print(f"🗑️ Removing {len(to_delete)} deleted files from index...")
        for path in to_delete:
            delete_points_for_path(client, path)
    
    # Index new/modified files with batching
    if to_index:
        print(f"📝 Processing {len(to_index)} new/modified files...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        
        for path in to_index:
            try:
                # Remove existing points for this file (if any)
                delete_points_for_path(client, path)
                
                # Load the document
                if path.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif path.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(path)
                else:
                    loader = TextLoader(path, encoding="utf-8")
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({"source": path})
                
                # Split into chunks
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                print(f"✅ Loaded {len(chunks)} chunks from: {os.path.basename(path)}")
                
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")
                continue
        
        # Index chunks in batches to avoid memory issues
        if all_chunks:
            print(f"🔄 Indexing {len(all_chunks)} total chunks in batches of {BATCH_SIZE}...")
            
            # Create vector store once (Fix #10)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
            
            # Process in batches
            total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch = all_chunks[i:i+BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                try:
                    vector_store.add_documents(batch)
                    print(f"✅ Indexed batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                except Exception as e:
                    print(f"❌ Error indexing batch {batch_num}: {e}")
                    # Continue with next batch instead of failing entirely
                    continue
        else:
            print("ℹ️ No new chunks to index.")
    
    # Save updated index
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(current_hashes, f, indent=4)
        print(f"✅ Index file updated: {INDEX_FILE}")
    except Exception as e:
        print(f"❌ Error saving index file: {e}")