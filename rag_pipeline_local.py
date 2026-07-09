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
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_index():
    """Load stored file hashes from JSON"""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_index(index):
    """Save file hashes to JSON"""
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

# ==========================================
# OLLAMA CONNECTION
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
        return True
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

def delete_docs_by_source(client, collection_name, source_path):
    """Delete all vectors for a specific source file"""
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
        # Add metadata to every document page/chunk
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
    """
    Compare current files with stored hashes and reindex only changed files.
    
    Returns:
        - added: list of newly indexed files
        - modified: list of modified files (re-indexed)
        - deleted: list of deleted files (removed from DB)
        - unchanged: list of unchanged files (skipped)
    """
    # Load previously stored hashes
    stored_index = load_index()
    
    # Calculate current hashes for all files in documents folder
    current_hashes = {}
    for root, _, files in os.walk(DOCS_FOLDER):
        for file in files:
            path = os.path.join(root, file)
            # Only process supported file types
            if file.endswith((".txt", ".md", ".pdf", ".docx")):
                try:
                    current_hashes[path] = md5_file(path)
                except Exception as e:
                    print(f"⚠️ Error hashing {path}: {e}")
    
    # ==========================================
    # COMPARE: Find what changed
    # ==========================================
    added = []      # New files (not in stored index)
    modified = []   # Existing files with different hash
    deleted = []    # Files in stored index but not found anymore
    unchanged = []  # Files with same hash (skip these)
    
    for path, current_hash in current_hashes.items():
        if path not in stored_index:
            added.append(path)
        elif stored_index[path] != current_hash:
            modified.append(path)
        else:
            unchanged.append(path)
    
    for path in stored_index:
        if path not in current_hashes:
            deleted.append(path)
    
    # ==========================================
    # REPORT STATUS
    # ==========================================
    print("\n" + "="*50)
    print("📊 FILE COMPARISON RESULTS:")
    print("="*50)
    print(f"  ✅ Unchanged: {len(unchanged)} files (skipped)")
    print(f"  🆕 Added:     {len(added)} files")
    print(f"  ✏️  Modified:  {len(modified)} files")
    print(f"  🗑️  Deleted:   {len(deleted)} files")
    print("="*50)
    
    # ==========================================
    # PROCESS DELETED FILES (remove from DB)
    # ==========================================
    for path in deleted:
        print(f"🗑️  Removing deleted file from DB: {path}")
        delete_docs_by_source(client, COLLECTION_NAME, path)
    
    # ==========================================
    # PROCESS ADDED & MODIFIED FILES (reindex)
    # ==========================================
    files_to_index = added + modified
    
    if not files_to_index:
        print("✅ No files need reindexing. Everything is up to date!\n")
        # Save current state even if nothing changed
        save_index(current_hashes)
        return added, modified, deleted, unchanged
    
    print(f"\n🔄 Indexing {len(files_to_index)} file(s)...\n")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []
    
    for path in files_to_index:
        status = "NEW" if path in added else "MODIFIED"
        print(f"  [{status}] {os.path.basename(path)}")
        
        # Delete old vectors for this file (important for modified files)
        delete_docs_by_source(client, COLLECTION_NAME, path)
        
        # Load and split document
        try:
            loaded = load_document(path)
            if loaded is None:
                print(f"    ⚠️ Unsupported file type, skipping")
                continue
            chunks = splitter.split_documents(loaded)
            all_docs.extend(chunks)
        except Exception as e:
            print(f"    ❌ Error loading: {e}")
            continue
    
    # Add to vector store
    if all_docs:
        ensure_collection_exists(client, COLLECTION_NAME, vector_size=768)
        
        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model
        )
        qdrant_store.add_documents(all_docs)
        print(f"\n✅ Added {len(all_docs)} chunks to database")
    else:
        print("\n⚠️ No documents were successfully loaded")
    
    # Save updated index (with current hashes)
    save_index(current_hashes)
    print("✅ Index file updated\n")
    
    return added, modified, deleted, unchanged


# ==========================================
# MAIN PROGRAM
# ==========================================
def main():
    # Create documents folder if needed
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"📁 Created folder '{DOCS_FOLDER}'. Add files and run again!")
        sys.exit()
    
    # Initialize embedding model
    print(f"🧠 Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedding_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url="http://localhost:11434"
    )
    
    # Connect to Qdrant
    print("🔌 Connecting to Qdrant...")
    client = QdrantClient(url="http://localhost:6333")
    ensure_collection_exists(client, COLLECTION_NAME, vector_size=768)
    
    # Check files and reindex if needed (uses MD5 comparison!)
    check_and_reindex(client, embedding_model)
    
    # Create retriever
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Initialize LLM
    print(f"🧠 Loading LLM model ({LLM_MODEL_NAME})...")
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    
    # RAG Prompt
    system_prompt = (
        "You are a helpful assistant. Answer the question using the provided context.\n"
        "If the answer is not in the context, say 'I don't know based on the documents.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    def format_docs_with_sources(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    # RAG Chain
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Chat loop
    print("\n" + "="*50)
    print("🚀 RAG SYSTEM READY")
    print("="*50)
    print(f"🔹 Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"🔹 LLM: {LLM_MODEL_NAME}")
    print(f"🔹 Collection: {COLLECTION_NAME}")
    print("="*50)
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("🙋 Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'izlaz']:
            print("Bye! 👋")
            break
        
        if not question:
            continue
        
        print("🤖 Thinking...")
        try:
            answer = rag_chain.invoke(question)
            if not answer or not answer.strip():
                answer = "I couldn't find a relevant answer in the documents."
            print(f"\n🤖 Answer:\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    check_services()
    main()