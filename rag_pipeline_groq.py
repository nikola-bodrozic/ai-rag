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
# IMPORTS
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
DOCS_FOLDER = "./documents"
INDEX_FILE = "indexed_files_groq.json"
COLLECTION_NAME = "rag_collection_groq"

EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# ==========================================
# 0. MD5 UTIL FUNCTIONS
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

    for root, _, files in os.walk(DOCS_FOLDER):
        for file in files:
            path = os.path.join(root, file)
            file_hash = md5_file(path)
            new_index[path] = file_hash
            if path not in current_index or current_index[path] != file_hash:
                changed_files.append(path)

    if changed_files:
        print(f"🔄 Changed files: {changed_files}")
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
                print(f"⚠️ Skipping unsupported file: {path}")
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
            print(f"✅ {len(docs)} chunks added to database")
        else:
            print("⚠️ No documents for indexing")

        save_index(new_index)
        print("✅ Reindexing is finished.")
    else:
        print("✅ Files didn't change.")

# ==========================================
# 1. DOES FOLDER EXIST
# ==========================================
if not os.path.exists(DOCS_FOLDER):
    os.makedirs(DOCS_FOLDER)
    print(f"📁 Folder is created '{DOCS_FOLDER}'. Copy files and rerun the script")
    sys.exit()

# ==========================================
# 2. CREATING EMBEDDINGS
# ==========================================
print(f"🧠 Starting embedding model ({EMBEDDING_MODEL_NAME} - LOCAL)...")
embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    base_url="http://localhost:11434"
)

print("🔌 Connecting with Qdrant data base...")
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
print("🤖 GROQ API (CLOUD LLM)")
print("="*50)
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL_NAME,
    max_tokens=1024,
)

print(f"📊 Model: {LLM_MODEL_NAME}")
print(f"🌐 Type: Cloud API (Internet required)")
print(f"💰 Cost: Free (currently)")
print(f"⚡ Speed: Very fast (hardware-optimized)")
print("="*50 + "\n")

# ==========================================
# 4. RAG PROMPT & CHAIN
# ==========================================
system_prompt = (
    "You are a helpful assistant. Answer the question exclusively using the provided context below.\n"
    "If the answer is not in the context, say 'I don't know the answer to that question based on internal documents.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n🚀 RAG system is ready (GROQ version)!")
print("="*50)
print(f"🔹 Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"🔹 LLM model: {LLM_MODEL_NAME} (GROQ - CLOUD)")
print(f"🔹 Collection: {COLLECTION_NAME}")
print(f"🔹 API Key: {'✅ Found' if GROQ_API_KEY else '❌ Not found'}")
print("="*50)
print("Type 'exit' to end.\n")

while True:
    question = input("🙋 Ask question: ")
    if question.lower() in ['izlaz', 'exit', 'quit']:
        print("Bye!")
        break

    if not question.strip():
        continue

    print("🤖 I'm thinking (GROQ cloud)...")
    try:
        anwser = rag_chain.invoke(question)
        print(f"\n🤖 Anwser (GROQ):\n{anwser}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
    print("-" * 50)