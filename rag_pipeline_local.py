import sys
import os
import hashlib
import json

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
            
            # ✅ NEW: Use QdrantVectorStore
            qdrant_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embedding_model
            )
            qdrant_store.add_documents(docs)
            print(f"✅ {len(docs)} chunks is added in database")
        else:
            print("⚠️ No documents for indexing")

        save_index(new_index)
        print("✅ Reindexing is finished.")
    else:
        print("✅ No changes in files.")

# ==========================================
# 1. PROVERA FOLDERA
# ==========================================
if not os.path.exists(DOCS_FOLDER):
    os.makedirs(DOCS_FOLDER)
    print(f"📁 Created folder '{DOCS_FOLDER}'. Add files and run the script again!")
    sys.exit()

# ==========================================
# 2. EMBEDDINGS - LOCAL OLLAMA
# ==========================================
print(f"🧠 Starting embedding model ({EMBEDDING_MODEL_NAME} - LOCAL)...")
embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    base_url="http://localhost:11434"
)

print("🔌 Connecting to the Qdrant database...")
client = QdrantClient(url="http://localhost:6333")
ensure_collection_exists(client, COLLECTION_NAME, vector_size=768)

if os.path.exists(INDEX_FILE):
    os.remove(INDEX_FILE)
    print("🗑️  Old index file removed - forcing full reindex!")

check_and_reindex(client, embedding_model)

# ✅ NEW: Create retriever using QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 3. LLM - LOCAL OLLAMA
# ==========================================
print(f"🧠 Starting LLM model ({LLM_MODEL_NAME} - LOCAL)...")
llm = OllamaLLM(model=LLM_MODEL_NAME)

# ==========================================
# 4. RAG PROMPT & CHAIN
# ==========================================
system_prompt = (
    "You are a helpful assistant. Answer the question exclusively using the provided context below.\n"
    "If the answer is not in the context, say 'I don't know the answer to that question based on internal documents.'\n"
    "When answering, be specific and include relevant details, dates, or requirements from the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# Then use this in your chain
rag_chain = (
    {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. INFERENCE
# ==========================================
print("\n🚀 RAG system is ready (LOCAL version)!")
print("="*50)
print(f"🔹 Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"🔹 LLM model: {LLM_MODEL_NAME}")
print(f"🔹 Collection: {COLLECTION_NAME}")
print("="*50)
print("Type 'exit' to end the program.\n")

while True:
    question = input("🙋 Ask question: ")
    if question.lower() in ['izlaz', 'exit', 'quit']:
        print("Bye!")
        break

    if not question.strip():
        continue

    print("🤖 I'm thinking...")
    try:
        anwser = rag_chain.invoke(question)
        if not anwser or anwser.strip() == "":
            anwser = "I couldn't find a relevant answer in the documents."
        print(f"\n🤖 Answer:\n{anwser}\n")
    except Exception as e:
        print(f"❌ Error: {e}")