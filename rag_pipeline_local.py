import sys
import os
import hashlib
import json
from typing import List, Tuple

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
            try:
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
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")
                continue

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

# Create vector store and retriever
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
# 4. CHAT MEMORY IMPLEMENTATION (PURE PYTHON - NO LANGCHAIN)
# ==========================================

class ChatMemory:
    """Simple RAM-based chat memory - no LangChain dependencies"""
    def __init__(self, max_history: int = 10):
        self.history: List[Tuple[str, str]] = []  # List of (question, answer)
        self.max_history = max_history
    
    def add_interaction(self, question: str, answer: str):
        """Add a new interaction to memory"""
        self.history.append((question, answer))
        # Keep only last N interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """Format history as context string"""
        if not self.history:
            return ""
        
        context_lines = ["Previous conversation:"]
        for i, (q, a) in enumerate(self.history, 1):
            context_lines.append(f"Q{i}: {q}")
            context_lines.append(f"A{i}: {a}")
        return "\n".join(context_lines)
    
    def clear(self):
        """Clear memory"""
        self.history = []
    
    def get_last_n(self, n: int) -> str:
        """Get last N interactions"""
        if not self.history:
            return ""
        
        recent = self.history[-n:] if len(self.history) > n else self.history
        context_lines = ["Previous conversation (recent):"]
        for i, (q, a) in enumerate(recent, 1):
            context_lines.append(f"Q: {q}")
            context_lines.append(f"A: {a}")
        return "\n".join(context_lines)

# Initialize chat memory
chat_memory = ChatMemory(max_history=10)

# ==========================================
# 5. RAG PROMPT WITH MEMORY
# ==========================================

def format_docs_with_sources(docs):
    """Format documents with source information"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        # Get page number if available
        page = doc.metadata.get('page', '')
        if page:
            source = f"{source} (page {page})"
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def create_rag_chain_with_memory(retriever, llm, chat_memory):
    """Create RAG chain with memory integration"""
    
    # Enhanced prompt with memory awareness
    system_prompt = (
        "You are a helpful assistant with access to previous conversations and document context.\n"
        "Follow these guidelines:\n"
        "1. Answer based on the provided relevant documents whenever possible.\n"
        "2. Use the conversation history to maintain context and answer follow-up questions.\n"
        "3. If the answer is not in the documents, say 'I don't know based on the available documents.'\n"
        "4. Be specific and include relevant details, dates, or requirements from the context.\n"
        "5. When referencing previous questions, use the conversation history.\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "Relevant Documents:\n{docs_context}\n\n"
        "Current Question: {question}\n\n"
        "Answer:"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Create the chain with memory
    def prepare_inputs(question):
        """Prepare all inputs for the chain"""
        # Get chat history
        chat_history = chat_memory.get_context()
        
        # Get relevant documents
        docs = retriever.invoke(question)
        docs_context = format_docs_with_sources(docs) if docs else "No relevant documents found."
        
        return {
            "question": question,
            "chat_history": chat_history or "No previous conversation.",
            "docs_context": docs_context
        }
    
    # Create the chain
    rag_chain = (
        RunnablePassthrough() | prepare_inputs
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Create the chain with memory
rag_chain = create_rag_chain_with_memory(retriever, llm, chat_memory)

# ==========================================
# 6. INFERENCE WITH MEMORY
# ==========================================

print("\n🚀 RAG system with chat memory is ready!")
print("="*50)
print(f"🔹 Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"🔹 LLM model: {LLM_MODEL_NAME}")
print(f"🔹 Collection: {COLLECTION_NAME}")
print(f"🔹 Memory size: {chat_memory.max_history} interactions")
print("="*50)
print("Commands:")
print("  'exit' - End the program")
print("  'clear' - Clear conversation memory")
print("  'history' - Show conversation history")
print("="*50 + "\n")

while True:
    question = input("🙋 Ask question: ")
    
    if question.lower() in ['izlaz', 'exit', 'quit']:
        print("Bye!")
        break
    
    if question.lower() == 'clear':
        chat_memory.clear()
        print("🗑️  Conversation memory cleared!\n")
        continue
    
    if question.lower() == 'history':
        print("\n📜 Conversation History:")
        print("-" * 40)
        history = chat_memory.get_context()
        if history:
            print(history)
        else:
            print("No conversation history yet.")
        print("-" * 40 + "\n")
        continue

    if not question.strip():
        continue

    print("🤖 I'm thinking... (with memory of previous conversations)")
    try:
        # Invoke the chain with the question
        answer = rag_chain.invoke(question)
        
        if not answer or answer.strip() == "":
            answer = "I couldn't find a relevant answer in the documents."
        
        # Store interaction in memory
        chat_memory.add_interaction(question, answer)
        
        print(f"\n🤖 Answer:\n{answer}\n")
        print(f"💾 Memory: {len(chat_memory.history)} stored interactions\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ==========================================
# 7. OPTIONAL: Persistent Memory (save to disk)
# ==========================================

class PersistentChatMemory(ChatMemory):
    """Chat memory with disk persistence"""
    
    def __init__(self, memory_file="chat_memory.json", max_history=10):
        super().__init__(max_history)
        self.memory_file = memory_file
        self.load_memory()
    
    def save_memory(self):
        """Save memory to disk"""
        try:
            data = [{"question": q, "answer": a} for q, a in self.history]
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Could not save memory: {e}")
    
    def load_memory(self):
        """Load memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = [(item["question"], item["answer"]) for item in data]
                    # Trim if too long
                    if len(self.history) > self.max_history:
                        self.history = self.history[-self.max_history:]
        except Exception as e:
            print(f"⚠️ Could not load memory: {e}")
    
    def add_interaction(self, question: str, answer: str):
        """Add interaction and save to disk"""
        super().add_interaction(question, answer)
        self.save_memory()
    
    def clear(self):
        """Clear memory and remove file"""
        super().clear()
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
