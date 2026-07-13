import os
import sys
import hashlib
import json
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration
DOCS_FOLDER = "./documents"
COLLECTION_NAME = "rag_collection_groq"
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"
LLM_MODEL_NAME = "llama-3.3-70b-versatile" # Or gemma2-9b-it

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    sys.exit(1)

def main():
    # 1. Initialize Models
    print(f"🧠 Initializing Local Embeddings ({EMBEDDING_MODEL_NAME})...")
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    print("🔌 Connecting to Qdrant...")
    client = QdrantClient(url="http://localhost:6333")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Initialize Groq LLM
    print(f"🚀 Initializing Groq LLM ({LLM_MODEL_NAME})...")
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME
    )

    # 3. Setup RAG Chain
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n"
        "Context: {context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Chat Loop
    print("\n✅ RAG SYSTEM READY (Groq Powered)")
    while True:
        question = input("\n🙋 Question: ").strip()
        if question.lower() in ['exit', 'quit']: break
        
        print("🤖 Answer: ", end="", flush=True)
        try:
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()