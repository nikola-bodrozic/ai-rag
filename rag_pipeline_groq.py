import os
import config_utils as cu
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    try:
        # Validate API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ GROQ_API_KEY not found in .env file")
            print("Please add GROQ_API_KEY=your_key to .env")
            return
        
        client = QdrantClient(url="http://localhost:6333")
        embeddings = OllamaEmbeddings(model=cu.EMBEDDING_MODEL)
        
        # Ensure collection exists
        if not cu.ensure_collection(client):
            print("❌ Failed to ensure collection exists. Exiting.")
            return
        
        # Sync documents (now with batching and error handling)
        cu.sync_documents(client, embeddings)
        
        # Create vector store once and reuse it (Fix #10)
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=cu.COLLECTION_NAME, 
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile"
        )
        
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use the following context to answer the question.\nContext: {context}\n\nQuestion: {question}"
        )
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        print("\n✅ GROQ-POWERED RAG READY (Shared Collection)")
        print(f"📚 Document folder: {cu.DOCS_FOLDER}")
        print(f"💾 Index file: {cu.INDEX_FILE}")
        print("Type 'exit' or 'quit' to stop\n")
        
        while True:
            try:
                question = input("🙋 Question: ").strip()
                if question.lower() in ['exit', 'quit']:
                    break
                if not question:
                    print("⚠️ Please enter a question.")
                    continue
                
                print("💭 Thinking...")
                for chunk in rag_chain.stream(question):
                    print(chunk, end="", flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return

if __name__ == "__main__":
    main()