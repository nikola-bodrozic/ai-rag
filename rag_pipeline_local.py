import config_utils as cu
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    try:
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
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        llm = OllamaLLM(model="gemma2:2b")
        
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer using this context:\n{context}\n\nQuestion: {question}"
        )
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        print("\n🚀 LOCAL RAG SYSTEM READY (Shared Collection)")
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