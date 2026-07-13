import os, config_utils as cu
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
    client = QdrantClient(url="http://localhost:6333")
    embeddings = OllamaEmbeddings(model=cu.EMBEDDING_MODEL)
    
    cu.ensure_collection(client)
    cu.sync_documents(client, embeddings)
    
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
    retriever = QdrantVectorStore(client=client, collection_name=cu.COLLECTION_NAME, embedding=embeddings).as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Use the following context to answer the question.\nContext: {context}\n\nQuestion: {question}")
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    print("\n✅ GROQ-POWERED RAG READY (Shared Collection)")
    while True:
        question = input("🙋 Question: ").strip()
        if question.lower() in ['exit', 'quit']: break
        for chunk in rag_chain.stream(question): print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()