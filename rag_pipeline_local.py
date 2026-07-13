import config_utils as cu
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    client = QdrantClient(url="http://localhost:6333")
    embeddings = OllamaEmbeddings(model=cu.EMBEDDING_MODEL)
    
    cu.ensure_collection(client)
    cu.sync_documents(client, embeddings)
    
    retriever = QdrantVectorStore(client=client, collection_name=cu.COLLECTION_NAME, embedding=embeddings).as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model="gemma2:2b")
    
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer using this context:\n{context}\n\nQuestion: {question}")
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    print("\n🚀 LOCAL RAG SYSTEM READY (Shared Collection)")
    while True:
        question = input("🙋 Question: ").strip()
        if question.lower() in ['exit', 'quit']: break
        for chunk in rag_chain.stream(question): print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()