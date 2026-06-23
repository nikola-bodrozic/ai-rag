import sys
import os

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Putanja do foldera sa tvojim fajlovima
FOLDER_SA_DOKUMENTIMA = "./dokumentacija"

# Osiguravamo da folder postoji
if not os.path.exists(FOLDER_SA_DOKUMENTIMA):
    os.makedirs(FOLDER_SA_DOKUMENTIMA)
    print(f"📁 Napravljen je folder '{FOLDER_SA_DOKUMENTIMA}'. Ubaci u njega .txt ili .md fajlove pa pokreni skriptu ponovo!")
    sys.exit()

# ==========================================
# 1. UČITAVANJE I CEPCANJE (CHUNKING) FAJLOVA
# ==========================================
print("📂 Učitavam dokumente iz foldera...")

# DirectoryLoader automatski skenira folder za .txt i .md fajlove
loader_txt = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
loader_md = DirectoryLoader(FOLDER_SA_DOKUMENTIMA, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})

dokumenti = loader_txt.load() + loader_md.load()

if not dokumenti:
    print(f"⚠️ Folder '{FOLDER_SA_DOKUMENTIMA}' je prazan. Ubaci bar jedan .txt ili .md fajl pre pokretanja.")
    sys.exit()

print(f"📄 Učitano dokumenata: {len(dokumenti)}")

# Pošto dokumenti mogu biti ogromni (npr. cela knjiga ili dugačak kod),
# moramo ih iseći na manje delove (chunks) kako bi stali u kontekst LLM-a.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Svaki deo će imati oko 500 karaktera
    chunk_overlap=50,     # Preklapanje od 50 karaktera da se ne izgubi smisao na ivicama
    length_function=len
)
izrezani_delovi = text_splitter.split_documents(dokumenti)
print(f"✂️ Dokumenti su isečeni na {len(izrezani_delovi)} manjih delova (chunks).")

# ==========================================
# 2. VEKTORIZACIJA I BAZA
# ==========================================
print("🧠 Pokretanje embedding modela i indeksiranje podataka...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vektorska_baza = Chroma.from_documents(izrezani_delovi, embedding_model)
retriever = vektorska_baza.as_retriever(search_kwargs={"k": 3}) # Izvlači top 3 najrelevantnija dela

# ==========================================
# 3. POVEZIVANJE SA OLLAMOM
# ==========================================
llm = OllamaLLM(model="gemma2:2b")

# ==========================================
# 4. RAG PROMPT I LANAC
# ==========================================
sistemski_prompt = (
    "Ti si koristan asistent. Odgovori na pitanje isključivo koristeći priloženi kontekst ispod.\n"
    "Ako u kontekstu nema odgovora, reci 'Ne znam odgovor na to pitanje na osnovu internih dokumenata.'\n\n"
    "Kontekst:\n{context}\n\n"
    "Pitanje: {question}"
)

prompt = ChatPromptTemplate.from_template(sistemski_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_lanac = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. INTERAKTIVNI RAD (Petlja za pitanja)
# ==========================================
print("\n🚀 RAG sistem je spreman za tvoje fajlove!")
print("Upisi 'izlaz' za kraj programa.\n")

while True:
    pitanje = input("🙋 Postavi pitanje: ")
    if pitanje.lower() in ['izlaz', 'exit', 'quit']:
        print("Doviđenja!")
        break
        
    if not pitanje.strip():
        continue

    print("🤖 Razmišljam...")
    odgovor = rag_lanac.invoke(pitanje)
    print(f"\n🤖 Odgovor:\n{odgovor}\n")
    print("-" * 50)