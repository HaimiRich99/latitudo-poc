import os
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_all_docs_parallel(doc_path="satellite_docs"):
    files = [f for f in os.listdir(doc_path) if f.endswith(".txt")]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda f: TextLoader(os.path.join(doc_path, f)).load(), files))
    docs = [doc for sublist in results for doc in sublist]  # Flatten list
    print(f"Loaded {len(docs)} documents.")
    return docs

def split_documents(docs, chunk_size=256, overlap=32):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def create_vector_db(index_path="satellite_docs/vector_db/faiss_index", doc_path="satellite_docs"):
    start = time.time()
    
    print("🔍 Loading documents...")
    docs = load_all_docs_parallel(doc_path)
    
    print("📄 Splitting documents...")
    chunks = split_documents(docs)

    print("🧠 Creating embeddings...")
    embed = OllamaEmbeddings(model="llama3.1")

    print("📦 Building FAISS index...")
    db = FAISS.from_documents(chunks, embed)
    db.save_local(index_path)
    
    print(f"✅ Done in {time.time() - start:.2f} seconds")
    print(f"Documents: {len(docs)}, Chunks: {len(chunks)}, Vectors: {db.index.ntotal}, Dimension: {db.index.d}")

def load_retriever(index_path="satellite_docs/vector_db/faiss_index"):
    if not os.path.exists(index_path):
        print("❗ Index not found. Creating it...")
        create_vector_db(index_path)
    
    print("📥 Loading vector database...")
    db = FAISS.load_local(
        index_path,
        OllamaEmbeddings(model="llama3.1"),
        allow_dangerous_deserialization=True
    )
    print("✅ Vector database loaded.")
    return db.as_retriever(k=3)

#create_vector_db()
#load_retriever()