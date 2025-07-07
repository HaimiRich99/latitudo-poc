import os
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec


# Set your Pinecone index name
PINECONE_INDEX_NAME = "llama-rag-index"
PINECONE_API_KEY = "pcsk_6EaKWM_KZFEZQz97UgaRaRux3pMMGxqyEVq7gukSd6hY8FUS889J7Bn6qpWBrxyHmf1mAZ"

# Load text files in parallel
def load_all_docs_parallel(doc_path="satellite_docs"):
    files = [f for f in os.listdir(doc_path) if f.endswith(".txt")]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda f: TextLoader(os.path.join(doc_path, f)).load(), files))
    docs = [doc for sublist in results for doc in sublist]
    print(f"Loaded {len(docs)} documents.")
    return docs

# Split text into chunks
def split_documents(docs, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# Create Pinecone index and upload vectors
def create_vector_db(doc_path="satellite_docs"):
    start = time.time()

    print("🔍 Loading documents...")
    docs = load_all_docs_parallel(doc_path)

    print("📄 Splitting documents...")
    chunks = split_documents(docs)

    print("🧠 Creating embeddings with llama3.1...")
    embed = OllamaEmbeddings(model="llama3.1")

    print("🌐 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    pc.delete_index(PINECONE_INDEX_NAME)

    if PINECONE_INDEX_NAME not in pc.list_indexes():
        print("📦 Creating Pinecone index...")
        pc.create_index(name=PINECONE_INDEX_NAME,
                              dimension=embed.embed_query("test").__len__(),
                              spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            ))

    print("🚀 Uploading vectors to Pinecone...")
    db = LangchainPinecone.from_documents(chunks, embed,
                                          index_name=PINECONE_INDEX_NAME)

    print(f"✅ Done in {time.time() - start:.2f} seconds")
    print(f"Documents: {len(docs)}, Chunks: {len(chunks)}")

# Load retriever from Pinecone
def load_retriever():
    print("🌐 Initializing Pinecone connection...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    print("📥 Loading retriever from Pinecone...")
    embed = OllamaEmbeddings(model="llama3.1")
    db = LangchainPinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embed)

    print("✅ Retriever ready.")
    return db.as_retriever(k=3)

# Uncomment below to create DB or load retriever
#create_vector_db()
#retriever = load_retriever()
