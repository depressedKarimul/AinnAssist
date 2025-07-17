import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# === Step 1: Load PDF Files ===
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print(f"[INFO] Loaded {len(documents)} documents.")

# === Step 2: Create Smart Chunks ===
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,               # More precise chunk size
        chunk_overlap=300,             # More overlap to preserve context
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    
    # Optional: Add section numbers or simple paragraph metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["paragraph"] = f"para_{i+1}"
    
    return chunks

text_chunks = create_chunks(documents)
print(f"[INFO] Created {len(text_chunks)} text chunks.")
print("-" * 60)
print(text_chunks[0].page_content[:1000])
print("-" * 60)

# === Step 3: Load Ollama Embedding Model ===
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"

def get_embedding_model():
    try:
        model = OllamaEmbeddings(model=OLLAMA_MODEL_NAME)
        print(f"[INFO] Using Ollama model: {OLLAMA_MODEL_NAME}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load Ollama model: {e}")
        exit(1)

embedding_model = get_embedding_model()

# === Step 4: Generate & Save FAISS Vector Store ===
DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

try:
    print("[INFO] Starting embedding and vector store creation... This may take a few minutes.")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"[SUCCESS] Vector store saved to: {DB_FAISS_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to embed documents or save vector store: {e}")
    exit(1)
