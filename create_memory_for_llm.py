import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# === Step 1: Load PDF Files ===
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print(f"[INFO] Loaded {len(documents)} documents.")

# === Step 2: Create Smart Chunks for Any Document ===
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],  # Paragraph > sentence > word fallback
    )
    chunks = splitter.split_documents(docs)
    return chunks

text_chunks = create_chunks(documents)
print(f"[INFO] Created {len(text_chunks)} text chunks.")
print("[INFO] Sample chunk previews:")
print("-" * 60)
for i in range(min(3, len(text_chunks))):
    print(f"[Chunk {i+1}]\n{text_chunks[i].page_content}")
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

def save_vector_store(chunks, embedding_model, path):
    try:
        print("[INFO] Creating FAISS vector store... This may take a few minutes.")
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(path)
        print(f"[SUCCESS] Vector store saved at: {path}")
    except Exception as e:
        print(f"[ERROR] Vector store creation failed: {e}")
        exit(1)

save_vector_store(text_chunks, embedding_model, DB_FAISS_PATH)
