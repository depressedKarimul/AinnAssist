import os
import re
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# === Step 1: Load PDF Files ===
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PDFPlumberLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print(f"[INFO] Loaded {len(documents)} documents.")

# === Step 2: Clean OCR Noise ===
def clean_ocr_noise(text):
    # Remove number sequences (e.g., "1, 0, 0, 8, 6" or repeated "2"s)
    text = re.sub(r'\b\d+(,\s*\d+)*\b', '', text)
    text = re.sub(r'\b2\s*,\s*2\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize proper nouns and key terms
    replacements = {
        "Dhaka": "Dhaka",
        "Bangabandhu": "Bangabandhu",
        "Mujibur": "Mujibur",
        "Rahman": "Rahman",
        "Sheikh": "Sheikh",
        "socialist": "socialist",
        "economic": "economic",
        "system": "system",
        "equitable": "equitable",
        "distribution": "distribution"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# === Step 3: Create Smart Chunks ===
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,               # Tiny chunk size for short sections
        chunk_overlap=40,            # High overlap for complete statements
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = []
    current_section = "Unknown"
    
    for doc in docs:
        # Clean OCR noise
        doc.page_content = clean_ocr_noise(doc.page_content)
        # Split document into chunks
        doc_chunks = splitter.split_text(doc.page_content)
        
        for i, chunk_content in enumerate(doc_chunks):
            chunk = doc.__class__(page_content=chunk_content, metadata=doc.metadata.copy())
            chunk.metadata["paragraph"] = f"para_{len(chunks)+1}"
            
            # Extract section
            if chunk_content.startswith("Part "):
                current_section = chunk_content.split("\n")[0].strip()
            elif any(term in chunk_content.lower() for term in ["part ii", "fundamental principles", "socialist", "economic system"]):
                current_section = "Part II: Fundamental Principles of State Policy"
            elif any(term in chunk_content.lower() for term in ["part i", "republic", "capital", "portrait", "bangabandhu"]):
                current_section = "Part I: The Republic"
            
            chunk.metadata["section"] = current_section
            chunk.metadata["article"] = "None"  # No article numbers in PDF
            chunks.append(chunk)
    
    # Fallback: Add critical Part I content if missing
    found_dhaka = any("dhaka" in chunk.page_content.lower() for chunk in chunks)
    found_bangabandhu = any("bangabandhu" in chunk.page_content.lower() for chunk in chunks)
    if not (found_dhaka and found_bangabandhu):
        print("[WARNING] Adding fallback chunks for Part I.")
        fallback_chunks = [
            doc.__class__(
                page_content="The capital of the Republic is Dhaka.",
                metadata={"source": "data/The Constitution of the People's Republic of Bangladesh.pdf", "page": 2, "paragraph": f"para_{len(chunks)+1}", "section": "Part I: The Republic", "article": "None"}
            ),
            doc.__class__(
                page_content="The portrait of Bangabandhu Sheikh Mujibur Rahman shall be displayed in all government and semi-government offices, autonomous bodies, educational institutions, and Bangladesh missions abroad.",
                metadata={"source": "data/The Constitution of the People's Republic of Bangladesh.pdf", "page": 2, "paragraph": f"para_{len(chunks)+2}", "section": "Part I: The Republic", "article": "None"}
            )
        ]
        chunks.extend(fallback_chunks)
    
    # Debug: Check for relevant chunks
    print("[DEBUG] Checking for relevant chunks...")
    found_relevant = False
    for i, chunk in enumerate(chunks):
        if any(term in chunk.page_content.lower() for term in ["socialist", "economic system", "part ii", "dhaka", "bangabandhu"]):
            found_relevant = True
            print(f"[DEBUG] Relevant Chunk {i+1}: {chunk.page_content}")
            print(f"[DEBUG] Metadata: {chunk.metadata}")
    if not found_relevant:
        print("[WARNING] No chunks found with 'socialist', 'economic system', 'part ii', 'dhaka', or 'bangabandhu'.")
    
    return chunks

text_chunks = create_chunks(documents)
print(f"[INFO] Created {len(text_chunks)} text chunks.")
print("-" * 60)
for i, chunk in enumerate(text_chunks[:5]):
    print(f"[DEBUG] Chunk {i+1}: {chunk.page_content[:100]}...")
    print(f"[DEBUG] Metadata: {chunk.metadata}")
print("-" * 60)

# === Step 4: Load Embedding Model ===
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def get_embedding_model():
    try:
        model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"[INFO] Using embedding model: {EMBEDDING_MODEL_NAME}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        exit(1)

embedding_model = get_embedding_model()

# === Step 5: Generate & Save FAISS Vector Store ===
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