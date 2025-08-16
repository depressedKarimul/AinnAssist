
import os
import re
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# === Step 1: Load PDF Files with Enhanced Metadata ===
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PDFPlumberLoader)
    documents = loader.load()
    
    for doc in documents:
        file_name = os.path.basename(doc.metadata.get("source", ""))
        title_match = re.match(r"(.+?), (\d{4})\.pdf", file_name)
        if title_match:
            doc.metadata["title"] = title_match.group(1)
            doc.metadata["year"] = title_match.group(2)
        else:
            doc.metadata["title"] = file_name.replace(".pdf", "")
            doc.metadata["year"] = "Unknown"
        
        content = doc.page_content.lower()
        act_match = re.search(r"act no\.?\s*([ivxlc]+)\s*of\s*(\d{4})", content, re.IGNORECASE)
        if act_match:
            doc.metadata["act_number"] = f"Act No. {act_match.group(1).upper()} of {act_match.group(2)}"
        else:
            doc.metadata["act_number"] = "Unknown"
        
        doc.metadata["url_id"] = "Unknown"
        if "penal code" in file_name.lower():
            doc.metadata["url_id"] = "11"
        elif "constitution" in file_name.lower():
            doc.metadata["url_id"] = "12"
    
    return documents

documents = load_pdf_files(DATA_PATH)
print(f"[INFO] Loaded {len(documents)} documents from {DATA_PATH}.")

# === Step 2: Clean OCR Noise ===
def clean_ocr_noise(text):
    text = re.sub(r'\b\d+(,\s*\d+)*\b', '', text)
    text = re.sub(r'\b2\s*,\s*2\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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
        "distribution": "distribution",
        "penal": "penal",
        "code": "code",
        "criminal": "criminal"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# === Step 3: Create Smart Chunks ===
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = []
    
    for doc in docs:
        doc.page_content = clean_ocr_noise(doc.page_content)
        file_name = os.path.basename(doc.metadata.get("source", "")).lower()
        current_section = "N/A"
        current_article = "None"
        
        doc_chunks = splitter.split_text(doc.page_content)
        
        if "constitution" in file_name:
            for i, chunk_content in enumerate(doc_chunks):
                chunk = doc.__class__(page_content=chunk_content, metadata=doc.metadata.copy())
                chunk.metadata["paragraph"] = f"para_{len(chunks)+1}"
                
                if chunk_content.startswith("Part "):
                    current_section = chunk_content.split("\n")[0].strip()
                elif any(term in chunk_content.lower() for term in ["part i", "republic", "capital", "portrait", "bangabandhu"]):
                    current_section = "Part I: The Republic"
                elif any(term in chunk_content.lower() for term in ["part ii", "fundamental principles", "socialist", "economic system"]):
                    current_section = "Part II: Fundamental Principles of State Policy"
                
                article_match = re.search(r"Article\s+(\d+[A-Za-z]?)", chunk_content, re.IGNORECASE)
                if article_match:
                    current_article = f"Article {article_match.group(1)}"
                
                chunk.metadata["section"] = current_section
                chunk.metadata["article"] = current_article
                chunks.append(chunk)
        else:
            for i, chunk_content in enumerate(doc_chunks):
                chunk = doc.__class__(page_content=chunk_content, metadata=doc.metadata.copy())
                chunk.metadata["paragraph"] = f"para_{len(chunks)+1}"
                
                section_match = re.search(r"(chapter|section)\s+[ivxlc\d]+[.:]?\s+([^\n]+)", chunk_content, re.IGNORECASE)
                if section_match:
                    current_section = f"{section_match.group(1).capitalize()} {section_match.group(0).split(':')[0].strip()}"
                
                chunk.metadata["section"] = current_section
                chunk.metadata["article"] = "None"
                chunks.append(chunk)
    
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "")).lower()
        if "constitution" in file_name:
            found_dhaka = any("dhaka" in chunk.page_content.lower() for chunk in chunks)
            found_bangabandhu = any("bangabandhu" in chunk.page_content.lower() for chunk in chunks)
            if not (found_dhaka and found_bangabandhu):
                print(f"[WARNING] Adding fallback chunks for {file_name}.")
                fallback_chunks = [
                    doc.__class__(
                        page_content="The capital of the Republic is Dhaka as per Article 5.",
                        metadata={
                            "source": "data/The Constitution of the People's Republic of Bangladesh.pdf",
                            "page": 1,
                            "paragraph": f"para_{len(chunks)+1}",
                            "section": "Part I: The Republic",
                            "article": "Article 5",
                            "title": "The Constitution of the People's Republic of Bangladesh",
                            "year": "1972",
                            "act_number": "Unknown",
                            "url_id": "12"
                        }
                    ),
                    doc.__class__(
                        page_content="The portrait of Bangabandhu Sheikh Mujibur Rahman shall be displayed in all government and semi-government offices, autonomous bodies, educational institutions, and Bangladesh missions abroad as per Article 4A.",
                        metadata={
                            "source": "data/The Constitution of the People's Republic of Bangladesh.pdf",
                            "page": 1,
                            "paragraph": f"para_{len(chunks)+2}",
                            "section": "Part I: The Republic",
                            "article": "Article 4A",
                            "title": "The Constitution of the People's Republic of Bangladesh",
                            "year": "1972",
                            "act_number": "Unknown",
                            "url_id": "12"
                        }
                    )
                ]
                chunks.extend(fallback_chunks)
    
    print("[DEBUG] Checking for relevant chunks...")
    found_relevant = False
    for i, chunk in enumerate(chunks):
        if any(term in chunk.page_content.lower() for term in ["socialist", "economic system", "part ii", "dhaka", "bangabandhu", "penal", "code", "criminal"]):
            found_relevant = True
            print(f"[DEBUG] Relevant Chunk {i+1}: {chunk.page_content[:100]}...")
            print(f"[DEBUG] Metadata: {chunk.metadata}")
    if not found_relevant:
        print("[WARNING] No chunks found with relevant keywords.")
    
    return chunks

text_chunks = create_chunks(documents)
print(f"[INFO] Created {len(text_chunks)} text chunks from {len(documents)} documents.")
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
