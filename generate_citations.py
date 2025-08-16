import os
import re
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader

# === Directory containing PDFs ===
DATA_PATH = "data/"

# === Citation Template ===
CITATION_TEMPLATE = "{source}, Page {page}"

def extract_metadata(file_path):
    loader = PDFPlumberLoader(file_path)
    doc = loader.load()[0]
    file_name = os.path.basename(file_path)
    
    title_match = re.match(r"(.+?), (\d{4})\.pdf", file_name)
    title = title_match.group(1) if title_match else file_name.replace(".pdf", "")
    year = title_match.group(2) if title_match else "Unknown"
    
    content = doc.page_content.lower()
    act_match = re.search(r"act no\.?\s*([ivxlc]+)\s*of\s*(\d{4})", content, re.IGNORECASE)
    act_number = f"Act No. {act_match.group(1).upper()} of {act_match.group(2)}" if act_match else "Unknown"
    
    url_id = "Unknown"
    if "penal code" in file_name.lower():
        url_id = "11"
    elif "constitution" in file_name.lower():
        url_id = "12"
    
    page = doc.metadata.get("page", 1)
    
    return {
        "source": file_name,
        "page": page,
        "title": title,
        "year": year,
        "act_number": act_number,
        "url_id": url_id
    }

def generate_citations(data_path):
    citations = []
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PDFPlumberLoader)
    documents = loader.load()
    
    for doc in documents:
        file_path = doc.metadata.get("source", "")
        metadata = extract_metadata(file_path)
        citation = CITATION_TEMPLATE.format(**metadata)
        citations.append(citation)
        print(f"[DEBUG] File: {metadata['source']}, Title: {metadata['title']}, Year: {metadata['year']}, Act: {metadata['act_number']}, URL ID: {metadata['url_id']}")
    
    return citations

def main():
    citations = generate_citations(DATA_PATH)
    print("[INFO] Generated Citations:")
    for i, citation in enumerate(citations, 1):
        print(f"{i}. {citation}")

if __name__ == "__main__":
    main()


