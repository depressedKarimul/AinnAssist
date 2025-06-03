from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS



# Step 1:clear(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print("Length of PDF pages: ", len(documents))



# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)
ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model():
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

embedding_model = get_embedding_model()
