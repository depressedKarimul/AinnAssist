
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# === Step 0: Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY is missing. Please set it in your .env file.")
    exit(1)

# === Step 1: Load the Vector Store ===
DB_FAISS_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"

def load_vector_store():
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_NAME)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("[INFO] Vector store loaded successfully.")
        return db
    except Exception as e:
        print(f"[ERROR] Failed to load vector store: {e}")
        exit(1)

faiss_db = load_vector_store()

# === Step 2: Setup LLM (Groq) ===
def load_llm():
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="deepseek-r1-distill-llama-70b")
        print("[INFO] Groq LLM loaded successfully.")
        return llm
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq LLM: {e}")
        exit(1)

llm_model = load_llm()

# === Step 3: Document Retrieval ===
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# === Step 4: Prompt and Response Generation ===
custom_prompt_template = """
Answer the question using only the information provided in the context.
❗ Your response must be a short summary — no more than **2 concise sentences**. Do NOT explain your reasoning.

Question: {question}
Context: {context}

Final Answer (Max 2 sentences):
"""


def answer_query(query):
    docs = retrieve_docs(query)
    context = get_context(docs)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    answer = chain.invoke({"question": query, "context": context})
    return answer

# === Optional: Interactive Run ===
if __name__ == "__main__":
    question = input("Ask a question: ")
    response = answer_query(question)
    print("\nAI Lawyer:", response)
