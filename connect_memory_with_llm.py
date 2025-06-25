import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# === Step 0: Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY is missing in .env.")
    exit(1)

if not SERPER_API_KEY:
    print("[ERROR] SERPER_API_KEY is missing in .env.")
    exit(1)

# === Step 1: Load FAISS vector store ===
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

# === Step 2: Load Groq LLM ===
def load_llm():
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="deepseek-r1-distill-llama-70b")
        print("[INFO] Groq LLM loaded successfully.")
        return llm
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq LLM: {e}")
        exit(1)

llm_model = load_llm()

# === Step 3: Retrieve documents from vector DB ===
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# === Step 4: Prompt Template ===
custom_prompt_template = """
Answer the question using only the information provided in the context.
❗ Do NOT explain your reasoning. Do NOT include any source links.

Question: {question}
Context: {context}

Final Answer:
"""

# === Step 5: Web search using Serper.dev ===
def web_search_fallback(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"q": query}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()

        snippets = []
        for result in results.get("organic", [])[:3]:
            snippet = result.get("snippet", "")
            if snippet:
                snippets.append(snippet)

        return "\n\n".join(snippets)
    except Exception as e:
        print(f"[ERROR] Serper.dev failed: {e}")
        return ""

# === Step 6: Main Answer Function ===
def answer_query(query):
    docs = retrieve_docs(query)

    if docs:
        print("[INFO] Answering from vector database...")
        context = get_context(docs)
        prompt = ChatPromptTemplate.from_template(custom_prompt_template)
        chain = prompt | llm_model
        answer = chain.invoke({"question": query, "context": context})
        answer_text = answer.content.strip()

        # Detect vague or unhelpful vector-based answer
        vague_phrases = [
            "provide",
            "context does not provide",
            "no relevant information",
            "not mentioned in the context",
            "unable to find information",
            "context doesn't mention",
            "context does not contain"
        ]

        if any(phrase in answer_text.lower() for phrase in vague_phrases):
            print("[INFO] Vector DB answer too vague. Trying Serper.dev...")
            web_context = web_search_fallback(query)

            if not web_context:
                return answer_text + "\n\n[⚠️ Could not find better info online.]"

            prompt = ChatPromptTemplate.from_template(custom_prompt_template)
            chain = prompt | llm_model
            web_answer = chain.invoke({"question": query, "context": web_context})
            return web_answer.content.strip()

        return answer_text

    else:
        print("[INFO] No documents found. Using Serper.dev for fallback...")
        web_context = web_search_fallback(query)

        if not web_context:
            return "Sorry, I couldn't find any relevant information online either."

        prompt = ChatPromptTemplate.from_template(custom_prompt_template)
        chain = prompt | llm_model
        answer = chain.invoke({"question": query, "context": web_context})
        return answer.content.strip()

# === Step 7: CLI Test ===
if __name__ == "__main__":
    print("🔍 Ask your legal questions! Type 'exit' to quit.")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        response = answer_query(question)
        print("\n🤖 AI Answer:\n", response, "\n")