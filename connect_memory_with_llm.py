import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# === Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY is missing. Please set it in your .env file.")
    exit(1)

# === Vector DB and Embedding config ===
DB_FAISS_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"

def load_vector_store():
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_NAME)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("[INFO] ✅ Vector store loaded.")
        return db
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load vector store: {e}")
        exit(1)

faiss_db = load_vector_store()

# === Load LLM from Groq ===
def load_llm():
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="deepseek-r1-distill-llama-70b")
        print("[INFO] ✅ Groq LLM loaded.")
        return llm
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load LLM: {e}")
        exit(1)

llm_model = load_llm()

# === Retrieve ALL documents with similarity scores ===
def retrieve_all_docs(query):
    all_doc_count = len(faiss_db.index_to_docstore_id)
    docs_and_scores = faiss_db.similarity_search_with_score(query, k=all_doc_count)
    return docs_and_scores

# === Combine all content into a single context ===
def get_context(documents):
    return "\n\n".join([doc.page_content for doc, _ in documents])

# === Calculate confidence score ===
def calculate_confidence_score(docs_and_scores):
    if not docs_and_scores:
        return 0.0
    
    # Extract similarity scores (lower distance = higher similarity)
    scores = [float(1 - score) for _, score in docs_and_scores]  # Convert to float
    
    # Average similarity score (weight: 70%)
    avg_similarity = sum(scores) / len(scores) if scores else 0.0
    
    # Metadata quality (weight: 30%)
    metadata_quality = 0.0
    for doc, _ in docs_and_scores:
        meta = doc.metadata
        # Check if metadata fields are present and non-empty
        source_present = meta.get("source", "Unknown") != "Unknown"
        page_present = meta.get("page", "?") != "?"
        para_present = meta.get("paragraph", "?") != "?"
        # Add points for each present field
        metadata_score = sum([source_present, page_present, para_present]) / 3.0
        metadata_quality += metadata_score
    metadata_quality = metadata_quality / len(docs_and_scores) if docs_and_scores else 0.0
    
    # Combine scores: 70% similarity, 30% metadata quality
    confidence = (0.7 * avg_similarity + 0.3 * metadata_quality) * 10  # Scale to 0-10
    return float(round(confidence, 1))  # Ensure Python float

# === Strict Prompt Template ===
custom_prompt_template = """
You are a legal assistant. Answer the following question using ONLY the information provided in the context.
Do not make assumptions, do not guess, and do not include anything beyond the context.
Use all relevant parts. If the answer cannot be found in the context, say clearly: "The answer is not available in the provided context."

Question: {question}
Context:
{context}

Answer:
"""

# === Main function to answer query ===
def answer_query(query):
    docs_and_scores = retrieve_all_docs(query)

    if not docs_and_scores:
        return {
            "answer": "❌ Sorry, no relevant information found.",
            "confidence": 0.0
        }

    context = get_context(docs_and_scores)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    answer = chain.invoke({"question": query, "context": context})

    # Format source metadata
    sources = []
    for i, (doc, score) in enumerate(docs_and_scores):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        para = meta.get("paragraph", "?")
        sources.append(f"📄 Source {i+1}: {source} | Page {page} | Paragraph {para}")

    sources_info = "\n".join(sources)
    confidence_score = calculate_confidence_score(docs_and_scores)
    
    return {
        "answer": f"{answer.content.strip()}\n\n---\n📚 Source Info:\n{sources_info}",
        "confidence": confidence_score
    }

# === CLI test runner ===
if __name__ == "__main__":
    question = input("Ask a legal question: ").strip()
    if question:
        print("\n🤖 AinnAssist is thinking...\n")
        result = answer_query(question)
        print(f"Answer:\n{result['answer']}\n\nConfidence: {result['confidence']}/10")