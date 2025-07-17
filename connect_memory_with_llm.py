import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
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
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_vector_store():
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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

# === Context Tracking ===
conversation_context = {
    "topic": None,
    "pdf": None,
    "last_question": None,
    "last_answer": None
}

def update_context(query, pdf="data/The Constitution of the People's Republic of Bangladesh.pdf"):
    global conversation_context
    if any(term in query.lower() for term in ["constitution", "bangladesh", "economic", "socialist", "part ii", "capital", "portrait", "bangabandhu"]):
        conversation_context["topic"] = "Constitution of Bangladesh"
        conversation_context["pdf"] = pdf
        conversation_context["last_question"] = query
    elif any(term in query.lower() for term in ["explain", "clearly", "more", "detail"]):
        # Retain last context for clarification
        pass
    else:
        # Assume Constitution context for follow-ups
        conversation_context["topic"] = "Constitution of Bangladesh"
        conversation_context["pdf"] = pdf
        conversation_context["last_question"] = query
    print(f"[DEBUG] Current context: {conversation_context}")
    return conversation_context

# === Keyword-based pre-filtering ===
def keyword_boost(query, doc_content):
    query_keywords = set(re.findall(r'\w+', query.lower()))
    constitution_keywords = {
        "socialist", "economic", "system", "equitable", "distribution", "wealth",
        "part ii", "fundamental", "principles", "state", "policy",
        "capital", "dhaka", "bangabandhu", "mujibur", "rahman", "portrait",
        "government", "offices", "institutions", "missions", "sheikh", "display",
        "constitution", "law", "public"
    }
    query_keywords.update(constitution_keywords)
    doc_keywords = set(re.findall(r'\w+', doc_content.lower()))
    matches = len(query_keywords.intersection(doc_keywords))
    return min(matches / len(query_keywords), 1.0) if query_keywords else 0.0

# === Retrieve top relevant documents with similarity scores ===
def retrieve_relevant_docs(query, k=60, relevance_threshold=0.1):
    global conversation_context
    # Preprocess query based on context
    if conversation_context["topic"] == "Constitution of Bangladesh":
        if any(term in query.lower() for term in ["economic", "socialist", "economy"]):
            query += " Part II Fundamental Principles of State Policy"
        if "capital" in query.lower():
            query += " Part I"
        if any(term in query.lower() for term in ["portrait", "bangabandhu", "display"]):
            query += " Part I"
        if any(term in query.lower() for term in ["explain", "clearly", "more", "detail"]):
            query = conversation_context["last_question"]  # Reuse last question
    
    docs_and_scores = faiss_db.similarity_search_with_score(query, k=k)
    
    # Re-rank based on keyword matches and section relevance
    reranked_docs = []
    for doc, score in docs_and_scores:
        keyword_score = keyword_boost(query, doc.page_content)
        section_score = 0.8 if doc.metadata.get("section", "").startswith("Part II") and "economic" in query.lower() else 0.7 if doc.metadata.get("section", "").startswith("Part I") else 0.0
        similarity_score = max(0.0, min(1.0, 1 - score))  # Normalize to [0, 1]
        combined_score = 0.05 * similarity_score + 0.8 * keyword_score + 0.15 * section_score
        reranked_docs.append((doc, score, combined_score))
    
    # Sort by combined score and filter by threshold
    reranked_docs.sort(key=lambda x: x[2], reverse=True)
    filtered_docs = [(doc, score) for doc, score, combined in reranked_docs if combined >= relevance_threshold]
    
    # Debug logging
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Retrieved {len(docs_and_scores)} docs, filtered to {len(filtered_docs)} with threshold {relevance_threshold}")
    for i, (doc, score, combined) in enumerate(reranked_docs[:5]):
        print(f"[DEBUG] Doc {i+1}: Page {doc.metadata.get('page', '?')} | Para {doc.metadata.get('paragraph', '?')} | Section {doc.metadata.get('section', '?')} | Article {doc.metadata.get('article', 'None')} | Similarity: {similarity_score:.2f} | Keyword: {keyword_score:.2f} | Combined: {combined:.2f}")
        print(f"[DEBUG] Content: {doc.page_content}")
    
    return filtered_docs if filtered_docs else [(docs_and_scores[0][0], docs_and_scores[0][1])]

# === Combine content from relevant documents ===
def get_context(documents):
    return "\n\n".join([doc.page_content for doc, _ in documents])

# === Calculate confidence score ===
def calculate_confidence_score(docs_and_scores):
    if not docs_and_scores:
        return 0.0
    
    top_doc, top_score = docs_and_scores[0]
    similarity_score = max(0.0, min(1.0, 1 - top_score))  # Normalize to [0, 1]
    keyword_score = keyword_boost("socialist economic system equitable distribution wealth part ii fundamental principles state policy capital dhaka bangabandhu mujibur rahman portrait government offices institutions missions sheikh display constitution law public", top_doc.page_content)
    section_score = 0.8 if top_doc.metadata.get("section", "").startswith("Part II") else 0.7 if top_doc.metadata.get("section", "").startswith("Part I") else 0.0
    
    # Metadata quality
    meta = top_doc.metadata
    source_present = meta.get("source", "Unknown") != "Unknown"
    page_present = meta.get("page", "?") != "?"
    para_present = meta.get("paragraph", "?") != "?"
    section_present = meta.get("section", "Unknown") != "Unknown"
    metadata_score = sum([source_present, page_present, para_present, section_present]) / 4.0
    
    # Confidence: 5% similarity, 80% keyword, 10% section, 5% metadata
    confidence = (0.05 * similarity_score + 0.8 * keyword_score + 0.1 * section_score + 0.05 * metadata_score) * 10
    confidence = max(0.0, min(9.0, confidence))  # Cap at 9.0
    return float(round(confidence, 1))

# === Refined Prompt Template ===
custom_prompt_template = """
You are a legal assistant specializing in the Constitution of Bangladesh. Answer the question using ONLY the provided context, prioritizing explicit statements from the relevant section (e.g., Part II: Fundamental Principles of State Policy for economic system queries). 
Do NOT include article numbers in the answer unless explicitly present in the context (e.g., "Article 5"). 
Provide a concise and complete answer, avoiding partial or incomplete responses. 
If the question asks for clarification (e.g., "explain more clearly"), rephrase the previous answer with more detail, maintaining accuracy and context.
Assume the question relates to the Constitution of Bangladesh unless otherwise specified.
If the answer is not found in the context, state: "The answer is not available in the provided context."

Previous Question: {last_question}
Question: {question}
Context:
{context}

Answer:
"""

# === Main function to answer query ===
def answer_query(query):
    global conversation_context
    # Update context
    update_context(query)
    
    docs_and_scores = retrieve_relevant_docs(query, k=60, relevance_threshold=0.1)

    if not docs_and_scores:
        conversation_context["last_answer"] = "❌ Sorry, no relevant information found."
        return {
            "answer": conversation_context["last_answer"],
            "confidence": 0.0
        }

    context = get_context(docs_and_scores)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    answer = chain.invoke({
        "question": query,
        "context": context,
        "last_question": conversation_context["last_question"] or "None"
    })

    # Use only the top document as the primary source
    top_doc, top_score = docs_and_scores[0]
    meta = top_doc.metadata
    source = meta.get("source", "Unknown")
    page = meta.get("page", "?")
    para = meta.get("paragraph", "?")
    section = meta.get("section", "Unknown")
    sources_info = f"📄 Source 1: {source} | Page {page} | Paragraph {para} | Section {section}"

    confidence_score = calculate_confidence_score(docs_and_scores)
    
    # Update last answer
    conversation_context["last_answer"] = answer.content.strip()
    
    return {
        "answer": f"{answer.content.strip()}\n\n---\n📚 Source Info:\n{sources_info}",
        "confidence": confidence_score
    }

# === CLI test runner ===
if __name__ == "__main__":
    while True:
        question = input("Ask a legal question (or 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            break
        if question:
            print("\n🤖 AinnAssist is thinking...\n")
            result = answer_query(question)
            print(f"Answer:\n{result['answer']}\n\nConfidence: {result['confidence']}/10")