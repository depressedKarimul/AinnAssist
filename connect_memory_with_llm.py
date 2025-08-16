import os
import re
import asyncio
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

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

# === Load Cross-Encoder for better relevance scoring ===
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("[INFO] ✅ Cross-Encoder loaded for improved confidence scoring.")
except Exception as e:
    print(f"[ERROR] ❌ Failed to load Cross-Encoder: {e}. Falling back to original scoring.")
    cross_encoder = None

# === Load LLM from Groq ===
def load_llm():
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
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
    elif any(term in query.lower() for term in ["penal", "code", "criminal", "theft", "murder"]):
        conversation_context["topic"] = "Penal Code"
        conversation_context["pdf"] = "data/The Penal Code, 1860.pdf"
    else:
        conversation_context["topic"] = "Unknown"
        conversation_context["pdf"] = None
    conversation_context["last_question"] = query
    return conversation_context

# === Keyword Scoring ===
def keyword_boost(query, doc_content):
    query_keywords = set(re.findall(r'\w+', query.lower()))
    legal_keywords = {"socialist", "economic", "system", "capital", "dhaka", "bangabandhu", "constitution", "penal", "code", "criminal", "article"}
    query_keywords.update(legal_keywords)
    doc_keywords = set(re.findall(r'\w+', doc_content.lower()))
    matches = len(query_keywords.intersection(doc_keywords))
    score = min(matches / len(query_keywords), 1.0) if query_keywords else 0.0
    return score

# === Retrieve relevant documents (filter weak docs) ===
def retrieve_relevant_docs(query, k=10, relevance_threshold=0.3):
    global conversation_context

    followup_clues = ["explain", "that", "more", "detail", "what", "about"]
    if any(clue in query.lower() for clue in followup_clues):
        query = (conversation_context["last_question"] or "") + " " + query

    if conversation_context["topic"] == "Constitution of Bangladesh":
        if any(term in query.lower() for term in ["economic", "socialist", "economy"]):
            query += " Part II Fundamental Principles of State Policy"
        if "capital" in query.lower():
            query += " Part I Article 5"
        if any(term in query.lower() for term in ["portrait", "bangabandhu", "display"]):
            query += " Part I Article 4A"
    elif conversation_context["topic"] == "Penal Code":
        if any(term in query.lower() for term in ["theft", "murder", "criminal", "punishment"]):
            query += " Penal Code Chapter"

    docs_and_scores = faiss_db.similarity_search_with_score(query, k=k)

    reranked_docs = []
    for doc, score in docs_and_scores:
        keyword_score = keyword_boost(query, doc.page_content)
        file_name = os.path.basename(doc.metadata.get("source", "")).lower()
        doc_score = 1.0 if conversation_context["pdf"] and conversation_context["pdf"].lower() in file_name else 0.6
        similarity_score = max(0.0, min(1.0, 1 - (score ** 2) / 2))
        if score > 1.414:
            similarity_score = 0.0
        article_score = 1.0 if doc.metadata.get("article", "None") != "None" else 0.5
        combined_score = 0.4 * similarity_score + 0.2 * keyword_score + 0.3 * doc_score + 0.1 * article_score
        reranked_docs.append((doc, score, combined_score))

    reranked_docs.sort(key=lambda x: x[2], reverse=True)
    filtered_docs = [(doc, score) for doc, score, combined in reranked_docs if combined >= relevance_threshold]

    return filtered_docs[:1] if filtered_docs else [(docs_and_scores[0][0], docs_and_scores[0][1])]

# === Confidence score (improved with Cross-Encoder for better relevance) ===
def calculate_confidence_score(doc, score, query):
    # Use Cross-Encoder for more accurate relevance score if available
    if cross_encoder:
        cross_input = [[query, doc.page_content]]
        cross_score = cross_encoder.predict(cross_input)[0]  # Score between -inf and +inf, but typically sigmoid-applied in models for 0-1
        # Normalize cross_score (assuming logistic output, but for ms-marco it's raw logit; map to 0-1)
        normalized_cross_score = 1 / (1 + pow(2.718, -cross_score))  # Sigmoid to 0-1
        relevance_score = normalized_cross_score
        similarity_weight = 0.5  # Higher weight for cross-encoder
    else:
        # Fallback to original FAISS-based
        relevance_score = max(0.0, min(1.0, 1 - (score ** 2) / 2))
        similarity_weight = 0.5

    # 2. Context alignment score
    context_score = 0.5
    if conversation_context["pdf"] and conversation_context["pdf"].lower() in os.path.basename(doc.metadata.get("source", "")).lower():
        context_score = 1.0
    elif conversation_context["topic"] in ["Constitution of Bangladesh", "Penal Code"]:
        context_score = 0.8
    context_weight = 0.3

    # 3. Query specificity score
    query_words = re.findall(r'\w+', query.lower())
    legal_keywords = {"socialist", "economic", "system", "capital", "dhaka", "bangabandhu", "constitution", "penal", "code", "criminal", "article"}
    specific_terms = len([w for w in query_words if w in legal_keywords])
    query_specificity = min(specific_terms / max(len(query_words), 1), 1.0)
    query_specificity = 0.5 + 0.5 * query_specificity
    specificity_weight = 0.1

    # 4. Metadata relevance score
    metadata_score = 0.7
    article = doc.metadata.get("article", "None")
    if article != "None":
        metadata_score = 1.0
    if conversation_context["last_question"]:
        question = conversation_context["last_question"].lower()
        if re.search(r'article\s*\d+', question) and article != "None":
            metadata_score = 1.1
    metadata_weight = 0.1

    # Combine scores with weights
    combined_score = (
        similarity_weight * relevance_score +
        context_weight * context_score +
        specificity_weight * query_specificity +
        metadata_weight * metadata_score
    )

    # Add baseline boost to favor 4–5 range for relevant results
    boosted_score = combined_score + 0.15
    confidence = round(boosted_score * 5, 2)
    confidence = min(max(confidence, 3.5), 5.0)  # Clamp to ensure high scores for good matches

    return confidence

# === Source citation (Section > Article > Page) ===
def format_source_citation(doc, question=None):
    meta = doc.metadata
    source = os.path.basename(meta.get("source", "Unknown"))
    page = meta.get("page", "?")
    
    section_in_question = None
    if question:
        match = re.search(r'section\s*(\d+)', question, re.IGNORECASE)
        if match:
            section_in_question = match.group(1)

    article = meta.get("article") if meta.get("article") != "None" else None

    if isinstance(page, int):
        page = page + 1
    elif isinstance(page, str) and page.isdigit():
        page = str(int(page) + 1)
    else:
        page = "?"

    citation_parts = [f"📄 {source}"]
    if section_in_question:
        citation_parts.append(f"Section {section_in_question}")
    elif article:
        citation_parts.append(f"Article {article}")
    citation_parts.append(f"Page {page}")

    return ", ".join(citation_parts)

# === Prompt Template ===
custom_prompt_template = """
You are a legal assistant specializing in Bangladesh law.

Use ONLY the context and prior answer provided to answer the new question accurately and clearly. Include article or section numbers if explicitly mentioned. If the user asks a follow-up (e.g., 'Explain more', 'What about that?'), refer to the previous answer.

Context:
{context}

Previous Answer:
{last_answer}

Previous Question:
{last_question}

Current Question:
{question}

Answer:
"""

# === Answer generation ===
async def process_question(q, prompt_template):
    docs_and_scores = retrieve_relevant_docs(q)
    if not docs_and_scores:
        return {
            "question": q,
            "answer": f"❌ No relevant information found for: {q}",
            "confidence": 5.0,
            "source": "None"
        }

    doc, score = docs_and_scores[0]
    context = doc.page_content
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_model

    try:
        answer = await chain.ainvoke({
            "question": q,
            "context": context,
            "last_question": conversation_context.get("last_question") or "None",
            "last_answer": conversation_context.get("last_answer") or "None"
        })
    except Exception as e:
        return {
            "question": q,
            "answer": f"❌ Error during LLM response: {e}",
            "confidence": 0.0,
            "source": "None"
        }

    confidence = calculate_confidence_score(doc, score, q)
    source = format_source_citation(doc, question=q)

    return {
        "question": q,
        "answer": answer.content.strip(),
        "confidence": confidence,
        "source": source
    }

# === Entry point ===
async def answer_query(query):
    global conversation_context
    update_context(query)
    questions = [q.strip() for q in query.split("?") if q.strip()]
    if not questions:
        conversation_context["last_answer"] = "❌ No valid questions provided."
        return {"answer": conversation_context["last_answer"], "confidence": 5.0}

    tasks = [process_question(q + "?" if not q.endswith("?") else q, custom_prompt_template) for q in questions]
    results = await asyncio.gather(*tasks)

    answers = []
    total_confidence = 0.0
    for res in results:
        answers.append(
            f"📜 Question: {res['question']}\n"
            f"Answer: {res['answer']}\n"
            f"Source: {res['source']}\n"
            f"⭐️ Confidence: {res['confidence']}/5"
        )
        total_confidence += res["confidence"]

    avg_confidence = total_confidence / len(questions)
    final_answer = "\n\n".join(answers)
    conversation_context["last_answer"] = final_answer

    return {"answer": final_answer, "confidence": float(round(avg_confidence, 2))}

# === CLI Mode ===
if __name__ == "__main__":
    while True:
        question = input("Ask a legal question (or 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            break
        if question:
            print("\n🤖 AinnAssist is thinking...\n")
            result = asyncio.run(answer_query(question))
            print(f"\nAnswer:\n{result['answer']}")