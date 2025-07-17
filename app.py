# import os
# import re
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # === Load Environment Variables ===
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"
# DB_FAISS_PATH = "vectorstore/db_faiss"

# # === Page Configuration ===
# st.set_page_config(page_title="⚖️ AinnAssist – Navigate the Laws of Bangladesh", layout="wide")

# # === Custom CSS for Styling ===
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f5f7fb;
#         padding: 20px;
#     }
#     .block-container {
#         padding: 2rem 2rem 2rem;
#     }
#     .title-style {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #2c3e50;
#     }
#     .subtitle-style {
#         font-size: 1.2rem;
#         color: #555;
#         margin-top: -15px;
#     }
#     .stTextInput>div>div>input {
#         border: 1px solid #ccc;
#         padding: 10px;
#         border-radius: 6px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # === Header ===
# st.markdown('<div class="title-style">⚖️ AinnAssist – Navigate the Laws of Bangladesh</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle-style">AinnAssist is a smart legal companion that helps citizens of Bangladesh easily understand and access the country\'s laws, rights, and legal procedures — all in one place.</div>', unsafe_allow_html=True)

# st.divider()

# # === Utility: Remove <think>...</think> from output ===
# def strip_think_tags(text):
#     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# # === Load Vector Store ===
# @st.cache_resource
# def load_vector_store():
#     embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_NAME)
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
#     return db

# @st.cache_resource
# def load_llm():
#     return ChatGroq(api_key=GROQ_API_KEY, model="deepseek-r1-distill-llama-70b")

# # === Prompt Template ===
# custom_prompt_template = """
# <think>Ignore this part in final output. It contains reasoning instructions.</think>
# Answer the question using only the information provided in the context.
# ❗ Your response must be a short summary — no more than **2 concise sentences**. Do NOT explain your reasoning.

# Question: {question}
# Context: {context}

# Final Answer (Max 2 sentences):
# """

# def answer_query(query, db, llm):
#     docs = db.similarity_search(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     prompt = ChatPromptTemplate.from_template(custom_prompt_template)
#     chain = prompt | llm
#     raw_output = chain.invoke({"question": query, "context": context})
#     clean_output = strip_think_tags(raw_output.content if hasattr(raw_output, 'content') else str(raw_output))
#     return clean_output

# # === Upload & Process PDFs ===
# def process_pdf_files(uploaded_files):
#     all_docs = []
#     for file in uploaded_files:
#         with open(f"data/{file.name}", "wb") as f:
#             f.write(file.read())
#         loader = PyPDFLoader(f"data/{file.name}")
#         all_docs.extend(loader.load())
    
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(all_docs)

#     embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL_NAME)
#     db = FAISS.from_documents(chunks, embedding_model)
#     db.save_local(DB_FAISS_PATH)
#     return len(all_docs), len(chunks)

# # === Upload Sidebar ===
# with st.sidebar:
#     st.header("📤 Upload Legal PDFs")
#     uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

#     if st.button("🔄 Process PDFs"):
#         if uploaded_files:
#             os.makedirs("data", exist_ok=True)
#             os.makedirs("vectorstore", exist_ok=True)
#             with st.spinner("Processing uploaded PDFs..."):
#                 doc_count, chunk_count = process_pdf_files(uploaded_files)
#             st.success(f"✅ {doc_count} documents processed into {chunk_count} information chunks.")
#         else:
#             st.warning("⚠️ Please upload at least one PDF file.")

# # === Chat Interface ===
# st.subheader("💬 Ask a Legal Question")
# query = st.text_input("📌 Type your question about Bangladeshi law:")

# if st.button("🧠 Get Legal Insight") and query:
#     with st.spinner("Analyzing with AinnAssist..."):
#         try:
#             db = load_vector_store()
#             llm = load_llm()
#             response = answer_query(query, db, llm)
#             st.markdown("#### ✅ Answer")
#             st.success(response)
#         except Exception as e:
#             st.error("❌ Something went wrong while generating the answer.")




import re
from fastapi import FastAPI, Response
from pydantic import BaseModel
from connect_memory_with_llm import answer_query

app = FastAPI()

# === Request Model ===
class QueryRequest(BaseModel):
    question: str

# === Root Check Endpoint ===
@app.get("/")
def root():
    return Response(status_code=204)

# === Legal Question Answering Endpoint ===
@app.post("/ask")
def ask_question(data: QueryRequest):
    try:
        result = answer_query(data.question)
        answer = str(result["answer"])

        # === Clean the LLM response ===
        # Remove <think>...</think> tags
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)

        # Remove additional_kwargs={}
        cleaned = re.sub(r"additional_kwargs=\{\s*.*?\s*\}", "", cleaned, flags=re.DOTALL)

        # Remove response_metadata={...}
        cleaned = re.sub(r"response_metadata=\{\s*.*?\s*\}", "", cleaned, flags=re.DOTALL)

        # Replace literal '\n' with actual newlines
        cleaned = cleaned.replace(r'\n', '\n')

        # Strip leading/trailing whitespace
        cleaned_answer = cleaned.strip()

        return {
            "question": data.question,
            "answer": cleaned_answer,
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"error": str(e)}