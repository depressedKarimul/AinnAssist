
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
async def root():
    return Response(status_code=204)

# === Legal Question Answering Endpoint ===
@app.post("/ask")
async def ask_question(data: QueryRequest):
    try:
        result = await answer_query(data.question)
        answer = str(result["answer"])

        # Clean the LLM response
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        cleaned = re.sub(r"additional_kwargs=\{\s*.*?\s*\}", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"response_metadata=\{\s*.*?\s*\}", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace(r'\n', '\n').strip()

        return {
            "question": data.question,
            "answer": cleaned,
            "confidence": result["confidence"],
            "sources": [line.split("Source: ")[1] for line in cleaned.split("\n\n") if "Source: " in line]
        }
    except Exception as e:
        return {"error": str(e)}