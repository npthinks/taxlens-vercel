from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
import json
import time
from datetime import datetime
from uuid import uuid4
import re

# ─── Load env FIRST before anything else ──────────────────────────
load_dotenv()

# ─── LangSmith Setup ───────────────────────────────────────────────
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "taxlens-production"

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langsmith import Client as LangSmithClient
import langsmith

# ─── LangSmith Client ─────────────────────────────────────────────
langsmith_client = LangSmithClient(api_key=os.getenv("LANGSMITH_API_KEY"))

# ─── Structured Logger ─────────────────────────────────────────────
logger = logging.getLogger("taxlens")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

def log_event(event_type: str, data: dict):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        **data
    }
    logger.info(json.dumps(log_entry))

# ─── App Setup ─────────────────────────────────────────────────────
app = FastAPI(title="Taxlens Fast API")

origins = ["https://taxlens.biz", "https://www.taxlens.biz"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Vector Store and LLM ──────────────────────────────────────────
embeddings = PineconeEmbeddings(model="llama-text-embed-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="30percentruling"
)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ─── Prompt ────────────────────────────────────────────────────────
template = """
You are TaxLens, an AI assistant for tax related information.

You have information on:
- International Students
- 30% ruling for employers
- 30% ruling for employees

Answer the question that the user asks based on the knowledge.
- If the answer is not in the knowledge, say "I do not have enough information to answer this question."
- Be polite and professional.

Question: {question}
Knowledge: {context}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# ─── Confidence Scoring ────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75
NO_INFO_PHRASE = "I do not have enough information"

def calculate_confidence(answer: str, docs: list) -> float:
    if NO_INFO_PHRASE.lower() in answer.lower():
        return 0.2
    if len(docs) == 0:
        return 0.1
    if len(docs) >= 3 and len(answer) > 100:
        return 0.9
    if len(docs) >= 1:
        return 0.75
    return 0.5

# ─── Text Cleaner ──────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("*", "")
    return text

# ─── Models ────────────────────────────────────────────────────────
class Question(BaseModel):
    question: str

class Feedback(BaseModel):
    query_id: str
    question: str
    answer: str
    correct: bool
    correction: str = None

# ─── Routes ────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return "Taxlens Fast API"

@app.post("/api/ask")
async def ask(question: Question):
    query_id = str(uuid4())
    start_time = time.time()

    with langsmith.trace(
        name="taxlens-rag-query",
        project_name="taxlens-production",
        inputs={"question": question.question}
    ) as run:

        # Step 1 — Log incoming query
        log_event("query_received", {
            "query_id": query_id,
            "question": question.question
        })

        # Step 2 — Retrieve documents
        retrieval_start = time.time()
        docs = vectorstore.similarity_search(question.question, k=5)
        retrieval_latency = round(time.time() - retrieval_start, 3)

        log_event("retrieval_complete", {
            "query_id": query_id,
            "docs_retrieved": len(docs),
            "retrieval_latency_seconds": retrieval_latency,
            "sources": [doc.page_content[:100] for doc in docs]
        })

        # Step 3 — Build prompt
        knowledge = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = prompt.format(
            context=knowledge,
            question=question.question
        )

        # Step 4 — Generate answer
        generation_start = time.time()
        result = llm.invoke(final_prompt)
        generation_latency = round(time.time() - generation_start, 3)
        answer_text = clean_text(result.content)

        log_event("generation_complete", {
            "query_id": query_id,
            "generation_latency_seconds": generation_latency,
            "answer_preview": answer_text[:150]
        })

        # Step 5 — Confidence scoring
        confidence = calculate_confidence(answer_text, docs)
        requires_human_review = confidence < CONFIDENCE_THRESHOLD
        total_latency = round(time.time() - start_time, 3)

        log_event("confidence_scored", {
            "query_id": query_id,
            "confidence_score": confidence,
            "requires_human_review": requires_human_review,
            "total_latency_seconds": total_latency
        })

        # Step 6 — Human in the loop flag
        if requires_human_review:
            log_event("human_review_flagged", {
                "query_id": query_id,
                "question": question.question,
                "answer": answer_text,
                "confidence_score": confidence,
                "reason": "Confidence below threshold or insufficient context"
            })

            # Send to annotation queue
            try:
                if run.id:
                    langsmith_client.add_runs_to_annotation_queue(
                        queue_name="taxlens-human-review",
                        run_ids=[str(run.id)]
                    )
                    log_event("sent_to_annotation_queue", {
                        "query_id": query_id,
                        "run_id": str(run.id),
                        "queue": "taxlens-human-review"
                    })
            except Exception as e:
                log_event("annotation_queue_error", {
                    "query_id": query_id,
                    "error": str(e)
                })

        # Step 7 — Auto feedback based on confidence
        try:
            if run.id:
                langsmith_client.create_feedback(
                    run_id=str(run.id),
                    key="auto_confidence_score",
                    score=confidence,
                    comment=f"Automated confidence score. Human review required: {requires_human_review}"
                )
                log_event("auto_feedback_sent", {
                    "query_id": query_id,
                    "run_id": str(run.id),
                    "confidence": confidence,
                    "requires_human_review": requires_human_review
                })
        except Exception as e:
            log_event("auto_feedback_error", {
                "query_id": query_id,
                "error": str(e)
            })

        # Set outputs on trace
        run.outputs = {
            "answer": answer_text,
            "confidence_score": confidence,
            "requires_human_review": requires_human_review,
            "latency_seconds": total_latency
        }

        return {
            "query_id": query_id,
            "answer": answer_text,
            "confidence_score": confidence,
            "requires_human_review": requires_human_review,
            "sources": [{"content": doc.page_content[:200] + "..."} for doc in docs],
            "latency_seconds": total_latency
        }

@app.post("/api/feedback")
async def feedback(feedback: Feedback):
    log_event("feedback_received", {
        "query_id": feedback.query_id,
        "question": feedback.question,
        "answer": feedback.answer,
        "correct": feedback.correct,
        "correction": feedback.correction
    })

    # Send manual feedback score to LangSmith
    try:
        langsmith_client.create_feedback(
            run_id=feedback.query_id,
            key="correctness",
            score=1.0 if feedback.correct else 0.0,
            comment=feedback.correction
        )
        log_event("manual_feedback_sent", {
            "query_id": feedback.query_id,
            "score": 1.0 if feedback.correct else 0.0
        })
    except Exception as e:
        log_event("feedback_langsmith_error", {
            "query_id": feedback.query_id,
            "error": str(e)
        })

    return {
        "message": "Feedback received. Thank you for helping improve TaxLens.",
        "query_id": feedback.query_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3500)