from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from uuid import uuid4
from langchain_groq import ChatGroq
import re

load_dotenv()
app = FastAPI(title = "Taxlens Fast API")

origins = ["https://taxlens.biz", "https://www.taxlens.biz"]

# CORS for taxlens.biz
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    # allow_origins=["https://taxlens.biz", "https://www.taxlens.biz", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embeddings = PineconeEmbeddings(model="llama-text-embed-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace="30percentruling")
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm = ChatGroq(model="llama-3.3-70b-versatile",
                #model="qwen/qwen3-32b",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                #reasoning_effort= "none"
                )


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

prompt = PromptTemplate(input_variables = ["context", "question"],
        template=template)

class Question(BaseModel):
    question: str
@app.get("/")
def read_root():
    return "Taxlens Fast API"
    
@app.post("/api/ask")
async def ask(question: Question):

    docs = vectorstore.similarity_search(question.question, k=5)
    knowledge = "\n\n".join(doc.page_content for doc in docs)
    
    final_prompt = prompt.format(context=knowledge, question=question.question)
    result = llm.invoke(final_prompt)
    
    answer_text = result.content

    def clean_text(text):
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", answer_text) 
        text = text.replace("*", "")
        return text
    
    return {
        "answer": clean_text(answer_text)
        #"sources" : [doc.page_content for doc in docs]
        #"sources": [{"content": doc.page_content[:200] + "..." for doc in docs}],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
