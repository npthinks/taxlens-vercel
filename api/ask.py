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

# Your existing sfetup (copy from main script)
embeddings = PineconeEmbeddings(model="llama-text-embed-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace="30percentruling")
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm = ChatGroq(model="qwen/qwen3-32b",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                reasoning_effort= "none"
                )

# Prompt template
template = """
You are an assistant to answer the user's questions for tax related information.

Answer the question that the user asks based on the context. 
If the answer is not in the context, say so.

Question: {question}
Context: {context}
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
    # Get relevant docs
    docs = vectorstore.similarity_search(question.question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Generate answer
    final_prompt = prompt.format(context=context, question=question.question)
    result = llm.invoke(final_prompt)
    
    return {
        "answer": result.content
        #"sources" : [doc.page_content for doc in docs]
        #"sources": [{"content": doc.page_content[:200] + "..." for doc in docs}],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
