import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from uuid import uuid4

load_dotenv()

# Check for API keys
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is not set.")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set.")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Google gemeini Embeddings
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-001",
#     task_type="RETRIEVAL_DOCUMENT",
#     output_dimensionality=1024
# )

#pinecone embeddings
embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",  
)


# Initialize Vector Store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace = "30percentruling"
)

# Load Documents
print("Loading documents...")
try:
    # loader = DirectoryLoader(
    #     "./Documents/",
    #     glob="**/*.txt",
    #     loader_cls=TextLoader
    # )
    # docs = loader.load()

    loader = TextLoader("/Users/nishanth_p/Documents/Interview_Code/Tax Discovery Product/Documents/InternationalStudents.txt")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
except Exception as e:
    print(f"Error loading documents: {e}")
    exit(1)

# Split Documents
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks.")

# Add to Vector Store
print("Adding to Pinecone...")
try:
    # Generate unique IDs for each chunk
    ids = [str(uuid4()) for _ in range(len(splits))]
    vectorstore.add_documents(documents=splits, ids=ids)
    print("Ingestion complete!")
except Exception as e:
    print(f"Error adding to Pinecone: {e}")
