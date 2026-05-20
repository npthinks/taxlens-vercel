from langsmith import Client
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "taxlens-production"



client = Client()

# Create a dataset
dataset = client.create_dataset(
    dataset_name="taxlens-evaluation",
    description="Test questions for TaxLens RAG pipeline"
)

# Add examples
examples = [
    {
        "inputs": {"question": "What is the 30 percent ruling?"},
        "outputs": {"answer": "The 30 percent ruling is a tax advantage for highly skilled migrants working in the Netherlands"}
    },
    {
        "inputs": {"question": "Who qualifies for the 30 percent ruling?"},
        "outputs": {"answer": "Employees recruited from abroad with specific expertise that is scarce in the Dutch labour market"}
    },
    {
        "inputs": {"question": "How long does the 30 percent ruling last?"},
        "outputs": {"answer": "The 30 percent ruling can be applied for a maximum of 5 years"}
    },
    {
        "inputs": {"question": "What is the salary requirement for the 30 percent ruling?"},
        "outputs": {"answer": "The employee must meet a minimum salary threshold which is adjusted annually"}
    },
    {
        "inputs": {"question": "Can international students apply for the 30 percent ruling?"},
        "outputs": {"answer": "International students are generally not eligible unless they meet the employment and salary requirements after graduation"}
    },
]

client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id
)

print("Dataset created successfully")