from dotenv import load_dotenv
import os

load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
import requests

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

def taxlens_pipeline(inputs):
    response = requests.post(
        "http://localhost:3500/api/ask",
        json={"question": inputs["question"]}
    )
    result = response.json()
    return {
        "answer": result["answer"],
        "confidence_score": result.get("confidence_score", 0),
        "requires_human_review": result.get("requires_human_review", False),
        "sources": result.get("sources", []),
        "latency_seconds": result.get("latency_seconds", 0)
    }

def correctness_evaluator(run, example):
    predicted = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    overlap = expected_words.intersection(predicted_words)
    score = len(overlap) / len(expected_words) if expected_words else 0
    return {
        "key": "correctness",
        "score": round(score, 2),
        "comment": "Word overlap score between predicted and expected answer"
    }

def helpfulness_evaluator(run, example):
    answer = run.outputs.get("answer", "")
    no_info_phrase = "I do not have enough information"
    if no_info_phrase.lower() in answer.lower():
        score = 0.2
        comment = "Answer indicates insufficient information"
    elif len(answer) > 100:
        score = 0.9
        comment = "Answer is detailed and helpful"
    elif len(answer) > 50:
        score = 0.7
        comment = "Answer is moderately helpful"
    else:
        score = 0.4
        comment = "Answer is too short to be fully helpful"
    return {
        "key": "helpfulness",
        "score": score,
        "comment": comment
    }

def confidence_evaluator(run, example):
    confidence = run.outputs.get("confidence_score", 0)
    return {
        "key": "confidence",
        "score": confidence,
        "comment": "System confidence score for this answer"
    }

def latency_evaluator(run, example):
    latency = run.outputs.get("latency_seconds", 0)
    if latency < 2:
        score = 1.0
        comment = "Excellent response time"
    elif latency < 5:
        score = 0.7
        comment = "Acceptable response time"
    else:
        score = 0.3
        comment = "Response time is too slow"
    return {
        "key": "latency",
        "score": score,
        "comment": comment
    }

def human_review_evaluator(run, example):
    requires_review = run.outputs.get("requires_human_review", False)
    answer = run.outputs.get("answer", "")
    no_info_phrase = "I do not have enough information"
    should_flag = no_info_phrase.lower() in answer.lower()
    correctly_flagged = requires_review == should_flag
    return {
        "key": "human_review_accuracy",
        "score": 1.0 if correctly_flagged else 0.0,
        "comment": "Whether human review flag was correctly applied"
    }

def source_coverage_evaluator(run, example):
    sources = run.outputs.get("sources", [])
    if len(sources) >= 4:
        score = 1.0
        comment = "Excellent source coverage"
    elif len(sources) >= 2:
        score = 0.7
        comment = "Good source coverage"
    elif len(sources) >= 1:
        score = 0.4
        comment = "Minimal source coverage"
    else:
        score = 0.0
        comment = "No sources retrieved"
    return {
        "key": "source_coverage",
        "score": score,
        "comment": comment
    }

def hallucination_evaluator(run, example):
    answer = run.outputs.get("answer", "").lower()
    sources = run.outputs.get("sources", [])
    expected = example.outputs.get("answer", "").lower()

    # Check 1 - answer contains no information phrase
    no_info_phrase = "i do not have enough information"
    if no_info_phrase in answer:
        return {
            "key": "hallucination",
            "score": 1.0,
            "comment": "No hallucination detected. System correctly admitted lack of knowledge."
        }

    # Check 2 - answer has no sources to ground it
    if len(sources) == 0:
        return {
            "key": "hallucination",
            "score": 0.0,
            "comment": "Potential hallucination. Answer generated with no retrieved sources."
        }

    # Check 3 - check if key expected words appear in answer
    expected_words = set(expected.split())
    answer_words = set(answer.split())
    overlap = expected_words.intersection(answer_words)
    overlap_ratio = len(overlap) / len(expected_words) if expected_words else 0

    if overlap_ratio > 0.5:
        score = 1.0
        comment = "Low hallucination risk. Answer aligns well with expected content."
    elif overlap_ratio > 0.2:
        score = 0.6
        comment = "Moderate hallucination risk. Partial alignment with expected content."
    else:
        score = 0.2
        comment = "High hallucination risk. Answer does not align with expected content."

    return {
        "key": "hallucination",
        "score": score,
        "comment": comment
    }

results = evaluate(
    taxlens_pipeline,
    data="taxlens-evaluation",
    evaluators=[
        correctness_evaluator,
        helpfulness_evaluator,
        confidence_evaluator,
        latency_evaluator,
        human_review_evaluator,
        source_coverage_evaluator,
        hallucination_evaluator
    ],
    experiment_prefix="taxlens-eval-run-3",
    client=client
)

print("Evaluation complete!")
print(results)