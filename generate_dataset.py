import json
from pathlib import Path

QUESTIONS_PER_TOPIC = 20
OUTPUT_PATH = Path("data/raw/cs_tutor_candidates.jsonl")

TOPICS_PATH = Path("data/raw/topics.json")

with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)
    topic_list = topics["topics"]

target_examples = len(topic_list) * QUESTIONS_PER_TOPIC
print(f"Target dataset size: {target_examples} examples")


def build_generation_prompt(topic: str) -> str:
    return f"""
    Generate {QUESTIONS_PER_TOPIC} high-quality and technically accurate computer science tutoring examples about {topic} for beginner-to-intermediate students.
    Each example should contain a student question and a clear tutor-style answer. Answers should explain the concept in accessible language, include a small example where useful, and introduce important terminology without being overly complex or overly brief.
    Cover a broad range of concepts, terminology, examples, operations, and trade-offs within {topic}. Avoid duplicate or near-duplicate questions.
    Return the examples in the required structured format and do not include any additional commentary.
    Return only valid JSON in this format:
    [
    {{
        "question": "Student question",
        "answer": "Tutor answer"
    }}
    ]
    """