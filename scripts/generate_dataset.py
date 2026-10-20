import json
from pathlib import Path
from google import genai


PROJECT_ID = "cs-tutor-llm-2026-47"
LOCATION = "global"
QUESTIONS_PER_TOPIC = 20
OUTPUT_PATH = Path("data/raw/cs_tutor_candidates.jsonl")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

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
    Prioritize technical accuracy over oversimplification.
    Avoid absolute claims unless they are universally true. When discussing algorithmic complexity, clearly distinguish average-case, worst-case, and amortized complexity where relevant.
    Do not describe Big O notation as inherently representing the worst case.
    When a concept depends on a programming language, implementation, operating system, or other context, state that dependency instead of presenting it as a universal rule.
    Include important caveats when omitting them could teach the student a misconception. Keep these caveats concise and beginner-friendly.
    Return only valid JSON in this format:
    [
    {{
        "question": "Student question",
        "answer": "Tutor answer"
    }}
    ]
    """

chat_examples = []
for topic in topic_list:
    print(f"Generating examples for: {topic}")
    prompt = build_generation_prompt(topic)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", 
        contents=prompt,
        
    )
    generated_examples = json.loads(response.text)

    if len(generated_examples) != QUESTIONS_PER_TOPIC:
        raise ValueError(
            f"Expected {QUESTIONS_PER_TOPIC} examples for {topic}, "
            f"but got {len(generated_examples)}"
    )

    print(f"Generated {len(generated_examples)} examples")


    
    for example in generated_examples:
        chat_example = {
            "messages": [
                {
                    "role": "user",
                    "content": example["question"]
                },
                {
                    "role": "assistant",
                    "content": example["answer"]
                }
            ]
        }
        chat_examples.append(chat_example)


with open(OUTPUT_PATH, "w", encoding="utf-8") as writer:
    for example in chat_examples:
        writer.write(json.dumps(example) + "\n")




print(f"Converted {len(chat_examples)} examples")