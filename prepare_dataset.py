import json
import random
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/cs_tutor_candidates.jsonl")
TOPICS_PATH = Path("data/raw/topics.json")
PROCESSED_DATA_PATH = Path("data/processed/cs_tutor_processed.jsonl")

TRAIN_DATA_PATH = Path("data/processed/train.jsonl")
EVAL_DATA_PATH = Path("data/processed/eval.jsonl")
CORRECTIONS_PATH = Path("data/review/corrections.json")

EVAL_PER_TOPIC = 2
RANDOM_SEED = 42

QUESTIONS_PER_TOPIC = 20


with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
    corrections = json.load(f)
print(f"Loaded {len(corrections)} corrections")

topics = []
with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics_data = json.load(f)
    topics = topics_data["topics"]

raw_examples = []
with open(RAW_DATA_PATH, "r", encoding="utf-8") as reader:
    for line in reader:
        raw_examples.append(json.loads(line))

expected_examples = len(topics) * QUESTIONS_PER_TOPIC
if len(raw_examples) != expected_examples:
    raise ValueError(f"Expected {expected_examples} examples, but found {len(raw_examples)}")






corrections_by_question = {correction["question"]: correction["corrected_answer"] for correction in corrections}
processed_examples = []
applied_corrections = 0

for index, example in enumerate(raw_examples):
    topic_index = index // QUESTIONS_PER_TOPIC
    topic = topics[topic_index]
    question = example["messages"][0]["content"]
    answer = example["messages"][1]["content"]
    if question in corrections_by_question:
        applied_corrections += 1
        answer = corrections_by_question[question]

    processed_example = {
        "topic": topic,
        "messages": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }
    processed_examples.append(processed_example)
print(f"Applied {applied_corrections} corrections")




with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as writer:
    for example in processed_examples:
        writer.write(json.dumps(example) + "\n")





train_examples = []
eval_examples = []
rng = random.Random(RANDOM_SEED)



for topic_index, topic in enumerate(topics):
    start = topic_index * QUESTIONS_PER_TOPIC
    end = start + QUESTIONS_PER_TOPIC
    topic_examples = processed_examples[start:end]
    rng.shuffle(topic_examples)
    eval_examples.extend(topic_examples[:EVAL_PER_TOPIC])
    train_examples.extend(topic_examples[EVAL_PER_TOPIC:])
print(f"Training examples: {len(train_examples)}")
print(f"Evaluation examples: {len(eval_examples)}")

with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as writer:
    for example in train_examples:
        writer.write(json.dumps(example) + "\n")

with open(EVAL_DATA_PATH, "w", encoding="utf-8") as writer:
    for example in eval_examples:
        writer.write(json.dumps(example) + "\n")

print(f"Loaded {len(topics)} topics")
print(f"Loaded {len(raw_examples)} examples")
print(f"Prepared {len(processed_examples)} processed examples")