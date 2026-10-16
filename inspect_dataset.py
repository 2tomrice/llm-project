import json
import random
from pathlib import Path

DATA_PATH = Path("data/raw/cs_tutor_candidates.jsonl")
SAMPLE_SIZE = 20

examples = []
with open(DATA_PATH, "r", encoding="utf-8") as reader:
    for line in reader:
        examples.append(json.loads(line))


sampled_examples = random.sample(examples, SAMPLE_SIZE)
print(f"Inspecting {len(sampled_examples)} of {len(examples)} examples")

for index, example in enumerate(sampled_examples, start=1):
    print(f"\nExample {index}:")
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}")
    print("-" * 60)