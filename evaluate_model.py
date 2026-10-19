import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "outputs/cs-tutor-lora/checkpoint-135"
EVAL_DATA_PATH = "data/processed/eval.jsonl"
EVAL_SAMPLE_SIZE = 5

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
fine_tuned_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

fine_tuned_model = PeftModel.from_pretrained(
    fine_tuned_base,
    ADAPTER_PATH,
)

base_model = base_model.to(device)
fine_tuned_model = fine_tuned_model.to(device)

base_model.eval()
fine_tuned_model.eval()


eval_examples = []
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as reader:
    for line in reader:
        eval_examples.append(json.loads(line))

print(f"Loaded {len(eval_examples)} evaluation examples")

def generate_answer(model, prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


for index, example in enumerate(eval_examples[:EVAL_SAMPLE_SIZE], start=1):
    question = example["messages"][0]["content"]
    reference_answer = example["messages"][1]["content"]
    base_answer = generate_answer(base_model, question)
    fine_tuned_answer = generate_answer(fine_tuned_model, question)
    print(f"\nExample {index}:")
    print(f"\nQuestion:\n{question}")
    print(f"\nReference answer:\n{reference_answer}")
    print(f"\nBase model:\n{base_answer}")
    print(f"\nFine-tuned model:\n{fine_tuned_answer}")


