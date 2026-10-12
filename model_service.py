from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_response(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response