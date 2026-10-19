from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "outputs/cs-tutor-lora-v2/checkpoint-141"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.to(device)
model.eval()

def generate_response(prompt: str) -> str:
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