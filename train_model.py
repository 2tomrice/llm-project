import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
TRAIN_DATA_PATH = "data/processed/train.jsonl"
EVAL_DATA_PATH = "data/processed/eval.jsonl"

data_files = {
    "train": TRAIN_DATA_PATH,
    "eval": EVAL_DATA_PATH
}

dataset = load_dataset("json", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

train_dataset = dataset["train"].remove_columns(["topic"])
eval_dataset = dataset["eval"].remove_columns(["topic"])

training_args = SFTConfig(
    output_dir="outputs/cs-tutor-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_length=512,
    optim="adamw_torch",
    report_to="none",

)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
print("Trainer initialized successfully")
