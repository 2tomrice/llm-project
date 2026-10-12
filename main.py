from fastapi import FastAPI
from pydantic import BaseModel
from model_service import generate_response

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(request: GenerateRequest):
    response = generate_response(request.prompt)
    return {"response": response}