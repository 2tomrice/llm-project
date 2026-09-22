from fastapi import FastAPI
from pydantic import BaseModel

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
    return {"prompt_received": request.prompt}