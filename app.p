from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = []

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU

@app.get("/")
def root():
    return {"message": "✅ KaushalGPT is live with long-term memory!"}

@app.post("/chat")
async def chat(request: Request):
    global chat_history
    data = await request.json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return {"response": "❌ Please provide a message."}

    if user_input.lower() in ["hi", "hello", "hey"]:
        chat_history = []

    chat_history.append(f"User: {user_input}")
    recent_context = chat_history[-8:]
    prompt = "\n".join(recent_context + ["Assistant:"])

    result = pipe(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)[0]["generated_text"]
    reply = result.split("Assistant:")[-1].strip()

    chat_history.append(f"Assistant: {reply}")
    return {"response": reply}

@app.get("/reset")
def reset_memory():
    global chat_history
    chat_history = []
    return {"message": "🧠 Memory cleared."}
