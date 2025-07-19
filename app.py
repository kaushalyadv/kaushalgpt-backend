from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()

# Allow your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your GitHub Pages domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = []

MODEL_ID = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
    prompt = "\n".join(chat_history[-8:] + ["Assistant:"])

    try:
        result = pipe(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)[0]["generated_text"]
        reply = result.split("Assistant:")[-1].strip()
        chat_history.append(f"Assistant: {reply}")
        return {"response": reply}
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}
