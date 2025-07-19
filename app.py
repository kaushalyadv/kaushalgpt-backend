from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Allow all origins (you can restrict to your GitHub Pages URL later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history = []

# Lightweight model that works on Render free tier
MODEL_ID = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.get("/")
def root():
    return {"message": "✅ KaushalGPT is live on Render with lightweight model!"}

@app.post("/chat")
async def chat(request: Request):
    global chat_history
    data = await request.json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return {"response": "❌ Please provide a message."}

    # Reset memory on greetings
    if user_input.lower() in ["hi", "hello", "hey"]:
        chat_history = []

    chat_history.append(f"User: {user_input}")
    prompt = "\n".join(chat_history[-8:] + ["Assistant:"])

    try:
        result = pipe(prompt, max_new_tokens=60, temperature=0.7, do_sample=True)[0]["generated_text"]
        reply = result.split("Assistant:")[-1].strip()
        chat_history.append(f"Assistant: {reply}")
        return {"response": reply}
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}
