import spaces

@spaces.GPU
def _gpu_placeholder():
    """Never called. Only here because free Gradio Spaces run on ZeroGPU,
    which requires at least one @spaces.GPU function."""
    return None
import os
import re
import gradio as gr
from openai import OpenAI

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """You are a friendly, helpful medical information assistant.

Rules you must always follow:
1. Answer general health questions in simple, clear, kind language.
2. Keep answers short: a brief explanation, then 3-5 bullet points at most.
3. NEVER diagnose the user. Say "possible causes include..." instead of "you have...".
4. NEVER give specific medicine doses, dosing schedules, or prescriptions.
   For medicine questions, give general safety information and tell the user to
   confirm with a doctor or pharmacist.
5. For serious symptoms (chest pain, trouble breathing, high fever in infants,
   severe bleeding, etc.), tell the user to seek urgent medical care.
6. If the question is not about health, politely say you can only help with
   general health questions. Do not write poems, code, or stories.
7. You provide information only, not a replacement for a doctor.

Example of your style:
User: What causes a sore throat?
Assistant: A sore throat is very common and usually not serious. Common causes include:
- Viral infections like the cold or flu
- Bacterial infections such as strep throat
- Dry air, allergies, or irritation from smoke
Rest, warm fluids, and staying hydrated often help. If it lasts more than a week,
or you have trouble swallowing or breathing, please see a doctor."""

DISCLAIMER = "\n\n⚕️ *This is general information only, not medical advice. Please consult a doctor or pharmacist for personal health concerns.*"

# ---------------- SAFETY ----------------
EMERGENCY_PATTERNS = [
    r"\bsuicid", r"\bkill myself\b", r"\bend my life\b", r"\bself[- ]?harm",
    r"\bchest pain\b", r"\bcan'?t breathe\b", r"\bnot breathing\b",
    r"\bheart attack\b", r"\bstroke\b", r"\bunconscious\b",
    r"\bsevere bleeding\b", r"\boverdosed?\b",
]
DANGEROUS_PATTERNS = [
    r"\bhow many (pills|tablets|capsules)\b.*\b(die|kill|overdose|fatal|lethal)",
    r"\b(lethal|fatal) dose\b",
    r"\bhow (much|many)\b.*\b(to|would) (die|kill)",
    r"\bget high\b.*\b(medicine|pills|tablets|drug)",
]
EMERGENCY_REPLY = (
    "🚨 What you describe may be serious. Please **contact your local emergency "
    "number or go to the nearest hospital right now**. If you are thinking about "
    "harming yourself, please reach out to a trusted person or a local crisis "
    "helpline immediately. You do not have to go through this alone."
)
DANGEROUS_REPLY = (
    "I'm sorry, but I can't help with that. If you or someone else may have taken "
    "too much medicine or might be in danger, please contact emergency services "
    "or a poison control centre right away."
)
DOSE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|tablets?|pills?|capsules?)\b", re.I)
DIAGNOSIS_PATTERN = re.compile(r"\byou (definitely |probably )?have\b", re.I)
DOSE_BLOCK_REPLY = (
    "The right dose of any medicine depends on age, weight, health conditions, and "
    "the exact product. For safety, I can't give specific doses. Please check the "
    "label, or ask a doctor or pharmacist."
)

def check_input(text):
    t = text.lower()
    for p in DANGEROUS_PATTERNS:
        if re.search(p, t):
            return DANGEROUS_REPLY
    for p in EMERGENCY_PATTERNS:
        if re.search(p, t):
            return EMERGENCY_REPLY
    return None

def check_output(text):
    if DOSE_PATTERN.search(text):
        return DOSE_BLOCK_REPLY + DISCLAIMER
    text = DIAGNOSIS_PATTERN.sub("you may have", text)
    return text + DISCLAIMER

# ---------------- LLM ----------------
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

def call_llm(messages):
    last_error = None
    for model in MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.4,
                extra_body={"reasoning_effort": "low"},
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("empty answer (model used all tokens thinking)")
            return text
        except Exception as e:
            last_error = e
            print(f"[warning] {model} failed: {str(e)[-300:]}")
    raise RuntimeError(f"All models failed. Last error: {last_error}")

def get_answer(user_message, history=None):
    blocked = check_input(user_message)
    if blocked:
        return blocked
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user", "content": user_message})
    try:
        answer = call_llm(messages)
    except Exception:
        return "Sorry, I couldn't reach the AI model right now. Please try again in a few minutes."
    return check_output(answer)

# ---------------- UI ----------------
def respond(message, history):
    return get_answer(message, history)

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="🩺 General Health Assistant",
    description="Ask general health questions. I provide information only, not medical advice.",
    examples=[
        "What causes a sore throat?",
        "Is paracetamol safe for children?",
        "How can I sleep better?",
        "What are the symptoms of dehydration?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()