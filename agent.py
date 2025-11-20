cat > agent.py << 'PY'
#!/usr/bin/env python3
"""
Baymax - Health & Calorie Companion Agent (local Gemma2B-Ti)

Features:
- Baymax persona (gentle, short replies)
- Food & water knowledge base
- Calorie & water logging to logs/nutrition_logs.jsonl
- Uses google/gemma-2b-ti via transformers
"""

import os, sys, time, json, readline, re
from typing import List

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception as e:
    print("Missing transformers/torch. Run: pip install -r requirements.txt")
    print("Error:", e)
    sys.exit(1)

AGENT_NAME = "Baymax"
LOG_DIR = "logs"
CONV_LOG = os.path.join(LOG_DIR, "conversations.txt")
NUTRITION_LOG = os.path.join(LOG_DIR, "nutrition_logs.jsonl")
MAX_HISTORY = 6
MODEL_NAME = "google/gemma-2b-ti"

SYSTEM_PROMPT = (
    f"You are {AGENT_NAME}, a gentle and caring health companion. "
    "Help with non-diagnostic general health tips, calorie counting, water tracking. "
    "Be brief, kind, and advise professional help for medical issues. End session if user says 'exit' or 'I am satisfied with my care'."
)

food_kb = {
    "apple": {"serving": "1 medium (182g)", "cal": 95},
    "banana": {"serving": "1 medium (118g)", "cal": 105},
    "egg": {"serving": "1 large (50g)", "cal": 78},
    "rice (cooked)": {"serving": "1 cup (158g)", "cal": 205},
    "bread (slice)": {"serving": "1 slice (30g)", "cal": 80},
    "chicken breast": {"serving": "100 g", "cal": 165},
    "salad (mixed)": {"serving": "1 cup (50g)", "cal": 20},
    "milk (whole)": {"serving": "1 cup (240ml)", "cal": 150},
    "pizza (slice)": {"serving": "1 slice (~100g)", "cal": 285},
    "orange": {"serving": "1 medium", "cal": 62},
    "water": {"serving": "1 ml", "cal": 0}
}

water_guidance = (
    "Aim for ~2-3 liters/day depending on activity and climate. 1 glass ≈ 250 ml. This is general guidance, not medical advice."
)

health_tips = {
    "sleep": "Aim for 7-9 hours of sleep per night for most adults.",
    "exercise": "Try 20–30 minutes of moderate exercise most days.",
    "emergency": "If this is an emergency, contact local emergency services immediately."
}

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def log_conversation(entry: dict):
    ensure_log_dir()
    with open(CONV_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log_nutrition(entry: dict):
    ensure_log_dir()
    with open(NUTRITION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def find_in_kb(text: str):
    text_low = text.lower()
    keys_sorted = sorted(list(food_kb.keys()) + list(health_tips.keys()), key=len, reverse=True)
    for key in keys_sorted:
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, text_low):
            if key in food_kb:
                return {"type": "food", "key": key, "data": food_kb[key]}, key
            else:
                return {"type": "tip", "key": key, "data": health_tips[key]}, key
    if re.search(r'\bwater\b|\bdrank\b|\bhydrate\b', text_low):
        return {"type": "water_guidance", "key": "water", "data": water_guidance}, "water"
    return None, None

def parse_and_log_nutrition(user_input: str, user_identity: str = "friend"):
    text = user_input.lower()
    m = re.search(r'drank\s+(\d+)\s*(ml|m|l|liters|liter|glass|glasses)?', text)
    if m:
        qty = int(m.group(1))
        unit = m.group(2) or "ml"
        if unit.startswith("l"):
            qty_ml = qty * 1000
        elif unit.startswith("glass"):
            qty_ml = qty * 250
        else:
            qty_ml = qty
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "type": "water",
            "user": user_identity,
            "amount_ml": qty_ml,
            "raw": user_input
        }
        log_nutrition(entry)
        return f"Logged water intake: {qty_ml} ml."

    m2 = re.search(r'drank\s+(one|a|an)\s+glass', text)
    if m2:
        qty_ml = 250
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "type": "water",
            "user": user_identity,
            "amount_ml": qty_ml,
            "raw": user_input
        }
        log_nutrition(entry)
        return f"Logged water intake: {qty_ml} ml."

    for food in food_kb.keys():
        if re.search(r'\b' + re.escape(food) + r'\b', text):
            mqty = re.search(r'(\d+)\s*(?:serving|servings|slice|slices|cup|cups|piece|pieces|grams|g)?', text)
            qty = 1
            if mqty:
                try:
                    qty = int(mqty.group(1))
                except:
                    qty = 1
            base_cal = food_kb[food].get("cal", 0)
            total_cal = base_cal * qty
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "type": "food",
                "user": user_identity,
                "item": food,
                "serving_info": food_kb[food].get("serving"),
                "servings": qty,
                "calories": total_cal,
                "raw": user_input
            }
            log_nutrition(entry)
            return f"Logged: {qty} × {food} (~{total_cal} kcal)."
    return None

def load_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading model {model_name} (may take time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}.")
    return model, tokenizer, device

def generate_model_reply(model, tokenizer, device, chat_history_ids, user_input):
    try:
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": user_input}
        ]
        prompt_tensor = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        try:
            prompt_tensor = prompt_tensor.to(device)
            prompt_len = prompt_tensor.shape[-1]
        except Exception:
            prompt_len = prompt_tensor["input_ids"].shape[-1] if isinstance(prompt_tensor, dict) else 0

        outputs = model.generate(
            prompt_tensor,
            max_new_tokens=120,
            do_sample=False,
        )
        gen_ids = outputs[0, prompt_len:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if len(reply.split(".")) > 2:
            parts = reply.split(".")
            reply = ".".join(parts[:2]).strip() + "."
        if not reply:
            return None, "Sorry — I don't have a confident answer. Can you rephrase?"
        return None, reply
    except Exception as e:
        return None, f"(Error generating reply: {type(e).__name__}: {e})"

BAYMAX_TEMPLATES = [
    "Hello. I am Baymax, your personal healthcare companion.",
    "I am here to help you track calories, water, and basic wellbeing.",
    "I have logged that for you.",
    "Please rest and drink water if you are thirsty.",
    "If you need urgent medical help, contact a professional or emergency services."
]

def run_console_agent():
    print(f"\nWelcome to {AGENT_NAME} — your gentle health companion!")
    print("Type 'exit' or 'I am satisfied with my care' to end the session, 'help' for tips, or 'kb' to list KB topics.\n")
    print(SYSTEM_PROMPT)
    print("-" * 60)

    model, tokenizer, device = load_model_and_tokenizer()
    chat_history_ids = None
    history_pairs: List[tuple] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Take care.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "i am satisfied with my care"):
            print(f"{AGENT_NAME}: Thank you. I will be here if you need further care. Goodbye.")
            break

        if user_input.lower() == "help":
            print("Commands: 'exit' or 'I am satisfied with my care' to quit, 'kb' to see knowledge base topics, 'clear' to reset context.")
            continue

        if user_input.lower() == "kb":
            print("Knowledge base - food items and approximate calories:")
            for k in sorted(food_kb.keys()):
                v = food_kb[k]
                print(f" - {k}: {v.get('cal')} kcal per {v.get('serving')}")
            print("\nHealth tips:")
            for k in sorted(health_tips.keys()):
                print(f" - {k}: {health_tips[k]}")
            print("\nWater guidance (short):")
            print(" -", water_guidance)
            continue

        if user_input.lower() == "clear":
            chat_history_ids = None
            history_pairs = []
            print("Context cleared.")
            continue

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_conversation({"timestamp": now, "role": "user", "text": user_input})

        parsed_msg = parse_and_log_nutrition(user_input, user_identity="friend")
        if parsed_msg:
            confirmation = "I have logged that for you."
            print(f"\n{AGENT_NAME}: {parsed_msg} {confirmation}")
            log_conversation({
                "timestamp": now,
                "role": "assistant",
                "source": "nutrition_logger",
                "text": parsed_msg + " " + confirmation
            })
            continue

        kb_answer, matched_key = find_in_kb(user_input)
        if kb_answer:
            if kb_answer["type"] == "food":
                info = kb_answer["data"]
                reply = f"{matched_key.title()}: ~{info.get('cal')} kcal per {info.get('serving')}."
            elif kb_answer["type"] == "tip":
                reply = kb_answer["data"]
            elif kb_answer["type"] == "water_guidance":
                reply = kb_answer["data"]
            else:
                reply = "I don't have that information right now."
            print(f"\n{AGENT_NAME}: {reply}")
            log_conversation({
                "timestamp": now,
                "role": "assistant",
                "source": f"kb:{matched_key}",
                "text": reply
            })
            continue

        try:
            chat_history_ids, bot_reply = generate_model_reply(model, tokenizer, device, chat_history_ids, user_input)
        except Exception as e:
            print(f"{AGENT_NAME}: Oops, model failed to generate (error: {e}). I'll answer simply.")
            bot_reply = "Sorry, I'm having trouble responding right now. Try again or ask something else."

        if bot_reply and len(bot_reply) < 10:
            bot_reply = BAYMAX_TEMPLATES[0]

        print(f"\n{AGENT_NAME}: {bot_reply}")

        now2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_conversation({"timestamp": now2, "role": "assistant", "source": "model", "text": bot_reply})
        history_pairs.append((user_input, bot_reply))
        if len(history_pairs) > MAX_HISTORY:
            history_pairs = history_pairs[-MAX_HISTORY:]

    print("Session ended.")

if __name__ == "__main__":
    run_console_agent()
PY

