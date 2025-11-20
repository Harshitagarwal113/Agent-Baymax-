# Agent-Baymax-

Baymax ![Uploading Gemini_Generated_Image_jhxtr4jhxtr4jhxt.png…]()
— a gentle health & calorie companion built with a local LLM.

## What Baymax does
- Answers basic, non-diagnostic health questions (sleep, exercise, hydration).
- Gives simple nutrition info for common foods.
- Logs food calories and water intake.
- Stores logs in `logs/`:
  - `conversations.txt`
  - `nutrition_logs.jsonl`

## Tech Stack
- Python 3
- Transformers (Hugging Face)
- Local LLM: `google/gemma-2b-it`
- Runs inside WSL (Ubuntu)

## Setup (WSL)
```bash
git clone https://github.com/Harshitagarwal113/Agent-Baymax-.git
cd Agent-Baymax-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login

