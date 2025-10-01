import re
from typing import Tuple, Dict

QUESTIONS = [
    {"key": "name", "question": "What's your full name?", "required": True},
    {"key": "email", "question": "What's your email address?", "required": True},
    {"key": "project", "question": "Which part are you interested in: Flow, RAG, or Both? (type Flow / RAG / Both)", "required": True},
    {"key": "notes", "question": "Any quick notes or goals for this project? (short text)", "required": False},
]

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def validate_field(key: str, val: str) -> Tuple[bool, str]:
    val = (val or "").strip()
    if key == "name":
        if not val:
            return False, "Please enter your name."
        if len(val) < 2:
            return False, "Name seems too short."
        return True, ""
    if key == "email":
        if not val:
            return False, "Please enter an email address."
        if not EMAIL_RE.match(val):
            return False, "That doesn't look like a valid email."
        return True, ""
    if key == "project":
        if not val:
            return False, "Please state Flow, RAG, or Both."
        if val.lower() not in ("flow", "rag", "both"):
            return False, "Answer must be one of: Flow, RAG, Both."
        return True, ""
    return True, ""

def handle_flow_step(step: int, answer: str, answers: dict) -> dict:
    if answers is None:
        answers = {}
    if step == 0:
        q0 = QUESTIONS[0]["question"]
        return {"next_step": 1, "question": q0, "finished": False, "answers": answers}
    prev_idx = step - 1
    if prev_idx < 0 or prev_idx >= len(QUESTIONS):
        return {"error": "Invalid step", "finished": False}
    key = QUESTIONS[prev_idx]["key"]
    ok, err = validate_field(key, answer)
    if not ok:
        return {"next_step": step, "question": QUESTIONS[prev_idx]["question"], "finished": False, "error": err, "answers": answers}
    answers[key] = answer.strip()
    if step >= len(QUESTIONS):
        summary_lines = [f"{k.title()}: {v}" for k, v in answers.items()]
        summary = "\n".join(summary_lines)
        return {"finished": True, "summary": summary, "answers": answers}
    next_q = QUESTIONS[step]["question"]
    return {"next_step": step + 1, "question": next_q, "finished": False, "answers": answers}
