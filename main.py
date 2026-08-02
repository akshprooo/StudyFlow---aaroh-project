import json
import os
import re
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(".env")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "users.json"
HTML_FILE = BASE_DIR / "index.html"

# ── Groq Configuration ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENABLED = bool(GROQ_API_KEY)

print(f"[Config] API Key: {'SET' if GROQ_API_KEY else 'NOT SET'}")
print(f"[Config] Base URL: {GROQ_BASE_URL}")
print(f"[Config] Model: {GROQ_MODEL}")

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

app = FastAPI(title="Study Flow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AccountRequest(BaseModel):
    name: str
    username: str
    interests: list[str] = []


class CourseRequest(BaseModel):
    topic: str


class ModuleCompleteRequest(BaseModel):
    course_title: str
    module_title: str


def load_users() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_users(data: dict) -> None:
    DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── AI Client ──

def ai_chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 4000, retries: int = 2) -> str | None:
    """Call Groq via the OpenAI-compatible client with retries."""
    if not GROQ_ENABLED:
        return None
    last_error = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            if content:
                return content
        except Exception as e:
            last_error = e
            print(f"[AI Error] Attempt {attempt + 1}/{retries + 1}: {e}")
    return None


def extract_json(text: str):
    """Extract JSON from LLM response."""
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Strategy 2: Strip markdown fences and parse
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except:
            pass

    # Strategy 3: Find ALL JSON objects, return the LAST one (actual output comes after reasoning)
    best_obj = None
    best_obj_len = 0
    for match in re.finditer(r"\{[\s\S]*?\}", text):
        try:
            candidate = match.group()
            parsed = json.loads(candidate)
            if len(candidate) > best_obj_len:
                best_obj_len = len(candidate)
                best_obj = parsed
        except:
            continue

    if best_obj and best_obj_len > 200:
        return best_obj

    # Strategy 4: Try arrays
    best_arr = None
    best_arr_len = 0
    for match in re.finditer(r"\[[\s\S]*?\]", text):
        try:
            candidate = match.group()
            parsed = json.loads(candidate)
            if len(candidate) > best_arr_len:
                best_arr_len = len(candidate)
                best_arr = parsed
        except:
            continue

    if best_arr and best_arr_len > 20:
        return best_arr

    if best_obj:
        return best_obj
    if best_arr:
        return best_arr

    return None


# ── AI-Powered Suggestions ──

def get_ai_suggestions(user: dict) -> list[str]:
    """Use Groq to generate personalized course suggestions."""
    interests = user.get("interests", [])
    courses = user.get("courses", [])
    completed_courses = user.get("completed_courses", [])

    existing = [c["course_title"] for c in courses] + completed_courses

    system_msg = (
        "You are a course recommendation engine. "
        "Output ONLY a JSON array of 5 strings. No thinking, no reasoning, no explanations, no markdown. "
        "Example: [\"Quantum Computing\", \"Behavioral Economics\", \"Creative Writing\", \"Cloud Architecture\", \"Molecular Gastronomy\"]"
    )

    user_msg = f"""Suggest exactly 5 new, diverse course topics.

User interests: {', '.join(interests) if interests else 'General learning'}
Already has: {', '.join(existing) if existing else 'None'}
Completed: {', '.join(completed_courses) if completed_courses else 'None'}

Rules:
1. NEVER suggest topics already listed above.
2. Branch from interests into adjacent areas.
3. Mix practical skills with intellectual curiosity.
4. Each suggestion: 1-4 words, title-cased.
5. Output ONLY the JSON array. No other text."""

    response = ai_chat(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.9,
        max_tokens=500,
    )

    if not response:
        return []

    print(f"[Suggestions Raw] {response[:600]}...")

    data = extract_json(response)
    if isinstance(data, list):
        seen = set(t.lower() for t in existing)
        filtered = [s for s in data if isinstance(s, str) and s.strip() and s.strip().lower() not in seen]
        if len(filtered) >= 2:
            return [s.strip().title() for s in filtered[:5]]
    return []


def get_suggestions(user: dict) -> list[str]:
    """Get suggestions — powered entirely by AI."""
    if not GROQ_ENABLED:
        return []
    return get_ai_suggestions(user)


# ── AI-Powered Course Builder ──

def build_course(topic: str, interests: list[str]) -> dict:
    """Build a course — powered entirely by AI."""
    if not GROQ_ENABLED:
        raise HTTPException(status_code=503, detail="AI service not configured. Set GROQ_API_KEY environment variable.")

    topic_title = topic.title()
    interests_str = ", ".join(interests) if interests else "general learning"

    system_msg = (
        "You are an expert curriculum designer. "
        "You write rich educational articles and output ONLY valid JSON. "
        "No thinking, no reasoning, no explanations, no markdown fences, no text before or after the JSON. "
        "Your entire response must be a single parseable JSON object."
    )

    user_msg = f"""Create a 8-9 module beginner course on "{topic_title}" for a learner interested in: {interests_str}.

Output ONLY a JSON object with this exact structure. No text before or after:

{{
  "course_title": "{topic_title}",
  "target_audience": "Self-paced learner",
  "estimated_duration_hours": 4,
  "difficulty": "Beginner",
  "modules": [
    {{
      "module_number": 1,
      "title": "Understanding {topic_title}",
      "summary": "A concise introduction to {topic_title} — what it is, why it matters, and how to approach it.",
      "concepts_covered": ["Core principles", "Historical context", "Real-world relevance"],
      "lesson": {{
        "summary": "Introductory article summary.",
        "content": "<h4>Heading</h4><p>Paragraph text here...</p>",
        "points": ["Point 1", "Point 2", "Point 3"],
        "sources": [
          {{"title": "Source Name", "url": "https://example.com"}}
        ]
      }}
    }}
  ],
  "flashcards": [
    {{"front": "Question?", "back": "Answer."}}
  ],
  "practice_questions": [
    {{"question": "Question?", "answer": "Answer."}}
  ]
}}

Module plan:
1. Understanding: Define {topic_title}, explain why it matters, give historical context.
2. Core Mechanics: Break down frameworks, methodologies, common pitfalls.
3. Applying in Practice: Realistic scenario with hands-on exercise.
4. Synthesis: Connect concepts, build knowledge system, suggest next steps.

NOTE: only mix in relevant concepts from the user's interests if they naturally align with the topic. Avoid forcing unrelated topics.

Rules:
- article.content: 400-500 words of HTML using only h4, p, ol, li, strong tags.
- Sources: real, verifiable educational URLs (Wikipedia, Khan Academy, Coursera, edX, MIT, Stanford, etc.).
- Style: clear, engaging, practical — like a well-written Medium article.
- JSON: valid, double quotes only, no trailing commas, no markdown."""

    response = ai_chat(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=4000,
        retries=2,
    )

    if not response:
        raise HTTPException(status_code=503, detail="AI service did not respond. Please try again.")

    print(f"[Course Raw] Length: {len(response)} chars")
    print(f"[Course Raw Preview] {response[:1000]}...")
    print(f"[Course Raw Tail] ...{response[-500:]}")

    data = extract_json(response)

    if not data:
        print(f"[Course Error] Could not extract JSON")
        raise HTTPException(status_code=502, detail="AI returned unparseable content. Please try again.")

    if isinstance(data, list):
        print(f"[Course Error] Got list instead of dict")
        raise HTTPException(status_code=502, detail="AI returned wrong structure. Please try again.")

    if not isinstance(data, dict):
        print(f"[Course Error] Extracted data type: {type(data)}")
        raise HTTPException(status_code=502, detail="AI returned invalid structure. Please try again.")

    modules = data.get("modules", [])
    if not isinstance(modules, list) or len(modules) < 2:
        print(f"[Course Error] Invalid modules: {modules}")
        raise HTTPException(status_code=502, detail="AI returned an invalid course structure. Please try again.")

    # Normalize and validate each module
    for i, mod in enumerate(modules):
        if not isinstance(mod, dict):
            mod = {}
            modules[i] = mod
        mod["module_number"] = i + 1
        if "title" not in mod or not mod["title"]:
            mod["title"] = f"Module {i + 1}"
        if "summary" not in mod or not mod["summary"]:
            mod["summary"] = f"Explore key concepts in {topic_title}."
        if "concepts_covered" not in mod or not isinstance(mod["concepts_covered"], list):
            mod["concepts_covered"] = ["Core concepts", "Key principles", "Practical applications"]
        if "lesson" not in mod or not isinstance(mod["lesson"], dict):
            mod["lesson"] = {}
        lesson = mod["lesson"]
        if "summary" not in lesson or not lesson["summary"]:
            lesson["summary"] = mod["summary"]
        if "content" not in lesson or not lesson["content"]:
            lesson["content"] = f"<h4>{mod['title']}</h4><p>This module explores important concepts in {topic_title}.</p>"
        if "points" not in lesson or not isinstance(lesson["points"], list):
            lesson["points"] = mod["concepts_covered"]
        if "sources" not in lesson or not isinstance(lesson["sources"], list) or len(lesson["sources"]) == 0:
            lesson["sources"] = [
                {"title": f"Introduction to {topic_title}", "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"},
                {"title": "Khan Academy", "url": "https://www.khanacademy.org"},
            ]

    # Ensure top-level fields
    data.setdefault("course_title", topic_title)
    data.setdefault("target_audience", "Self-paced learner")
    data.setdefault("estimated_duration_hours", 4)
    data.setdefault("difficulty", "Beginner")
    data.setdefault("flashcards", [
        {"front": f"What is {topic_title}?", "back": "A practical discipline for systematic problem-solving."},
        {"front": "What is the core framework?", "back": "Observe, analyze, act, reflect."},
    ])
    data.setdefault("practice_questions", [
        {"question": f"Why learn {topic_title}?", "answer": "It provides structured methods for solving real problems."},
        {"question": "What is the first step?", "answer": "Observe and gather information without judgment."},
    ])

    return data


def get_user(user_id: str) -> dict:
    users = load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def save_user(user_id: str, user_data: dict) -> None:
    users = load_users()
    users[user_id] = user_data
    save_users(users)


def make_user_payload(name: str, username: str, interests: list[str]) -> dict:
    return {
        "user_id": str(uuid.uuid4())[:8],
        "name": name,
        "username": username,
        "interests": [i.strip().title() for i in interests] or ["Learning"],
        "courses": [],
        "progress": {},
        "completed_modules": [],
        "completed_courses": [],
        "daily_streak": 1,
        "last_active": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


@app.post("/api/account")
def create_account(payload: AccountRequest):
    user = make_user_payload(payload.name, payload.username, payload.interests)
    save_user(user["user_id"], user)
    return {"status": "created", "user": user, "suggested_courses": get_suggestions(user)}


@app.get("/api/profile")
def get_profile(x_user_id: str | None = Header(default=None)):
    user_id = x_user_id or ""
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user id")
    user = get_user(user_id)
    return {
        "user_id": user["user_id"],
        "name": user["name"],
        "username": user["username"],
        "interests": user["interests"],
        "courses": user["courses"],
        "progress": user["progress"],
        "completed_modules": user["completed_modules"],
        "completed_courses": user.get("completed_courses", []),
        "daily_streak": user["daily_streak"],
        "suggested_courses": get_suggestions(user),
    }


@app.post("/api/generate-course")
def generate_course(payload: CourseRequest, x_user_id: str | None = Header(default=None)):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing user id")

    user = get_user(x_user_id)
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    existing = next((c for c in user["courses"] if c["course_title"].lower() == topic.lower()), None)
    if existing:
        return {"status": "exists", "course": existing}

    course = build_course(topic, user["interests"])
    user["courses"].append(course)
    user["progress"][course["course_title"]] = 0

    topic_title = topic.title()
    if topic_title not in user["interests"]:
        user["interests"].append(topic_title)

    save_user(x_user_id, user)
    return {"status": "success", "course": course}


@app.post("/api/complete-module")
def complete_module(payload: ModuleCompleteRequest, x_user_id: str | None = Header(default=None)):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing user id")

    user = get_user(x_user_id)
    key = f"{payload.course_title}:{payload.module_title}"
    if key not in user["completed_modules"]:
        user["completed_modules"].append(key)

    course = next((c for c in user["courses"] if c["course_title"] == payload.course_title), None)
    if course:
        total = len(course.get("modules", []))
        completed = len([m for m in user["completed_modules"] if m.startswith(payload.course_title + ":")])
        user["progress"][payload.course_title] = int((completed / total) * 100) if total else 100

        if completed >= total and total > 0:
            if payload.course_title not in user.get("completed_courses", []):
                user.setdefault("completed_courses", []).append(payload.course_title)
            if payload.course_title not in user["interests"]:
                user["interests"].append(payload.course_title)

    save_user(x_user_id, user)
    return {"status": "completed"}


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)