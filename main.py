# ner_service/main.py
# Labour Chowk — Python NER Microservice
# Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Labour Chowk NER Service", version="1.0.0")

# CORS — NestJS aur React dono se calls aayengi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production mein apna domain daalo
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model load ────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("NER_MODEL_PATH", "./labour_chowk_model/model-best")

print(f"Loading NER model from: {MODEL_PATH}")
try:
    nlp = spacy.load(MODEL_PATH)
    print("✅ NER Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    nlp = None

# ── Request / Response schemas ────────────────────────────────────────────────
class NERRequest(BaseModel):
    text: str              # "humar naam ramesh ba..."
    language: str = "hi"   # hi = Hindi, bh = Bhojpuri (future use)

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int

class NERResponse(BaseModel):
    original_text: str
    entities: dict         # {"NAME": "ramesh", "SKILL": "rajmistri", ...}
    all_entities: list     # raw list with positions
    profile_complete: bool # sare required fields mile ya nahi

# ── Required fields for a complete profile ───────────────────────────────────
REQUIRED_FIELDS = ["NAME", "SKILL"]
OPTIONAL_FIELDS = ["GENDER", "DOB_AGE", "HOME_ADDRESS", "WORKING_ADDRESS"]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Labour Chowk NER Microservice",
        "status": "running",
        "model_loaded": nlp is not None
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": nlp is not None}


@app.post("/extract", response_model=NERResponse)
def extract_entities(request: NERRequest):
    """
    Main endpoint — text lo, entities nikaalo.
    NestJS backend yahan POST karega.
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="NER model not loaded")

    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # SpaCy se process karo
    doc = nlp(request.text.strip())

    # Entities extract karo
    entities_dict = {}
    all_entities  = []

    for ent in doc.ents:
        label = ent.label_
        # Agar same label ke multiple entities hain toh pehla lo
        if label not in entities_dict:
            entities_dict[label] = ent.text
        all_entities.append({
            "text":  ent.text,
            "label": label,
            "start": ent.start_char,
            "end":   ent.end_char
        })

    # Check karo profile complete hai ya nahi
    profile_complete = all(field in entities_dict for field in REQUIRED_FIELDS)

    return NERResponse(
        original_text    = request.text,
        entities         = entities_dict,
        all_entities     = all_entities,
        profile_complete = profile_complete
    )


@app.post("/extract-step")
def extract_step(request: NERRequest):
    """
    Step-by-step voice flow ke liye —
    Batao kaunsa field abhi tak mila aur kaunsa baaki hai.
    NestJS isko use karega yeh decide karne ke liye ki
    user se aage kya poochha jaaye.
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="NER model not loaded")

    doc = nlp(request.text.strip())

    found = {}
    for ent in doc.ents:
        if ent.label_ not in found:
            found[ent.label_] = ent.text

    all_fields    = REQUIRED_FIELDS + OPTIONAL_FIELDS
    missing       = [f for f in all_fields if f not in found]
    next_question = QUESTIONS.get(missing[0]) if missing else None

    return {
        "found":          found,
        "missing_fields": missing,
        "next_question":  next_question,
        "is_complete":    len(missing) == 0
    }


# ── Question prompts (NestJS in these as next prompt) ────────────────────────
QUESTIONS = {
    "NAME":             "Aapka naam kya hai?",
    "GENDER":           "Aap aadmi hain ya aurat?",
    "DOB_AGE":          "Aapki umar kitni hai ya janam saal kya hai?",
    "HOME_ADDRESS":     "Aap kahan ke rehne wale hain? Apna gaon ya jila batayein.",
    "WORKING_ADDRESS":  "Aap abhi kahan kaam dhundh rahe hain?",
    "SKILL":            "Aap kaunsa kaam karte hain? Jaise raj mistri, electrician, painter...",
}
