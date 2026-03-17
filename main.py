# main.py — Labour Chowk NER Microservice
# Deploy: https://labour-chowk-ner.onrender.com
# Swagger UI: https://labour-chowk-ner.onrender.com/docs

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import spacy
import os

# ══════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════
app = FastAPI(
    title="Labour Chowk NER Service",
    description="""
## Labour Chowk — Named Entity Recognition API

Hindi/Bhojpuri/Hinglish text se worker profile extract karta hai.

### Entities:
- **NAME** — Worker ka naam
- **GENDER** — Aadmi / Aurat / Mahila
- **DOB_AGE** — Umar ya janam saal
- **HOME_ADDRESS** — Ghar ka pata
- **WORKING_ADDRESS** — Kaam dhundne ki jagah
- **SKILL** — Kaam (Raj Mistri, Electrician, Plumber etc.)
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# MODEL LOAD
# ══════════════════════════════════════════════════════════
MODEL_PATH = os.getenv("NER_MODEL_PATH", "./labour_chowk_model/model-best")

print(f"Loading model from: {MODEL_PATH}")
try:
    nlp = spacy.load(MODEL_PATH)
    print("Model loaded!")
except Exception as e:
    print(f"Model load failed: {e}")
    nlp = None

# ══════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════

class ExtractRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "humar naam ramesh ba, hum rajmistri bani, 30 saal ke bani, nalanda se bani, patna station pe kaam chahiye"
            }
        }


class StepRequest(BaseModel):
    text: str
    current_profile: Optional[dict] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "text": "mera naam sunita hai, main painter hoon",
                "current_profile": {}
            }
        }


class BulkRequest(BaseModel):
    texts: list

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "naam ba ramesh, rajmistri bani",
                    "mera naam sunita hai, painter hoon, 25 saal",
                ]
            }
        }


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

QUESTIONS = {
    "NAME":             "Aapka naam kya hai?",
    "SKILL":            "Aap kaunsa kaam karte hain? Jaise raj mistri, electrician, painter...",
    "GENDER":           "Aap aadmi hain ya aurat?",
    "DOB_AGE":          "Aapki umar kitni hai?",
    "HOME_ADDRESS":     "Aap kahan ke rehne wale hain?",
    "WORKING_ADDRESS":  "Aap abhi kahan kaam dhundh rahe hain?",
}

REQUIRED   = ["NAME", "SKILL"]
ALL_FIELDS = ["NAME", "SKILL", "GENDER", "DOB_AGE", "HOME_ADDRESS", "WORKING_ADDRESS"]


def run_ner(text: str) -> dict:
    doc = nlp(text.strip())
    entities = {}
    raw = []
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = ent.text
        raw.append({
            "text":  ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char,
        })
    return {"entities": entities, "raw": raw}


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "service":      "Labour Chowk NER Microservice",
        "status":       "running",
        "model_loaded": nlp is not None,
        "version":      "2.0.0",
        "docs":         "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": nlp is not None}


@app.post("/extract", tags=["NER"])
def extract(req: ExtractRequest):
    """
    Text se saari entities ek saath extract karo.
    NestJS backend mainly yahi endpoint use karega.
    """
    if not nlp:
        raise HTTPException(503, "Model not loaded")
    if not req.text.strip():
        raise HTTPException(400, "Text empty hai")

    result      = run_ner(req.text)
    entities    = result["entities"]
    is_complete = all(f in entities for f in REQUIRED)
    missing     = [f for f in ALL_FIELDS if f not in entities]

    return {
        "original_text":  req.text,
        "entities":       entities,
        "raw_entities":   result["raw"],
        "is_complete":    is_complete,
        "missing_fields": missing,
    }


@app.post("/extract-step", tags=["NER"])
def extract_step(req: StepRequest):
    """
    Step-by-step voice profile ke liye.
    Har ek voice answer ke baad call karo — next_question batayega.
    """
    if not nlp:
        raise HTTPException(503, "Model not loaded")

    result = run_ner(req.text)
    found  = result["entities"]
    merged = {**(req.current_profile or {}), **found}

    missing       = [f for f in ALL_FIELDS if f not in merged]
    next_question = QUESTIONS.get(missing[0]) if missing else None
    is_complete   = len(missing) == 0

    return {
        "extracted_this_turn": found,
        "merged_profile":      merged,
        "missing_fields":      missing,
        "next_question":       next_question,
        "is_complete":         is_complete,
    }


@app.post("/extract-bulk", tags=["NER"])
def extract_bulk(req: BulkRequest):
    """
    Multiple texts ek saath process karo — ek merged profile milega.
    """
    if not nlp:
        raise HTTPException(503, "Model not loaded")
    if not req.texts:
        raise HTTPException(400, "Texts list empty hai")

    results = []
    for text in req.texts:
        if str(text).strip():
            r = run_ner(str(text))
            results.append({"text": text, "entities": r["entities"]})

    merged = {}
    for r in results:
        for k, v in r["entities"].items():
            if k not in merged:
                merged[k] = v

    return {
        "individual_results": results,
        "merged_profile":     merged,
        "is_complete":        all(f in merged for f in REQUIRED),
        "missing_fields":     [f for f in ALL_FIELDS if f not in merged],
    }


@app.get("/questions", tags=["Config"])
def get_questions():
    """Frontend ke liye saare questions ki list."""
    return {
        "questions": [
            {"field": field, "question": question}
            for field, question in QUESTIONS.items()
        ],
        "required_fields": REQUIRED,
        "all_fields":      ALL_FIELDS,
    }