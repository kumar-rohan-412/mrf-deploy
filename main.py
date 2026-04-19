from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from huggingface_hub import snapshot_download
import numpy as np
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import re
import unicodedata
import requests
from dotenv import load_dotenv

# ── ENVIRONMENT ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# ── RATE LIMITER ──────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── PATHS ─────────────────────────────────────────────────
TFIDF_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
LR_PATH    = os.path.join(BASE_DIR, 'models', 'lr_model.pkl')
RF_PATH    = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
PA_PATH    = os.path.join(BASE_DIR, 'models', 'pa_model.pkl')

# ── LOAD TF-IDF ───────────────────────────────────────────
print("Loading TF-IDF vectorizer...")
with open(TFIDF_PATH, 'rb') as f:
    tfidf = pickle.load(f)

# ── LOAD TRADITIONAL MODELS ───────────────────────────────
print("Loading traditional models...")
with open(LR_PATH, 'rb') as f:
    lr_model = pickle.load(f)
with open(RF_PATH, 'rb') as f:
    rf_model = pickle.load(f)
with open(PA_PATH, 'rb') as f:
    pa_model = pickle.load(f)

# ── LOAD BERT FROM HF HUB ─────────────────────────────────
print("Downloading BERT model from HF Hub...")
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_DIR  = snapshot_download(repo_id="darc412/mrf-bert-model")

bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR, local_files_only=True)
bert_model     = BertForSequenceClassification.from_pretrained(BERT_DIR, local_files_only=True)
bert_model     = bert_model.to(device)
bert_model.eval()
print(f"All models loaded! Device: {device}")

# ── APP SETUP ─────────────────────────────────────────────
app = FastAPI(title="Misinformation Resilience Framework")

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, 'frontend')),
    name="static"
)

@app.get("/app")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, 'frontend', 'index.html'))

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONSTANTS ─────────────────────────────────────────────
MAX_CHARS                = 5000
MIN_WORDS                = 3
LOW_CONFIDENCE_THRESHOLD = 60.0

# ── ENSEMBLE WEIGHTS ──────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    'bert':                0.35,
    'passive_aggressive':  0.30,
    'random_forest':       0.20,
    'logistic_regression': 0.15,
}

# ── HELPERS ───────────────────────────────────────────────
def strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_gibberish(text: str) -> bool:
    words = re.findall(r'[a-zA-Z]{2,}', text)
    return len(words) < MIN_WORDS

# ── FACT CHECK ────────────────────────────────────────────
def query_fact_check_api(claim: str) -> dict:
    if not FACT_CHECK_API_KEY:
        return {"found": False, "rating": None, "source": None, "url": None}
    try:
        query    = claim[:200]
        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "query":        query,
                "key":          FACT_CHECK_API_KEY,
                "languageCode": "en",
                "maxAgeDays":   3650,
                "pageSize":     3
            },
            timeout=5
        )
        if response.status_code != 200:
            return {"found": False, "rating": None, "source": None, "url": None}
        claims = response.json().get("claims", [])
        if not claims:
            return {"found": False, "rating": None, "source": None, "url": None}
        reviews = claims[0].get("claimReview", [])
        if not reviews:
            return {"found": False, "rating": None, "source": None, "url": None}
        top = reviews[0]
        return {
            "found":  True,
            "rating": top.get("textualRating", "").strip(),
            "source": top.get("publisher", {}).get("name", "Unknown"),
            "url":    top.get("url", "")
        }
    except requests.exceptions.Timeout:
        return {"found": False, "rating": None, "source": None, "url": None, "error": "Fact check API timed out"}
    except Exception:
        return {"found": False, "rating": None, "source": None, "url": None}

# ── VERDICT LOGIC ─────────────────────────────────────────
def compute_final_verdict(ensemble_label: str, fact_check: dict) -> dict:
    if not fact_check["found"]:
        return {
            "final_label":    ensemble_label,
            "verdict_reason": "No matching fact-check record found. Verdict based on model ensemble only.",
            "conflict":       False
        }
    rating = fact_check["rating"].lower()
    false_indicators = ["false", "fake", "incorrect", "misleading", "pants on fire",
                        "mostly false", "fabricated", "misinformation", "debunked",
                        "inaccurate", "wrong", "untrue", "baseless"]
    true_indicators  = ["true", "correct", "accurate", "mostly true",
                        "verified", "real", "confirmed", "fact"]
    fc_is_false = any(ind in rating for ind in false_indicators)
    fc_is_true  = any(ind in rating for ind in true_indicators)
    if ensemble_label == "FAKE" and fc_is_false:
        return {"final_label": "FAKE", "verdict_reason": f"Both ensemble and fact-checkers ({fact_check['source']}) agree this is false — rated: '{fact_check['rating']}'.", "conflict": False}
    elif ensemble_label == "REAL" and fc_is_true:
        return {"final_label": "REAL", "verdict_reason": f"Both ensemble and fact-checkers ({fact_check['source']}) agree this is credible — rated: '{fact_check['rating']}'.", "conflict": False}
    elif ensemble_label == "REAL" and fc_is_false:
        return {"final_label": "FAKE", "verdict_reason": f"Conflict — ensemble predicted REAL but fact-checkers ({fact_check['source']}) rated this as '{fact_check['rating']}'. Fact-check takes priority.", "conflict": True}
    elif ensemble_label == "FAKE" and fc_is_true:
        return {"final_label": "REAL", "verdict_reason": f"Conflict — ensemble predicted FAKE but fact-checkers ({fact_check['source']}) rated this as '{fact_check['rating']}'. Fact-check takes priority.", "conflict": True}
    else:
        return {"final_label": ensemble_label, "verdict_reason": f"Fact-checkers ({fact_check['source']}) rated this as '{fact_check['rating']}'. Ensemble verdict retained.", "conflict": False}

# ── WEIGHTED ENSEMBLE ─────────────────────────────────────
def weighted_ensemble(predictions: dict) -> dict:
    fake_score = 0.0
    real_score = 0.0
    for model_key, result in predictions.items():
        weight     = ENSEMBLE_WEIGHTS.get(model_key, 0.0)
        confidence = float(result['confidence'])
        prediction = result['prediction']
        weighted   = confidence * weight
        if prediction == 'FAKE':
            fake_score += weighted
        else:
            real_score += weighted
    total = fake_score + real_score
    if total == 0:
        return {'label': 'UNKNOWN', 'confidence': 0.0, 'fake_score': 0.0, 'real_score': 0.0, 'low_confidence': True}
    if fake_score >= real_score:
        final_label      = 'FAKE'
        final_confidence = (fake_score / total) * 100
    else:
        final_label      = 'REAL'
        final_confidence = (real_score / total) * 100
    return {
        'label':          final_label,
        'confidence':     round(final_confidence, 2),
        'fake_score':     round(fake_score, 4),
        'real_score':     round(real_score, 4),
        'low_confidence': final_confidence < LOW_CONFIDENCE_THRESHOLD,
    }

# ── REQUEST SCHEMA ────────────────────────────────────────
class TextInput(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if v is None or v.strip() == '':
            raise ValueError('Text cannot be empty.')
        if len(v) > MAX_CHARS:
            raise ValueError(f'Text exceeds maximum allowed length of {MAX_CHARS} characters.')
        return v

# ── HEALTH ENDPOINTS ──────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Misinformation Detection API is running!"}

@app.get("/health")
def health():
    return {
        "status":           "healthy",
        "device":           str(device),
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "fact_check_api":   "connected" if FACT_CHECK_API_KEY else "missing"
    }

# ── PREDICT ENDPOINT ──────────────────────────────────────
@app.post("/predict")
@limiter.limit("20/minute")
def predict(request: Request, input: TextInput):
    raw_text = input.text
    text = strip_html(raw_text)
    text = normalize_text(text)
    if is_gibberish(text):
        raise HTTPException(status_code=422, detail="Input does not contain enough meaningful text.")
    encoded = bert_tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids      = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    with torch.no_grad():
        outputs   = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        probs     = F.softmax(outputs.logits, dim=1)
        bert_pred = torch.argmax(probs, dim=1).item()
        bert_conf = probs[0][bert_pred].item() * 100
    tfidf_vec = tfidf.transform([text])
    lr_pred  = lr_model.predict(tfidf_vec)[0]
    lr_conf  = float(max(lr_model.predict_proba(tfidf_vec)[0])) * 100
    rf_pred  = rf_model.predict(tfidf_vec)[0]
    rf_conf  = float(max(rf_model.predict_proba(tfidf_vec)[0])) * 100
    pa_pred     = pa_model.predict(tfidf_vec)[0]
    pa_decision = pa_model.decision_function(tfidf_vec)[0]
    pa_conf     = float(1 / (1 + np.exp(-abs(pa_decision)))) * 100
    model_results = {
        'bert':                {'prediction': 'REAL' if bert_pred == 1 else 'FAKE', 'confidence': round(bert_conf, 2)},
        'logistic_regression': {'prediction': 'REAL' if lr_pred == 1 else 'FAKE',   'confidence': round(lr_conf, 2)},
        'random_forest':       {'prediction': 'REAL' if rf_pred == 1 else 'FAKE',   'confidence': round(rf_conf, 2)},
        'passive_aggressive':  {'prediction': 'REAL' if pa_pred == 1 else 'FAKE',   'confidence': round(pa_conf, 2)},
    }
    ensemble   = weighted_ensemble(model_results)
    fact_check = query_fact_check_api(text)
    verdict    = compute_final_verdict(ensemble['label'], fact_check)
    warning     = None
    alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
    if alpha_ratio < 0.4:
        warning = "Input contains low alphabetic content. Results may be unreliable."
    response = {
        "label":          verdict["final_label"],
        "confidence":     ensemble['confidence'],
        "color":          "green" if verdict["final_label"] == "REAL" else "red",
        "verdict_reason": verdict["verdict_reason"],
        "conflict":       verdict["conflict"],
        "fact_check":     {"found": fact_check["found"], "rating": fact_check.get("rating"), "source": fact_check.get("source"), "url": fact_check.get("url")},
        "ensemble":       {"method": "weighted_voting", "weights": ENSEMBLE_WEIGHTS, "fake_score": ensemble['fake_score'], "real_score": ensemble['real_score'], "low_confidence": ensemble['low_confidence']},
        "models":         model_results,
        "text_analyzed":  text[:100] + "..." if len(text) > 100 else text,
        "disclaimer":     "This system detects misinformation patterns. Results reflect linguistic analysis and fact-check data — not absolute truth."
    }
    if warning:
        response["warning"] = warning
    if ensemble['low_confidence']:
        response["low_confidence_warning"] = f"Ensemble confidence is below {LOW_CONFIDENCE_THRESHOLD}%. Treat this result with caution."
    return response