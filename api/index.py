import csv
import json
import os
import re
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin", "*"],
)

HERE = Path(__file__).parent

# ---------- Q: FastAPI server to serve student data ----------
with open(HERE / "students.csv", newline="") as f:
    STUDENTS = [{"studentId": int(r["studentId"]), "class": r["class"]} for r in csv.DictReader(f)]


@app.get("/api")
def get_students(class_: List[str] = Query(default=[], alias="class")):
    if not class_:
        return {"students": STUDENTS}
    wanted = set(class_)
    return {"students": [s for s in STUDENTS if s["class"] in wanted]}


# ---------- Q: Batch sentiment analysis ----------
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({
    "spectacular": 3.0, "exceeded": 2.0, "smiling": 2.5, "layoffs": -2.5,
    "heartbroken": -3.0, "saved": 0.0, "accepts": 0.0, "empty": -1.5,
})
PHRASES = {
    "passed away": -3.0, "can't stop smiling": 3.0, "cannot stop smiling": 3.0,
    "nobody showed up": -3.0, "no one showed up": -3.0, "falling apart": -3.0,
    "heart is broken": -3.0, "broken heart": -3.0, "cloud nine": 3.0,
    "empty inside": -3.0, "exceeded all my expectations": 3.0,
}


def classify(text: str) -> str:
    score = analyzer.polarity_scores(text)["compound"]
    low = text.lower()
    for phrase, value in PHRASES.items():
        if phrase in low:
            score += value
    if score >= 0.3:
        return "happy"
    if score <= -0.3:
        return "sad"
    return "neutral"


class SentimentRequest(BaseModel):
    sentences: List[str]


@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    return {"results": [{"sentence": s, "sentiment": classify(s)} for s in req.sentences]}


# ---------- Q: Code interpreter with error analysis ----------
class CodeRequest(BaseModel):
    code: str


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {"__name__": "__main__"})
        return {"success": True, "output": sys.stdout.getvalue()}
    except BaseException:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout


def analyze_error(code: str, tb: str) -> List[int]:
    """Find the line(s) in the submitted code where the error occurred.

    Uses an LLM via AI Pipe when AIPIPE_TOKEN is set; otherwise reads the
    line numbers for the submitted code (<string>) from the traceback.
    """
    token = os.environ.get("AIPIPE_TOKEN")
    if token:
        try:
            import urllib.request
            body = {
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": "Return JSON {\"error_lines\": [int]} with the 1-based line number(s) in CODE where the error occurred."},
                    {"role": "user", "content": f"CODE:\n{code}\n\nTRACEBACK:\n{tb}"},
                ],
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request(
                "https://aipipe.org/openai/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"]
            lines = [int(x) for x in json.loads(content)["error_lines"]]
            if lines:
                return lines
        except Exception:
            pass
    nums = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(nums[-1])] if nums else []


@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    result = execute_python_code(req.code)
    if result["success"]:
        return {"error": [], "result": result["output"]}
    return {"error": analyze_error(req.code, result["output"]), "result": result["output"]}


# ---------- Q: Vercel latency analytics ----------
with open(HERE / "latency.json") as f:
    TELEMETRY = json.load(f)


def percentile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = int(pos)
    frac = pos - lo
    return v[lo] + frac * (v[lo + 1] - v[lo]) if lo + 1 < len(v) else v[lo]


@app.post("/latency")
async def latency(request: Request):
    body = await request.json()
    regions = body.get("regions", [])
    threshold = body.get("threshold_ms", 180)
    out = []
    for region in regions:
        recs = [r for r in TELEMETRY if r["region"] == region]
        lat = [r["latency_ms"] for r in recs]
        up = [r["uptime_pct"] for r in recs]
        if not recs:
            out.append({"region": region, "avg_latency": None, "p95_latency": None, "avg_uptime": None, "breaches": 0})
            continue
        out.append({
            "region": region,
            "avg_latency": round(sum(lat) / len(lat), 2),
            "p95_latency": round(percentile(lat, 0.95), 2),
            "avg_uptime": round(sum(up) / len(up), 3),
            "breaches": sum(1 for x in lat if x > threshold),
        })
    return {"regions": out}


@app.get("/")
def root():
    return {"message": "TDS GA0 API", "endpoints": ["/api", "/sentiment", "/code-interpreter", "/latency"]}
