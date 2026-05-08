from __future__ import annotations

import base64
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from resemblyzer import VoiceEncoder, preprocess_wav

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "voicebiometric.sqlite3"
DEFAULT_ADMIN_PASSCODE = "5846"
VERIFY_THRESHOLD = 0.72
ENROLL_SAMPLE_TARGET = 6

STATIC_DIR.mkdir(parents=True, exist_ok=True)

encoder = VoiceEncoder()

app = FastAPI(title="Voice Biometric Local")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS members (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                employee_id TEXT NOT NULL UNIQUE,
                embeddings TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                enrolled_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id TEXT,
                member_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('granted', 'denied')),
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            ("admin_passcode", DEFAULT_ADMIN_PASSCODE),
        )


def read_setting(key: str, default: str | None = None) -> str | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return default
    return str(row["value"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def decode_audio_base64(audio_base64: str) -> bytes:
    payload = audio_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc


def compute_embedding(audio_base64: str) -> np.ndarray:
    audio_bytes = decode_audio_base64(audio_base64)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = Path(tmp_file.name)
        wav = preprocess_wav(tmp_path)
        return encoder.embed_utterance(wav)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(exc)}") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            os.unlink(tmp_path)


def member_similarity_summary(new_embedding: np.ndarray, raw_embeddings: str) -> tuple[float, float] | None:
    try:
        embeddings = json.loads(raw_embeddings)
    except Exception:
        return None
    if not embeddings:
        return None
    try:
        emb_array = np.asarray(embeddings, dtype=np.float32)
        if emb_array.ndim != 2:
            return None
        scores = np.array([cosine_similarity(new_embedding, e) for e in emb_array])
        return float(np.max(scores)), float(np.mean(scores))
    except Exception:
        return None


def resolve_member_for_verification(connection: sqlite3.Connection, identifier: str) -> sqlite3.Row:
    normalized = identifier.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Enter a member name or employee ID")

    row = connection.execute(
        "SELECT id, name, employee_id, embeddings FROM members WHERE active = 1 AND id = ?",
        (normalized,),
    ).fetchone()
    if row is not None:
        return row

    row = connection.execute(
        "SELECT id, name, employee_id, embeddings FROM members WHERE active = 1 AND employee_id = ?",
        (normalized,),
    ).fetchone()
    if row is not None:
        return row

    rows = connection.execute(
        "SELECT id, name, employee_id, embeddings FROM members WHERE active = 1 AND lower(name) = lower(?) ORDER BY enrolled_at DESC",
        (normalized,),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        raise HTTPException(status_code=409, detail="Member name is ambiguous; use the employee ID")

    raise HTTPException(status_code=404, detail="Member not found")


def member_payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        embeddings = json.loads(row["embeddings"])
        sample_count = len(embeddings) if isinstance(embeddings, list) else 0
    except Exception:
        sample_count = 0
    return {
        "id": row["id"],
        "name": row["name"],
        "employee_id": row["employee_id"],
        "active": bool(row["active"]),
        "enrolled_at": row["enrolled_at"],
        "sample_count": sample_count,
    }


def log_access(member_id: str | None, member_name: str, status: str, confidence: float) -> None:
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO access_logs (member_id, member_name, status, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
            (member_id, member_name, status, confidence, now_iso()),
        )


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    with get_connection() as connection:
        member_count = connection.execute("SELECT COUNT(*) AS count FROM members").fetchone()["count"]
        active_count = connection.execute("SELECT COUNT(*) AS count FROM members WHERE active = 1").fetchone()["count"]
        log_count = connection.execute("SELECT COUNT(*) AS count FROM access_logs").fetchone()["count"]
    return {"status": "ok", "platform": "Operational",
            "total_members": member_count, "active_members": active_count, "log_count": log_count}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    with get_connection() as connection:
        total = connection.execute("SELECT COUNT(*) AS count FROM members").fetchone()["count"]
        active = connection.execute("SELECT COUNT(*) AS count FROM members WHERE active = 1").fetchone()["count"]
        logs = connection.execute("SELECT COUNT(*) AS count FROM access_logs").fetchone()["count"]
    return {"total_members": total, "active_members": active, "log_count": logs, "platform": "Operational"}


class VerifyRequest(BaseModel):
    audio_base64: str
    member_identifier: str


class EnrollRequest(BaseModel):
    name: str
    employee_id: str
    samples: list[str]
    admin_passcode: str


class LoginRequest(BaseModel):
    passcode: str


@app.post("/api/admin/login")
def admin_login(req: LoginRequest) -> dict[str, Any]:
    configured = read_setting("admin_passcode", DEFAULT_ADMIN_PASSCODE) or DEFAULT_ADMIN_PASSCODE
    return {"ok": req.passcode == configured}


@app.get("/api/members")
def list_members() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, name, employee_id, embeddings, active, enrolled_at FROM members WHERE active = 1 ORDER BY enrolled_at DESC"
        ).fetchall()
    return [member_payload(row) for row in rows]


@app.get("/api/logs")
def list_logs(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, member_id, member_name, status, confidence, timestamp FROM access_logs ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": row["id"], "member_id": row["member_id"], "member_name": row["member_name"],
             "status": row["status"], "confidence": row["confidence"], "timestamp": row["timestamp"]}
            for row in rows]


@app.post("/api/members/enroll")
def enroll_member(req: EnrollRequest) -> dict[str, Any]:
    name = req.name.strip()
    employee_id = req.employee_id.strip()
    configured = read_setting("admin_passcode", DEFAULT_ADMIN_PASSCODE) or DEFAULT_ADMIN_PASSCODE

    if not name or not employee_id:
        raise HTTPException(status_code=400, detail="Name and employee ID are required")
    if req.admin_passcode.strip() != configured:
        raise HTTPException(status_code=403, detail="Administrator access required")
    if len(req.samples) < ENROLL_SAMPLE_TARGET:
        raise HTTPException(status_code=400, detail="Record six voice samples before saving")

    embeddings: list[list[float]] = []
    for sample in req.samples:
        embedding = compute_embedding(sample)
        embeddings.append(embedding.tolist())

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM members WHERE employee_id = ?", (employee_id,)
        ).fetchone()
        if existing is not None:
            raise HTTPException(status_code=409, detail="Employee ID already exists")

        member_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO members (id, name, employee_id, embeddings, active, enrolled_at) VALUES (?, ?, ?, ?, 1, ?)",
            (member_id, name, employee_id, json.dumps(embeddings), now_iso()),
        )

    return {"ok": True, "member": {"id": member_id, "name": name,
                                    "employee_id": employee_id, "sample_count": len(embeddings)}}


@app.post("/api/verify")
def verify_audio(req: VerifyRequest) -> dict[str, Any]:
    new_embedding = compute_embedding(req.audio_base64)

    with get_connection() as connection:
        member = resolve_member_for_verification(connection, req.member_identifier)

    summary = member_similarity_summary(new_embedding, member["embeddings"])
    if summary is None:
        log_access(member["id"], member["name"], "denied", 0.0)
        return {"access": False, "score": 0.0, "user": member["name"]}

    best_score, avg_score = summary
    confidence = avg_score  # average across all enrolled samples

    if confidence < VERIFY_THRESHOLD:
        log_access(member["id"], member["name"], "denied", confidence)
        return {"access": False, "score": confidence, "user": member["name"]}

    log_access(member["id"], member["name"], "granted", confidence)
    return {"access": True, "score": confidence, "user": member["name"]}


@app.post("/api/members/{member_id}/deactivate")
def deactivate_member(member_id: str) -> dict[str, Any]:
    with get_connection() as connection:
        result = connection.execute(
            "UPDATE members SET active = 0 WHERE id = ?", (member_id,)
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"ok": True, "member_id": member_id}
