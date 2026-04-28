from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import base64, tempfile, os
from pathlib import Path

from resemblyzer import VoiceEncoder, preprocess_wav
from supabase import create_client

# ------------------ CONFIG ------------------
SUPABASE_URL = "https://wvscgofzdxkkkkktmbjh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind2c2Nnb2Z6ZHhra2tra3RtYmpoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU3NDIxMDYsImV4cCI6MjA5MTMxODEwNn0.C3EERBYdX4l-_MpM0iN3AcQ9kOVQUx3JXS8p_CTmGec"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
encoder = VoiceEncoder()

# ------------------ MODELS ------------------
class EmbedRequest(BaseModel):
    audio_base64: str

# ------------------ UTILS ------------------
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return float(np.dot(a, b) / denom)

# ------------------ EMBED ------------------
@app.post("/api/embed")
def embed_audio(req: EmbedRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav = preprocess_wav(Path(tmp_path))
        embedding = encoder.embed_utterance(wav)

        return {"embedding": embedding.tolist()}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ------------------ VERIFY ------------------
@app.post("/api/verify")
def verify_audio(req: EmbedRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 🔊 Generate embedding
        wav = preprocess_wav(Path(tmp_path))
        new_embedding = encoder.embed_utterance(wav)

        # 🔥 Fetch members
        response = supabase.table("members").select("id,name,embeddings").execute()
        members = response.data or []

        best_score = 0
        best_user = None

        # 🔥 IMPROVED COMPARISON (AVERAGE EMBEDDINGS)
        for m in members:
            embeddings = m.get("embeddings")

            if not embeddings or len(embeddings) == 0:
                continue

            try:
                emb_array = np.array(embeddings)
                avg_embedding = np.mean(emb_array, axis=0)

                score = cosine_similarity(new_embedding, avg_embedding)

                if score > best_score:
                    best_score = score
                    best_user = m
            except:
                continue

        # 🔐 STRICT LOGIC
        THRESHOLD = 0.85       # for granting access
        MIN_VALID_SCORE = 0.65 # for rejecting unknown voices

        # 🚫 Reject completely unknown voices
        if best_score < MIN_VALID_SCORE:
            return {
                "access": False,
                "score": float(best_score),
                "user": "Unknown"
            }

        # ✅ Grant only strong matches
        access = best_score > THRESHOLD

        return {
            "access": access,
            "score": float(best_score),
            "user": best_user["name"] if access else "Unknown"
        }

    except Exception as e:
        return {
            "access": False,
            "score": 0,
            "user": "Error",
            "error": str(e)
        }

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)