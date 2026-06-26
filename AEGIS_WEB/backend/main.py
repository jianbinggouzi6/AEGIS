from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def mock_aegis_score():
    score = round(random.uniform(0.05, 0.95), 2)

    if score < 0.3:
        level = "normal"
        advice = "设备运行正常，无需处理"
    elif score < 0.6:
        level = "warning"
        advice = "建议48小时内复检设备状态"
    else:
        level = "critical"
        advice = "建议立即停机检查"

    return score, level, advice


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    score, level, advice = mock_aegis_score()

    return {
        "filename": file.filename,
        "score": score,
        "level": level,
        "advice": advice
    }


@app.get("/")
def root():
    return {"msg": "AEGIS backend running"}