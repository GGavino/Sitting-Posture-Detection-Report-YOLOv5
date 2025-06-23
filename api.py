from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import subprocess
import sys

app = FastAPI()

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded file
    video_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"Video saved to {video_path}")
    
    # Run your report generator script
    process = subprocess.Popen(
        [sys.executable, "report_generator.py", video_path, "small640.pt", "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in process.stdout:
        print(line, end="")  # Print each line as it comes

    process.wait()

    # Return CSV and AI analysis as response
    csv_path = "posture_report.csv"
    ai_txt_path = "ai_posture_analysis.txt"
    if os.path.exists(csv_path) and os.path.exists(ai_txt_path):
        with open(ai_txt_path, encoding="utf-8") as f:
            ai_comments = f.read()
        return {
            "csv_url": f"/download/csv",
            "ai_comments": ai_comments
        }
    else:
        return JSONResponse(status_code=500, content={"error": "Processing failed"})

@app.get("/download/csv")
def download_csv():
    return FileResponse("posture_report.csv", media_type="text/csv", filename="posture_report.csv")