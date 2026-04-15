from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
from src.inference import PotholePredictor

app = FastAPI(title="Pothole Prediction API")
predictor = PotholePredictor()

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict_risk(
    image: UploadFile = File(...),
    weather: str = Form(None),
    traffic: str = Form("Low"),
    temp: float = Form(25.0)
):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Auto-detect weather if not provided
        if weather is None:
            weather = predictor.estimate_weather(file_path)
        
        # Run inference
        risk_score = predictor.predict(file_path, weather=weather, traffic=traffic, temperature=temp)
        
        # Gemini cross-verification (if key available)
        gemini_result = predictor.verify_with_gemini(file_path)
        
        # Risk level calibration
        risk_level = "CRITICAL" if risk_score > 0.1 else \
                     "HIGH" if risk_score > 0.07 else \
                     "MODERATE" if risk_score > 0.05 else "LOW"
        
        return {
            "prediction": {
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "conditions": {
                    "weather": weather,
                    "traffic": traffic,
                    "temperature": temp
                }
            },
            "gemini_verification": gemini_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
