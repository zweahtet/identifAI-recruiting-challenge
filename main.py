from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import uvicorn
from facial_analysis import FacialAnalyzer

app = FastAPI()


class Profile(BaseModel):
    description: str


analyzer = FacialAnalyzer()


@app.post("/create-profile", response_model=Profile)
async def create_profile(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    profile_description = analyzer.analyze_face_torch(img)
    return Profile(description=profile_description)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
