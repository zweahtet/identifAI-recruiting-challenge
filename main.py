from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import uvicorn
from models.face_analysis import FaceAnalyzer

app = FastAPI()
analyzer = FaceAnalyzer()


class Profile(BaseModel):
    description: str


class Verification(BaseModel):
    match: bool


@app.post("/create-profile", response_model=Profile)
async def create_profile(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        profile_description = analyzer.analyze_face(img)
        return Profile(description=profile_description)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


# @app.post("/verify-profile", response_model=Verification)
# async def verify_profile(file: UploadFile = File(...)):
#     try:
#         img = Image.open(BytesIO(await file.read()))
#         match = analyzer.verify_face(img)
#         return Verification(match=match)
#     except Exception as e:
#         return HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
