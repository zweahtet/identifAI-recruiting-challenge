from typing import Dict, List
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from models.image_analyzer import (
    ImageAnalyzer,
    Demographics,
    FacialExpressions,
    FacialAnalysis,
)

# Create a FastAPI app
app = FastAPI()

# Initialize the ImageAnalyzer class
analyzer = ImageAnalyzer()


class Profile(BaseModel):
    demographics: Demographics
    facial_expressions: FacialExpressions
    facial_analysis: FacialAnalysis


@app.post(
    "/create-profile",
    response_model=Profile,
    description="Create a detailed profile of a person from an image.",
    summary="Create Profile",
)
async def create_profile(file: UploadFile = File(...)) -> Profile:
    """Create a detailed profile of a person from an image.

    Args:
        file (UploadFile): The image file containing the person's face.

    Returns:
        Profile: A Pydantic model containing the detailed profile of the person.

    Raises:
        HTTPException: If an error occurs during the image analysis.
    """
    try:
        img = Image.open(BytesIO(await file.read()))
        img_array = np.array(img)
        profile = analyzer.analyze_face(img_array)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return Profile(**profile)


class Verification(BaseModel):
    match: bool = Field(
        ..., description="Whether the two images are of the same person."
    )


@app.post(
    "/verify-images",
    response_model=Verification,
    description="Compare two images to verify if they are of the same person.",
    summary="Verify Images",
)
async def verify_images(
    file_1: UploadFile = File(...), file_2: UploadFile = File(...)
) -> Verification:
    """Compare two images to check if they are of the same person.

    Args:
        - file_1 (UploadFile): The first image file containing the person's face.
        - file_2 (UploadFile): The second image file containing the person's face.

    Returns:
        Verification: A Pydantic model containing the verification result.

    Raises:
        HTTPException: If an error occurs during the image verification
    """
    try:
        img_1 = Image.open(BytesIO(await file_1.read()))
        img_2 = Image.open(BytesIO(await file_2.read()))
        match = analyzer.verify_faces(np.array(img_1), np.array(img_2))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return Verification(match=match)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
