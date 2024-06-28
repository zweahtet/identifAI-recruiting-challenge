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
    FacialFeatures,
    DeepFaceAnalysis,
)

# Create a FastAPI app
app = FastAPI()

# Initialize the ImageAnalyzer class
analyzer = ImageAnalyzer()


class Profile(BaseModel):
    demographics: Demographics
    facial_features: FacialFeatures
    deepface_analysis: DeepFaceAnalysis


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
        facial_attributes = analyzer.analyze_face(img_array)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    profile = analyzer.generate_profile(facial_attributes)
    return Profile(**profile)


class TargetVerification(BaseModel):
    calculated_similarity_score: float = Field(
        ...,
        description="The calculated similarity score between the reference and target images using extracted facial attributes.",
    )
    predicted_similary_score: float = Field(
        ...,
        description="The predicted similarity score between the reference and target images using DeepFace model.",
    )
    is_real: bool = Field(
        ...,
        description="A boolean value indicating whether the target image is real or fake, based on DeepFace model prediction.",
    )


@app.post(
    "/verify-target-image",
    response_model=TargetVerification,
    description="Compare reference and target images to determine if the target is real or fake.",
    summary="Verify Target Image",
)
async def verify_target_image(
    reference_image: UploadFile = File(...), target_image: UploadFile = File(...)
) -> TargetVerification:
    """Compare features of reference and target images to determine if the target is real or fake."""
    try:
        img_1 = Image.open(BytesIO(await reference_image.read()))
        img_2 = Image.open(BytesIO(await target_image.read()))
        predicted_score, calculated_score, is_real = analyzer.verify_target_image(
            np.array(img_1), np.array(img_2)
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return TargetVerification(
        calculated_similarity_score=calculated_score,
        predicted_similary_score=predicted_score,
        is_real=is_real,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
