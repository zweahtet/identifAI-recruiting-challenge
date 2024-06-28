# https://huggingface.co/spaces/aaronespasa/deepfake-detection/blob/main/app.py
# https://github.com/serengil/deepface
# https://medium.com/@gajarsky.tomas/facetorch-user-guide-a0e9fd2a5552

import pdb
import cv2
import os
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from deepface import DeepFace
from skimage.feature import graycomatrix, graycoprops


class FacialFeatures(BaseModel):
    symmetry: str = Field(
        ...,
        description="Real human faces tend to have a certain level of asymmetry, while fake images might exhibit perfect symmetry due to algorithmic generation.",
    )

    skin_texture: str = Field(
        ...,
        description="Real images have detailed skin texture and visible pores, whereas fake images might have smooth and unrealistic skin.",
    )

    eyes: str = Field(
        ...,
        description="The eyes should have a realistic appearance, including the sclera, iris, and pupil details. Fake images often fail to replicate the complexity of real human eyes.",
    )

    lighting: str = Field(
        ...,
        description=" Consistent lighting and shadows are crucial. Real images have natural light falloff and shadow details, whereas fake images might have inconsistent lighting.",
    )

    mouth: str = Field(
        ...,
        description="The teeth and mouth region should have a realistic structure and texture",
    )

    background: str = Field(
        ...,
        description="Real images have background details that blend naturally with the subject, while fake images might have blurry or mismatched backgrounds",
    )


class Demographics(BaseModel):
    age: int = Field(..., description="Estimated age of the person.")
    gender: str = Field(
        ...,
        description="The dominant gender of the person. Possible values include 'Man' or 'Woman.'",
    )
    race: str = Field(
        ...,
        description="The dominant race of the person. Possible values include 'indian,' 'asian,' 'latino hispanic,' 'black,' 'middle eastern,' and 'white.'",
    )


class FacialExpressions(BaseModel):
    emotion: str = Field(
        ...,
        description="The dominant emotion in the detected face. Possible values include 'sad,' 'angry,' 'surprise,' 'fear,' 'happy,' 'disgust,' and 'neutral' (no emotion detected).",
    )


class FacialAnalysis(BaseModel):
    face_confidence: float = Field(
        ...,
        description="The confidence score for the detected face. Indicates the reliability of the face detection.",
    )

    gender_confidence: dict = Field(
        ...,
        description="The confidence scores for each gender category. 'Man': Confidence score for the male gender. 'Woman': Confidence score for the female gender.",
    )

    facial_landmarks: List[Tuple[int, int]] = Field(
        ...,
        description="Facial landmarks detected in the image. Each landmark is represented as a tuple of (x, y) coordinates.",
    )


class ImageAnalyzer:
    def __init__(self):
        self.face_analyzer_model = "opencv"

    def _check_symmetry(self, img_array: np.ndarray) -> float:
        "Check facial symmetry using the left and right halves of the face."
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        left_half = gray[:, : w // 2]
        right_half = cv2.flip(gray[:, w // 2 :], 1)
        diff = cv2.absdiff(left_half, right_half)
        score = np.mean(diff)
        return score

    def _check_skin_texture(self, img_array) -> float:
        "Check skin texture using Gray Level Co-occurrence Matrix (GLCM) and its properties (e.g., contrast)"
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(
            gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
        )
        contrast = graycoprops(glcm, "contrast")[0, 0]
        return contrast

    def _generate_profile(
        self, facial_attributes: Tuple[Demographics, FacialExpressions, FacialAnalysis]
    ) -> Dict[str, Any]:
        """Generate a detailed profile of a person based on their facial attributes.

        Args:
            facial_attributes (Tuple[Demographics, FacialExpressions, FacialAnalysis]): A tuple containing the extracted facial attributes.

        Returns:
            Dict[str, Any]: A dictionary containing the detailed profile of the person.
        """
        profile = {
            "demographics": facial_attributes[0].model_dump(),
            "facial_expressions": facial_attributes[1].model_dump(),
            "facial_analysis": facial_attributes[2].model_dump(),
        }

        return profile

    def _extract_facial_attributes(
        self, img_array: np.ndarray
    ) -> Tuple[Demographics, FacialExpressions, FacialAnalysis]:
        """Extract facial attributes from an image using DeepFace.

        Args:
            img_array (np.ndarray): The image array containing the person's face.

        Returns:
            Tuple[Demographics, FacialExpressions, FacialAnalysis]: A tuple containing the extracted facial attributes.

        Raises:
            ValueError: If an error occurs during the image analysis.
        """
        analysis_result = DeepFace.analyze(
            img_path=img_array,
            actions=["age", "gender", "race", "emotion"],
            detector_backend=self.face_analyzer_model,
        )[0]

        symmetry_score = self._check_symmetry(img_array)
        texture_score = self._check_skin_texture(img_array)

        demographics = Demographics(
            age=analysis_result["age"],
            gender=analysis_result["dominant_gender"],
            race=analysis_result["dominant_race"],
        )

        facial_expressions = FacialExpressions(
            emotion=analysis_result["dominant_emotion"]
        )

        facial_analysis = FacialAnalysis(
            face_confidence=analysis_result["face_confidence"],
            gender_confidence=analysis_result["gender"],
        )

        return demographics, facial_expressions, facial_analysis

    def analyze_face(self, face_img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze the facial attributes of a person in an image.

        Args:
            face_img_array (np.ndarray): The image array containing the person's face.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted facial attributes.

        Raises:
            ValueError: If an error occurs during the image analysis.
        """
        try:
            facial_attributes = self._extract_facial_attributes(face_img_array)
            profile = self._generate_profile(facial_attributes)
        except Exception as e:
            return ValueError(f"Error analyzing image: {str(e)}")

        return profile

    def verify_faces(self, face_img_1: np.ndarray, face_img_2: np.ndarray) -> bool:
        """Verify if two images contain the same person."""
        try:
            verification_result = DeepFace.verify(face_img_1, face_img_2)
            verified = verification_result["verified"]
            distance = verification_result["distance"]
            match = all([verified, distance < 0.6])
        except Exception as e:
            return ValueError(f"Error verifying images: {str(e)}")

        return match
