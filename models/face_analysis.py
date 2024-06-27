# https://huggingface.co/spaces/aaronespasa/deepfake-detection/blob/main/app.py
# https://github.com/serengil/deepface

import pdb
import cv2
import os
import numpy as np
from PIL import Image
from typing import Dict
from pydantic import BaseModel, Field
from deepface import DeepFace
from skimage.feature import graycomatrix, graycoprops


class FacialFeatures(BaseModel):
    age: int = Field(..., description="Estimated age of the person.")

    gender: Dict = Field(
        ...,
        description="Confidence scores for each gender category. 'Man': Confidence score for the male gender. 'Woman': Confidence score for the female gender.",
    )

    race: str = Field(
        ...,
        description="The dominant race of the person. Possible values include 'indian,' 'asian,' 'latino hispanic,' 'black,' 'middle eastern,' and 'white.'",
    )

    emotion: str = Field(
        ...,
        description="The dominant emotion in the detected face. Possible values include 'sad,' 'angry,' 'surprise,' 'fear,' 'happy,' 'disgust,' and 'neutral' (no emotion detected).",
    )

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


class FaceAnalyzer:
    def __init__(self):
        # self.detector = DeepFace.build_model("RetinaFace")
        # self.predictor = DeepFace.build_model("FacialLandmark")
        # self.analyzer = DeepFace.build_model("DeepFace")
        pass

    def analyze_face(self, image: Image.Image) -> str:
        """Analyze the facial features of a person in an image."""
        img_array = np.array(image)
        try:
            facial_features = self._extract_facial_features(img_array)
            profile_description = self._generate_profile_description(facial_features)
            return profile_description
        except Exception as e:
            return ValueError(f"Face analysis failed: {e}")

    def _extract_facial_features(self, img_array: np.ndarray) -> FacialFeatures:
        """Extract facial features from an image using DeepFace."""
        analysis = DeepFace.analyze(
            img_path=img_array, actions=["age", "gender", "race", "emotion"]
        )[0]

        symmetry = self._check_symmetry(img_array)

        skin_texture = self._check_skin_texture(img_array)

        eyes = self._check_eyes(img_array)

        lighting = self._check_lighting(img_array)

        mouth = self._check_smile(img_array)

        background = self._check_background(img_array)

        pdb.set_trace()
        return FacialFeatures(
            age=analysis["age"],
            gender=analysis["gender"],
            race=analysis["dominant_race"],
            emotion=analysis["dominant_emotion"],
            symmetry=symmetry,
            skin_texture=skin_texture,
            eyes=eyes,
            lighting=lighting,
            mouth=mouth,
            background=background,
        )

    def _check_symmetry(self, img_array: np.ndarray) -> str:
        "Check facial symmetry using the left and right halves of the face."
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        left_half = gray[:, : w // 2]
        right_half = cv2.flip(gray[:, w // 2 :], 1)
        diff = cv2.absdiff(left_half, right_half)
        score = np.mean(diff)
        return f"Symmetry score: {score:.2f} (lower is better)"

    def _check_skin_texture(self, img_array) -> str:
        "Check skin texture using Gray Level Co-occurrence Matrix (GLCM) and its properties (e.g., contrast)"
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(
            gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
        )
        contrast = graycoprops(glcm, "contrast")[0, 0]
        return f"Skin texture contrast: {contrast:.2f} (higher is better)"

    def _check_eyes(self, img_array: np.ndarray) -> str:
        "Check the presence of eyes using Haar cascades."
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        if eye_cascade.empty():
            raise ValueError("Haar cascade xml file for eyes not found.")
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return f"Eyes detected: {len(eyes)} (should be 2)"

    def _check_lighting(self, img_array: np.ndarray) -> str:
        "Check lighting conditions using the mean value of the Value channel in HSV color space."
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        mean_value = np.mean(value_channel)
        return f"Lighting mean value: {mean_value:.2f} (consistent lighting expected)"

    def _check_smile(self, img_array: np.ndarray) -> str:
        "Check the presence of mouth using Haar cascades."
        mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        mouths = mouth_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return f"Teeth/mouth regions detected: {len(mouths)}"

    def _check_background(self, img_array: np.ndarray) -> str:
        "Check the background of the image using edge detection."
        # Simplified background check assuming single-color background for simplicity
        edges = cv2.Canny(img_array, 100, 200)
        edge_density = np.mean(edges)
        return f"Background edge density: {edge_density:.2f}"

    def _generate_profile_description(self, features: FacialFeatures) -> str:
        pdb.set_trace()
        profile_description = (
            f"Age: {features.age}, "
            f"Gender: {features.gender}, "
            f"Race: {features.race}, "
            f"Emotion: {features.emotion}, "
            f"Symmetry: {features.symmetry}, "
            f"Skin Texture: {features.skin_texture}, "
            f"Eyes: {features.eyes}, "
            f"Lighting: {features.lighting}, "
            f"Mouth: {features.mouth}, "
            f"Background: {features.background}"
        )

        return profile_description
