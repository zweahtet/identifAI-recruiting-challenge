# https://huggingface.co/spaces/aaronespasa/deepfake-detection/blob/main/app.py
# https://github.com/serengil/deepface
# https://medium.com/@gajarsky.tomas/facetorch-user-guide-a0e9fd2a5552

import pdb
import cv2
import dlib
import os
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from deepface import DeepFace
from utils import PREDEFINED_MIN_MAX, RACE_MAPPING


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


class FacialFeatures(BaseModel):
    facial_width_to_height_ratio: float = Field(
        ...,
        description="The ratio of the width of the face to its height. Real human faces tend to have a certain ratio, while fake images might exhibit atypical ratios due to synthetic generation.",
    )

    left_eye_aspect_ratio: float = Field(
        ...,
        description="The ratio of the width of the eye to its height. Real human eyes tend to have a certain level of openness, while fake images might exhibit atypical ratios due to synthetic generation.",
    )

    right_eye_aspect_ratio: float = Field(
        ...,
        description="The ratio of the width of the eye to its height. Real human eyes tend to have a certain level of openness, while fake images might exhibit atypical ratios due to synthetic generation.",
    )

    mouth_aspect_ratio: float = Field(
        ...,
        description="The ratio of the width of the mouth to its height. Real human mouths tend to have a certain level of openness, while fake images might exhibit atypical ratios due to synthetic generation.",
    )

    nose_to_chin_ratio: float = Field(
        ...,
        description="The ratio of the length of the nose to the total length of the lower face. Real human faces tend to have a certain ratio, while fake images might exhibit atypical ratios due to synthetic generation.",
    )

    eye_separation_ratio: float = Field(
        ...,
        description="The ratio of the distance between the eyes to the width of the face. Real human faces tend to have a certain level of eye separation, while fake images might exhibit atypical ratios due to synthetic generation.",
    )

    jaw_angle: float = Field(
        ...,
        description="The angle of the jawline. Real human faces tend to have a certain jaw angle, while fake images might exhibit atypical angles due to synthetic generation.",
    )

    symmetry: float = Field(
        ...,
        description="Real human faces tend to have a certain level of asymmetry, while fake images might exhibit perfect symmetry due to algorithmic generation.",
    )


class DeepFaceAnalysis(BaseModel):
    face_confidence: float = Field(
        ...,
        description="The confidence score for the detected face. Indicates the reliability of the face detection.",
    )

    gender_confidence: dict = Field(
        ...,
        description="The confidence scores for each gender category. 'Man': Confidence score for the male gender. 'Woman': Confidence score for the female gender.",
    )

    race_confidence: dict = Field(
        ..., description="The confidence scores for each race category."
    )


class ImageAnalyzer:
    def __init__(self):
        self.face_analyzer_model = "opencv"

    def _shape_to_np(self, shape, dtype="int") -> np.ndarray:
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _calculate_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def _calculate_angle(self, point1, point2, point3):
        v1 = point1 - point2
        v2 = point3 - point2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def _facial_width_to_height_ratio(self, landmarks):
        """Compares the width of the face to its height."""
        face_width = self._calculate_distance(landmarks[0], landmarks[16])
        face_height = self._calculate_distance(landmarks[8], landmarks[27])
        return face_width / face_height

    def _eye_aspect_ratio(self, eye_landmarks):
        """Measures the openness of the eyes."""
        A = self._calculate_distance(eye_landmarks[1], eye_landmarks[5])
        B = self._calculate_distance(eye_landmarks[2], eye_landmarks[4])
        C = self._calculate_distance(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth_landmarks):
        """Measures the openness of the mouth."""
        A = self._calculate_distance(mouth_landmarks[2], mouth_landmarks[10])
        B = self._calculate_distance(mouth_landmarks[4], mouth_landmarks[8])
        C = self._calculate_distance(mouth_landmarks[0], mouth_landmarks[6])
        return (A + B) / (2.0 * C)

    def _nose_to_chin_ratio(self, landmarks):
        """Compares the length of the nose to the total length of the lower face."""
        noseTip_to_chinBottom = self._calculate_distance(landmarks[30], landmarks[8])
        center_to_chinBottom = self._calculate_distance(landmarks[27], landmarks[8])
        return noseTip_to_chinBottom / center_to_chinBottom

    def _eye_separation_ratio(self, landmarks):
        """Measures how far apart the eyes are relative to their width."""
        inner_corners = self._calculate_distance(landmarks[39], landmarks[42])
        outer_corners = self._calculate_distance(landmarks[36], landmarks[45])
        return inner_corners / outer_corners

    def _jaw_angle(self, landmarks):
        """Measures the angle of the jawline."""
        return self._calculate_angle(landmarks[2], landmarks[8], landmarks[14])

    def _measure_symmetry(self, landmarks):
        """Measures the symmetry of the face."""

        # Define the midline of the face
        midline = (
            landmarks[27] + landmarks[8]
        ) / 2  # Midpoint between nose bridge and chin

        # Calculate distances from each landmark to the midline
        left_distances = []
        right_distances = []

        for i in range(0, 27):  # Left side landmarks
            left_distances.append(np.linalg.norm(landmarks[i] - midline))

        for i in range(27, 68):  # Right side landmarks
            right_distances.append(np.linalg.norm(landmarks[i] - midline))

        # Compare corresponding left and right distances
        asymmetry_scores = []
        for left, right in zip(left_distances, right_distances):
            asymmetry_scores.append(abs(left - right) / ((left + right) / 2))

        # Overall symmetry score (lower is more symmetric)
        symmetry_score = 1 - np.mean(asymmetry_scores)

        return symmetry_score

    def _extract_facial_landmarks(self, img_array: np.ndarray) -> np.ndarray:
        """Extract facial landmarks from an image using dlib."""
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            os.path.join("dlib_models", "shape_predictor_68_face_landmarks.dat")
        )
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            return []
        landmarks = predictor(gray, faces[0])
        facial_landmarks = self._shape_to_np(landmarks)
        return facial_landmarks

    def _extract_facial_attributes(self, img_array: np.ndarray):
        """Extract facial attributes from an image.

        Raises:
            ValueError: If no face is detected in the image.
        """
        # Extract facial landmarks
        # Landmarks are in the following order:
        # 0-16: Jawline,
        # 17-21: Right eyebrow,
        # 22-26: Left eyebrow,
        # 27-30: Nose bridge,
        # 31-35: Nose base,
        # 36-41: Right eye,
        # 42-47: Left eye,
        # 48-59: Mouth outer,
        # 60-67: Mouth inner
        landmarks = self._extract_facial_landmarks(img_array)
        if len(landmarks) == 0:
            raise ValueError("No face detected in the image.")

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]

        facial_features = FacialFeatures(
            facial_width_to_height_ratio=self._facial_width_to_height_ratio(landmarks),
            left_eye_aspect_ratio=self._eye_aspect_ratio(left_eye),
            right_eye_aspect_ratio=self._eye_aspect_ratio(right_eye),
            mouth_aspect_ratio=self._mouth_aspect_ratio(mouth),
            nose_to_chin_ratio=self._nose_to_chin_ratio(landmarks),
            eye_separation_ratio=self._eye_separation_ratio(landmarks),
            jaw_angle=self._jaw_angle(landmarks),
            symmetry=self._measure_symmetry(landmarks),
        )

        # Analyze facial attributes using DeepFace
        analysis_result = DeepFace.analyze(
            img_path=img_array,
            actions=["age", "gender", "race"],
            detector_backend=self.face_analyzer_model,
        )[0]

        demographics = Demographics(
            age=analysis_result["age"],
            gender=analysis_result["dominant_gender"],
            race=analysis_result["dominant_race"],
        )

        deepface_analysis = DeepFaceAnalysis(
            face_confidence=analysis_result["face_confidence"],
            gender_confidence=analysis_result["gender"],
            race_confidence=analysis_result["race"],
        )

        return demographics, facial_features, deepface_analysis

    def analyze_face(self, face_img_array: np.ndarray):
        """Analyze the facial attributes of a person in an image.

        Args:
            face_img_array (np.ndarray): The image array containing the person's face.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted facial attributes.
        """
        facial_attributes = self._extract_facial_attributes(face_img_array)
        return facial_attributes

    def generate_profile(self, facial_attributes) -> Dict[str, Any]:
        """Generate a detailed profile of a person based on their facial attributes.

        Args:
            facial_attributes (Tuple[Demographics, FacialExpressions, FacialAnalysis]): A tuple containing the extracted facial attributes.

        Returns:
            Dict[str, Any]: A dictionary containing the detailed profile of the person.
        """
        profile = {
            "demographics": facial_attributes[0].model_dump(),
            "facial_features": facial_attributes[1].model_dump(),
            "deepface_analysis": facial_attributes[2].model_dump(),
        }

        return profile

    def _create_vector(
        self, profile_data: Tuple[Demographics, FacialFeatures, DeepFaceAnalysis]
    ) -> List[float]:
        demographics = profile_data[0]
        facial_features = profile_data[1]
        deepface_analysis = profile_data[2]

        # Combine all features into a single vector
        profile_vector = [
            demographics.age,
            1 if demographics.gender == "Man" else 0,
            RACE_MAPPING[demographics.race],
            facial_features.facial_width_to_height_ratio,
            facial_features.left_eye_aspect_ratio,
            facial_features.right_eye_aspect_ratio,
            facial_features.mouth_aspect_ratio,
            facial_features.nose_to_chin_ratio,
            facial_features.eye_separation_ratio,
            facial_features.jaw_angle,
            facial_features.symmetry,
            deepface_analysis.face_confidence,
        ]

        return np.array(profile_vector).astype(float)

    def _normalize_vector(
        self, profile_vector: np.ndarray, min_max_values
    ) -> np.ndarray:
        normalized_vector = []
        for i, value in enumerate(profile_vector):
            key = list(min_max_values.keys())[i]
            min_value, max_value = min_max_values[key]
            normalized_value = (value - min_value) / (max_value - min_value)
            normalized_vector.append(normalized_value)
        return np.array(normalized_vector)

    def verify_target_image(
        self, ref_img_array: np.ndarray, target_img_array: np.ndarray
    ):
        """Compare features of reference"""
        ref_profile = self.analyze_face(ref_img_array)
        target_profile = self.analyze_face(target_img_array)

        ref_profile_vector = self._create_vector(ref_profile)
        target_profile_vector = self._create_vector(target_profile)

        ref_profile_vector_normalized = self._normalize_vector(
            ref_profile_vector, PREDEFINED_MIN_MAX
        )
        target_profile_vector_normalized = self._normalize_vector(
            target_profile_vector, PREDEFINED_MIN_MAX
        )

        calculated_score = self._calculate_distance(
            ref_profile_vector_normalized, target_profile_vector_normalized
        )

        # Use DeepFace to verify the faces for further validation
        # in addition to verifying with extracted facial attributes
        predicted_verification_result = DeepFace.verify(
            ref_img_array,
            target_img_array,
        )
        predicted_verified = predicted_verification_result["verified"]
        predicted_score = predicted_verification_result["distance"]
        is_real = all([predicted_verified, predicted_score < 0.5])

        return predicted_score, calculated_score, is_real
