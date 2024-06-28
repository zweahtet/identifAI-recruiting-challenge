# These are the constants used in the project for demo.
# These values should be chosen based on domain knowledge or 
# statistical analysis of a representative dataset.

# Predefined standardized min-max values for each attribute
PREDEFINED_MIN_MAX = {
    "age": (0, 100),
    "gender": (0, 1),  # Binary encoding: Man=1, Woman=0
    "race": (0, 5),  # Simplified one-hot representation
    "facial_width_to_height_ratio": (0.5, 2.0),
    "left_eye_aspect_ratio": (0.1, 1.0),
    "right_eye_aspect_ratio": (0.1, 1.0),
    "mouth_aspect_ratio": (0.2, 1.0),
    "nose_to_chin_ratio": (0.3, 1.5),
    "eye_separation_ratio": (0.2, 1.0),
    "jaw_angle": (70, 110),
    "symmetry": (0.0, 1.0),
    "face_confidence": (0.0, 1.0),
    "gender_confidence_man": (0.0, 100.0),
    "gender_confidence_woman": (0.0, 100.0),
}

# Mapping races to standardized integer values
RACE_MAPPING = {
    "indian": 0,
    "asian": 1,
    "latino hispanic": 2,
    "black": 3,
    "middle eastern": 4,
    "white": 5,
}