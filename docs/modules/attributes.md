# Attributes

Facial attribute analysis for age, gender, race, and emotion detection.

---

## Available Models

| Model | Attributes | Size | Notes |
|-------|------------|------|-------|
| **AgeGender** | Age, Gender | 8 MB | Exact age prediction |
| **FairFace** | Gender, Age Group, Race | 44 MB | Balanced demographics |
| **Emotion** | 7-8 emotions | 2 MB | Requires PyTorch |

---

## AgeGender

Predicts exact age and binary gender.

### Basic Usage

```python
from uniface import RetinaFace, AgeGender

detector = RetinaFace()
age_gender = AgeGender()

faces = detector.detect(image)

for face in faces:
    result = age_gender.predict(image, face.bbox)
    print(f"Gender: {result.sex}")  # "Female" or "Male"
    print(f"Age: {result.age} years")
```

### Output

```python
# AttributeResult fields
result.gender     # 0=Female, 1=Male
result.sex        # "Female" or "Male" (property)
result.age        # int, age in years
result.age_group  # None (not provided by this model)
result.race       # None (not provided by this model)
```

---

## FairFace

Predicts gender, age group, and race with balanced demographics.

### Basic Usage

```python
from uniface import RetinaFace, FairFace

detector = RetinaFace()
fairface = FairFace()

faces = detector.detect(image)

for face in faces:
    result = fairface.predict(image, face.bbox)
    print(f"Gender: {result.sex}")
    print(f"Age Group: {result.age_group}")
    print(f"Race: {result.race}")
```

### Output

```python
# AttributeResult fields
result.gender     # 0=Female, 1=Male
result.sex        # "Female" or "Male"
result.age        # None (not provided by this model)
result.age_group  # "20-29", "30-39", etc.
result.race       # Race/ethnicity label
```

### Race Categories

| Label |
|-------|
| White |
| Black |
| Latino Hispanic |
| East Asian |
| Southeast Asian |
| Indian |
| Middle Eastern |

### Age Groups

| Group |
|-------|
| 0-2 |
| 3-9 |
| 10-19 |
| 20-29 |
| 30-39 |
| 40-49 |
| 50-59 |
| 60-69 |
| 70+ |

---

## Emotion

Predicts facial emotions. Requires PyTorch.

!!! warning "Optional Dependency"
    Emotion detection requires PyTorch. Install with:
    ```bash
    pip install torch
    ```

### Basic Usage

```python
from uniface import RetinaFace
from uniface.attribute import Emotion
from uniface.constants import DDAMFNWeights

detector = RetinaFace()
emotion = Emotion(model_weights=DDAMFNWeights.AFFECNET7)

faces = detector.detect(image)

for face in faces:
    result = emotion.predict(image, face.landmarks)
    print(f"Emotion: {result.emotion}")
    print(f"Confidence: {result.confidence:.2%}")
```

### Emotion Classes

=== "7-Class (AFFECNET7)"

    | Label |
    |-------|
    | Neutral |
    | Happy |
    | Sad |
    | Surprise |
    | Fear |
    | Disgust |
    | Anger |

=== "8-Class (AFFECNET8)"

    | Label |
    |-------|
    | Neutral |
    | Happy |
    | Sad |
    | Surprise |
    | Fear |
    | Disgust |
    | Anger |
    | Contempt |

### Model Variants

```python
from uniface.attribute import Emotion
from uniface.constants import DDAMFNWeights

# 7-class emotion
emotion = Emotion(model_weights=DDAMFNWeights.AFFECNET7)

# 8-class emotion
emotion = Emotion(model_weights=DDAMFNWeights.AFFECNET8)
```

---

## Combining Models

### Full Attribute Analysis

```python
from uniface import RetinaFace, AgeGender, FairFace

detector = RetinaFace()
age_gender = AgeGender()
fairface = FairFace()

faces = detector.detect(image)

for face in faces:
    # Get exact age from AgeGender
    ag_result = age_gender.predict(image, face.bbox)

    # Get race from FairFace
    ff_result = fairface.predict(image, face.bbox)

    print(f"Gender: {ag_result.sex}")
    print(f"Exact Age: {ag_result.age}")
    print(f"Age Group: {ff_result.age_group}")
    print(f"Race: {ff_result.race}")
```

### Using FaceAnalyzer

```python
from uniface import FaceAnalyzer, RetinaFace, AgeGender

analyzer = FaceAnalyzer(
    RetinaFace(),
    age_gender=AgeGender(),
)

faces = analyzer.analyze(image)

for face in faces:
    print(f"Age: {face.age}, Gender: {face.sex}")
```

---

## Visualization

```python
import cv2

def draw_attributes(image, face, result):
    """Draw attributes on image."""
    x1, y1, x2, y2 = map(int, face.bbox)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Build label
    label = f"{result.sex}"
    if result.age:
        label += f", {result.age}y"
    if result.age_group:
        label += f", {result.age_group}"
    if result.race:
        label += f", {result.race}"

    # Draw label
    cv2.putText(
        image, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

    return image

# Usage
for face in faces:
    result = age_gender.predict(image, face.bbox)
    image = draw_attributes(image, face, result)

cv2.imwrite("attributes.jpg", image)
```

---

## Accuracy Notes

!!! note "Model Limitations"
    - **AgeGender**: Trained on CelebA; accuracy varies by demographic
    - **FairFace**: Trained for balanced demographics; better cross-racial accuracy
    - **Emotion**: Accuracy depends on facial expression clarity

    Always test on your specific use case and consider cultural context.

---

## Next Steps

- [Parsing](parsing.md) - Face semantic segmentation
- [Gaze](gaze.md) - Gaze estimation
- [Image Pipeline Recipe](../recipes/image-pipeline.md) - Complete workflow
