# Parsing

Face parsing segments faces into semantic components (skin, eyes, nose, mouth, hair, etc.).

---

## Available Models

| Model | Backbone | Size | Classes | Best For |
|-------|----------|------|---------|----------|
| **BiSeNet ResNet18** ⭐ | ResNet18 | 51 MB | 19 | Balanced (recommended) |
| **BiSeNet ResNet34** | ResNet34 | 89 MB | 19 | Higher accuracy |

---

## Basic Usage

```python
import cv2
from uniface.parsing import BiSeNet
from uniface.visualization import vis_parsing_maps

# Initialize parser
parser = BiSeNet()

# Load face image (cropped)
face_image = cv2.imread("face.jpg")

# Parse face
mask = parser.parse(face_image)
print(f"Mask shape: {mask.shape}")  # (H, W)

# Visualize
face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
vis_result = vis_parsing_maps(face_rgb, mask, save_image=False)

# Save result
vis_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
cv2.imwrite("parsed.jpg", vis_bgr)
```

---

## 19 Facial Component Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Background | 10 | Ear Ring |
| 1 | Skin | 11 | Nose |
| 2 | Left Eyebrow | 12 | Mouth |
| 3 | Right Eyebrow | 13 | Upper Lip |
| 4 | Left Eye | 14 | Lower Lip |
| 5 | Right Eye | 15 | Neck |
| 6 | Eye Glasses | 16 | Neck Lace |
| 7 | Left Ear | 17 | Cloth |
| 8 | Right Ear | 18 | Hair |
| 9 | Hat | | |

---

## Model Variants

```python
from uniface.parsing import BiSeNet
from uniface.constants import ParsingWeights

# Default (ResNet18)
parser = BiSeNet()

# Higher accuracy (ResNet34)
parser = BiSeNet(model_name=ParsingWeights.RESNET34)
```

| Variant | Params | Size | Notes |
|---------|--------|------|-------|
| **RESNET18** ⭐ | 13.3M | 51 MB | Recommended |
| RESNET34 | 24.1M | 89 MB | Higher accuracy |

---

## Full Pipeline

### With Face Detection

```python
import cv2
from uniface import RetinaFace
from uniface.parsing import BiSeNet
from uniface.visualization import vis_parsing_maps

detector = RetinaFace()
parser = BiSeNet()

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for i, face in enumerate(faces):
    # Crop face
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = image[y1:y2, x1:x2]

    # Parse
    mask = parser.parse(face_crop)

    # Visualize
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    vis_result = vis_parsing_maps(face_rgb, mask, save_image=False)

    # Save
    vis_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"face_{i}_parsed.jpg", vis_bgr)
```

---

## Extract Specific Components

### Get Single Component Mask

```python
import numpy as np

# Parse face
mask = parser.parse(face_image)

# Extract specific component
SKIN = 1
HAIR = 18
LEFT_EYE = 4
RIGHT_EYE = 5

# Binary mask for skin
skin_mask = (mask == SKIN).astype(np.uint8) * 255

# Binary mask for hair
hair_mask = (mask == HAIR).astype(np.uint8) * 255

# Binary mask for eyes
eyes_mask = ((mask == LEFT_EYE) | (mask == RIGHT_EYE)).astype(np.uint8) * 255
```

### Count Pixels per Component

```python
import numpy as np

mask = parser.parse(face_image)

component_names = {
    0: 'Background', 1: 'Skin', 2: 'L-Eyebrow', 3: 'R-Eyebrow',
    4: 'L-Eye', 5: 'R-Eye', 6: 'Glasses', 7: 'L-Ear', 8: 'R-Ear',
    9: 'Hat', 10: 'Earring', 11: 'Nose', 12: 'Mouth',
    13: 'U-Lip', 14: 'L-Lip', 15: 'Neck', 16: 'Necklace',
    17: 'Cloth', 18: 'Hair'
}

for class_id in np.unique(mask):
    pixel_count = np.sum(mask == class_id)
    name = component_names.get(class_id, f'Class {class_id}')
    print(f"{name}: {pixel_count} pixels")
```

---

## Applications

### Face Makeup

Apply virtual makeup using component masks:

```python
import cv2
import numpy as np

def apply_lip_color(image, mask, color=(180, 50, 50)):
    """Apply lip color using parsing mask."""
    result = image.copy()

    # Get lip mask (upper + lower lip)
    lip_mask = ((mask == 13) | (mask == 14)).astype(np.uint8)

    # Create color overlay
    overlay = np.zeros_like(image)
    overlay[:] = color

    # Blend with original
    lip_region = cv2.bitwise_and(overlay, overlay, mask=lip_mask)
    non_lip = cv2.bitwise_and(result, result, mask=1 - lip_mask)

    # Combine with alpha blending
    alpha = 0.4
    result = cv2.addWeighted(result, 1 - alpha * lip_mask[:,:,np.newaxis] / 255,
                             lip_region, alpha, 0)

    return result.astype(np.uint8)
```

### Background Replacement

```python
def replace_background(image, mask, background):
    """Replace background using parsing mask."""
    # Create foreground mask (everything except background)
    foreground_mask = (mask != 0).astype(np.uint8)

    # Resize background to match image
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Combine
    result = image.copy()
    result[foreground_mask == 0] = background[foreground_mask == 0]

    return result
```

### Hair Segmentation

```python
def get_hair_mask(mask):
    """Extract clean hair mask."""
    hair_mask = (mask == 18).astype(np.uint8) * 255

    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)

    return hair_mask
```

---

## Visualization Options

```python
from uniface.visualization import vis_parsing_maps

# Default visualization
vis_result = vis_parsing_maps(face_rgb, mask)

# With different parameters
vis_result = vis_parsing_maps(
    face_rgb,
    mask,
    save_image=False,  # Don't save to file
)
```

---

## Factory Function

```python
from uniface import create_face_parser

parser = create_face_parser()  # Returns BiSeNet
```

---

## Next Steps

- [Gaze](gaze.md) - Gaze estimation
- [Privacy](privacy.md) - Face anonymization
- [Detection](detection.md) - Face detection
