# UniFace

Face analysis library: detect faces in an image, then enrich each detected face with per-face predictions (identity embedding, demography, pose, state, …) through a single pipeline.

## Language

### Geometry

**Landmarks**:
The 5-point facial set (eye centers, nose tip, mouth corners) every detector produces for every face. Used for alignment. A detection fact — always present on a Face. (Kept as the v3 name; see ADR-0001.)
_Avoid_: keypoints, kps

**Dense landmarks**:
A dense facial point set (e.g. 106, WFLW-98, 300W-68) produced by a dedicated landmark model. Optional, scheme-dependent, and never a substitute for the 5-point landmarks. Always qualified as "dense" to distinguish from the detection set.

**Scheme**:
The layout a set of dense landmarks follows (which index is which facial point). Dense landmarks from different schemes are not index-compatible.

### Pipeline

**Face**:
The record of one detected face: its detection facts plus a slot for each prediction made about it. The single landing zone for all face-level model outputs.

**Detection facts**:
What detection guarantees on every Face: bounding box, confidence, and landmarks. The only inputs a face predictor may rely on.

**Face predictor**:
A per-face model that consumes an image and a Face, does its own cropping/alignment from the detection facts, and writes its result to its own slot. Order-independent: reads only detection facts, writes only its own slot. In code, the seam is the `Attribute` base class and `FaceAnalyzer(attributes=...)` — the v3 names, kept deliberately (see ADR-0002).

**Attribute**:
An intrinsic property of a face (age, gender, emotion, eyeglasses, mask). Also the code name of the face-predictor base class, which future non-attribute predictors (head pose, gaze) will subclass despite the name — a deliberate stretch (ADR-0002).

**Slot**:
The typed optional field on Face that exactly one kind of face predictor fills. `None` means that predictor did not run.

**FaceResults**:
The per-image result: the analyzed image together with its list of Faces, and the operations that need both (plotting, cropping, serialization).
