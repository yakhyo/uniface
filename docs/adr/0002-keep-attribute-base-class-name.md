# Keep `Attribute` as the per-face predictor base class name

During the v4.0.0 unified-results design we planned to rename the `Attribute` base class to `FacePredictor` (and `FaceAnalyzer(attributes=...)` to `predictors=`), because future per-face models like head pose and gaze are transient geometric state, not attributes. We decided to keep the v3 names: the rename would break `Attribute` subclasses and every `FaceAnalyzer(attributes=...)` caller for a purely semantic gain, and precedent shows the stretch is tolerable (DeepFace runs everything through `actions=`; InsightFace runs landmark and pose models through the same loop as its `Attribute` class).

Consequences: when head pose, gaze, spoofing, or quality predictors join the analyzer pipeline, they subclass `Attribute` despite the name. "Face predictor" remains the conceptual term in CONTEXT.md for what this seam holds. Reviews should not re-propose the rename without new evidence of real user confusion.
