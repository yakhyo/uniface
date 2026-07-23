# Keep `Face.landmarks` as the name for the 5-point detection set

During the v4.0.0 unified-results design we considered renaming `Face.landmarks` to `Face.keypoints` (MediaPipe/InsightFace-style: sparse detector output = keypoints, dense model output = landmarks), freeing `landmarks` for a future dense-landmark predictor slot. We decided against it: `Face.landmarks` is the most-accessed field in the library, and breaking every v3 caller was judged worse than the naming ambiguity — dlib and the RetinaFace paper also call the 5-point set "landmarks," so the kept name is defensible. The future dense slot will be named `Face.dense_landmarks` instead.

Consequences: "landmarks" unqualified always means the 5-point detection set; dense sets are always qualified ("dense landmarks"). Future architecture reviews should not re-propose the keypoints rename — the trade-off was considered and compatibility won.
