# Uniface C++

C++ implementation of the Uniface face analysis library.

## Features

- **Face Detection** - RetinaFace detector with 5-point landmarks

## Requirements

- C++17 compiler
- CMake 3.14+
- OpenCV 4.x

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Image Detection

```bash
./examples/detect <model_path> <image_path>
```

### Webcam Demo

```bash
./examples/webcam <model_path> [camera_id]
```

### Code Example

```cpp
#include <uniface/uniface.hpp>
#include <opencv2/highgui.hpp>

int main() {
    uniface::RetinaFace detector("retinaface.onnx");

    cv::Mat image = cv::imread("photo.jpg");
    auto faces = detector.detect(image);

    for (const auto& face : faces) {
        cv::rectangle(image, face.bbox, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("result.jpg", image);
    return 0;
}
```

## Models

Download models from the main uniface repository or use:

```bash
# RetinaFace MobileNet V2
wget https://github.com/your-repo/uniface/releases/download/v1.0/retinaface_mv2.onnx -P models/
```

## License

Same license as the main uniface project.
