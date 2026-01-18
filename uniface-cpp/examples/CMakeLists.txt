# Examples

find_package(OpenCV REQUIRED COMPONENTS highgui imgcodecs videoio)

# Image detection example
add_executable(detect detect.cpp)
target_link_libraries(detect PRIVATE uniface ${OpenCV_LIBS})

# Face recognition example
add_executable(recognize recognize.cpp)
target_link_libraries(recognize PRIVATE uniface ${OpenCV_LIBS})

# Facial landmarks example
add_executable(landmarks landmarks.cpp)
target_link_libraries(landmarks PRIVATE uniface ${OpenCV_LIBS})

# Face analyzer example
add_executable(analyzer analyzer.cpp)
target_link_libraries(analyzer PRIVATE uniface ${OpenCV_LIBS})

# Webcam example
add_executable(webcam webcam.cpp)
target_link_libraries(webcam PRIVATE uniface ${OpenCV_LIBS})
