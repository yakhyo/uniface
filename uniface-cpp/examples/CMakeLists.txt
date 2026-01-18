# Examples

find_package(OpenCV REQUIRED COMPONENTS highgui imgcodecs videoio)

# Image detection example
add_executable(detect detect.cpp)
target_link_libraries(detect PRIVATE uniface ${OpenCV_LIBS})

# Webcam example
add_executable(webcam webcam.cpp)
target_link_libraries(webcam PRIVATE uniface ${OpenCV_LIBS})
