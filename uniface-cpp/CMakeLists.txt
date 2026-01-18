cmake_minimum_required(VERSION 3.14)

project(uniface
    VERSION 1.0.0
    DESCRIPTION "Uniface C++ face analysis library"
    LANGUAGES CXX
)

# Options
option(UNIFACE_BUILD_EXAMPLES "Build example programs" ON)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Find dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn calib3d)

# Library
add_library(uniface
    src/utils.cpp
    src/detector.cpp
    src/recognizer.cpp
    src/landmarker.cpp
    src/analyzer.cpp
)

target_include_directories(uniface
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(uniface
    PUBLIC
        ${OpenCV_LIBS}
)

# Examples
if(UNIFACE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
