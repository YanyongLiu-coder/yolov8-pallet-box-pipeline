cmake_minimum_required(VERSION 3.18)
project(yolo_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime (installed at /usr/local or /opt/onnxruntime)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/local/include/onnxruntime /opt/onnxruntime/include /usr/include/onnxruntime)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/local/lib /opt/onnxruntime/lib /usr/lib/x86_64-linux-gnu)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIB.")
endif()

# cpp-httplib (header only, downloaded in Dockerfile)
# nlohmann/json (header only, downloaded in Dockerfile)

add_executable(yolo_server main.cpp)
target_include_directories(yolo_server PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)
target_link_libraries(yolo_server PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    pthread
)
