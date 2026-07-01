cmake_minimum_required(VERSION 3.16)
project(yolo_trt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/cuda/include)
find_library(TENSORRT_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/cuda/lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT headers or libraries were not found.")
endif()

# Download header-only deps for server
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
file(MAKE_DIRECTORY ${THIRD_PARTY_DIR}/nlohmann)

if(NOT EXISTS ${THIRD_PARTY_DIR}/httplib.h)
  file(DOWNLOAD
    https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.15.3/httplib.h
    ${THIRD_PARTY_DIR}/httplib.h
    SHOW_PROGRESS)
endif()
if(NOT EXISTS ${THIRD_PARTY_DIR}/nlohmann/json.hpp)
  file(DOWNLOAD
    https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
    ${THIRD_PARTY_DIR}/nlohmann/json.hpp
    SHOW_PROGRESS)
endif()

# CLI inference tool (original)
add_executable(yolo_trt_infer src/main.cpp)
target_include_directories(yolo_trt_infer PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR})
target_link_libraries(yolo_trt_infer PRIVATE
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIB})

# HTTP server
add_executable(yolo_trt_server src/server.cpp)
target_include_directories(yolo_trt_server PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${THIRD_PARTY_DIR})
target_link_libraries(yolo_trt_server PRIVATE
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIB}
  pthread)
