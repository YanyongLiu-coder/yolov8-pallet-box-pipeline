#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using nvinfer1::Dims;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IRuntime;
using nvinfer1::ILogger;

class TrtLogger final : public ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << msg << '\n';
    }
  }
};

struct Args {
  std::string engine_path;
  std::string image_path;
  std::string labels_path = "assets/pallet_box.names";
  std::string output_path = "outputs/result.jpg";
  float conf_threshold = 0.25f;
  float iou_threshold = 0.45f;
};

struct LetterboxInfo {
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
};

struct Detection {
  cv::Rect box;
  int class_id = -1;
  float score = 0.0f;
};

template <typename T>
struct TrtDestroy {
  void operator()(T* obj) const {
    if (obj != nullptr) {
      obj->destroy();
    }
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroy<T>>;

void checkCuda(cudaError_t status, const std::string& message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(message + ": " + cudaGetErrorString(status));
  }
}

std::vector<char> readBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  file.seekg(0, std::ios::end);
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Cannot read file: " + path);
  }
  return buffer;
}

std::vector<std::string> readLabels(const std::string& path) {
  std::ifstream file(path);
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

int64_t volume(const Dims& dims) {
  int64_t value = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    value *= dims.d[i];
  }
  return value;
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto needValue = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };
    if (key == "--engine") args.engine_path = needValue(key);
    else if (key == "--image") args.image_path = needValue(key);
    else if (key == "--labels") args.labels_path = needValue(key);
    else if (key == "--output") args.output_path = needValue(key);
    else if (key == "--conf") args.conf_threshold = std::stof(needValue(key));
    else if (key == "--iou") args.iou_threshold = std::stof(needValue(key));
    else throw std::runtime_error("Unknown argument: " + key);
  }
  if (args.engine_path.empty() || args.image_path.empty()) {
    throw std::runtime_error("Usage: yolo_trt_infer --engine model.engine --image image.jpg [--output result.jpg]");
  }
  return args;
}

std::vector<float> preprocess(const cv::Mat& image, int input_h, int input_w, LetterboxInfo& info) {
  const float scale = std::min(static_cast<float>(input_w) / image.cols, static_cast<float>(input_h) / image.rows);
  const int resized_w = static_cast<int>(std::round(image.cols * scale));
  const int resized_h = static_cast<int>(std::round(image.rows * scale));
  info.scale = scale;
  info.pad_x = (input_w - resized_w) / 2;
  info.pad_y = (input_h - resized_h) / 2;

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(resized_w, resized_h));

  cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, resized_w, resized_h)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  cv::split(canvas, channels);

  std::vector<float> chw(3 * input_h * input_w);
  const int channel_size = input_h * input_w;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(chw.data() + c * channel_size, channels[c].data, channel_size * sizeof(float));
  }
  return chw;
}

float intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
  const int area_intersection = (a & b).area();
  const int area_union = a.area() + b.area() - area_intersection;
  return area_union > 0 ? static_cast<float>(area_intersection) / area_union : 0.0f;
}

std::vector<Detection> nms(std::vector<Detection> detections, float iou_threshold) {
  std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
    return a.score > b.score;
  });

  std::vector<Detection> kept;
  std::vector<bool> removed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (removed[i]) continue;
    kept.push_back(detections[i]);
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (!removed[j] && detections[i].class_id == detections[j].class_id &&
          intersectionOverUnion(detections[i].box, detections[j].box) > iou_threshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

std::vector<Detection> decodeYolov8(
    const std::vector<float>& output,
    const Dims& output_dims,
    const cv::Size& original_size,
    const LetterboxInfo& letterbox,
    float conf_threshold,
    float iou_threshold) {
  int channels = 0;
  int anchors = 0;

  if (output_dims.nbDims == 3) {
    channels = output_dims.d[1];
    anchors = output_dims.d[2];
  } else if (output_dims.nbDims == 2) {
    channels = output_dims.d[0];
    anchors = output_dims.d[1];
  } else {
    throw std::runtime_error("Unsupported YOLO output dimensions.");
  }

  const int num_classes = channels - 4;
  std::vector<Detection> detections;

  for (int anchor = 0; anchor < anchors; ++anchor) {
    float best_score = 0.0f;
    int best_class = -1;
    for (int cls = 0; cls < num_classes; ++cls) {
      const float score = output[(4 + cls) * anchors + anchor];
      if (score > best_score) {
        best_score = score;
        best_class = cls;
      }
    }
    if (best_score < conf_threshold) {
      continue;
    }

    const float cx = output[0 * anchors + anchor];
    const float cy = output[1 * anchors + anchor];
    const float w = output[2 * anchors + anchor];
    const float h = output[3 * anchors + anchor];

    const float x1 = (cx - w / 2.0f - letterbox.pad_x) / letterbox.scale;
    const float y1 = (cy - h / 2.0f - letterbox.pad_y) / letterbox.scale;
    const float x2 = (cx + w / 2.0f - letterbox.pad_x) / letterbox.scale;
    const float y2 = (cy + h / 2.0f - letterbox.pad_y) / letterbox.scale;

    const int left = std::clamp(static_cast<int>(std::round(x1)), 0, original_size.width - 1);
    const int top = std::clamp(static_cast<int>(std::round(y1)), 0, original_size.height - 1);
    const int right = std::clamp(static_cast<int>(std::round(x2)), 0, original_size.width - 1);
    const int bottom = std::clamp(static_cast<int>(std::round(y2)), 0, original_size.height - 1);

    if (right > left && bottom > top) {
      detections.push_back({cv::Rect(cv::Point(left, top), cv::Point(right, bottom)), best_class, best_score});
    }
  }

  return nms(std::move(detections), iou_threshold);
}

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    const std::vector<std::string> labels = readLabels(args.labels_path);

    TrtLogger logger;
    const std::vector<char> engine_data = readBinaryFile(args.engine_path);
    TrtUniquePtr<IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    TrtUniquePtr<ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    TrtUniquePtr<IExecutionContext> context(engine->createExecutionContext());

    if (!engine || !context) {
      throw std::runtime_error("Failed to create TensorRT runtime, engine, or context.");
    }

    int input_index = -1;
    int output_index = -1;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      if (engine->bindingIsInput(i)) input_index = i;
      else output_index = i;
    }
    if (input_index < 0 || output_index < 0) {
      throw std::runtime_error("Failed to find input/output bindings.");
    }

    const Dims input_dims = engine->getBindingDimensions(input_index);
    const Dims output_dims = engine->getBindingDimensions(output_index);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3) {
      throw std::runtime_error("This sample expects static NCHW input with shape [1, 3, H, W].");
    }

    const int input_h = input_dims.d[2];
    const int input_w = input_dims.d[3];

    cv::Mat image = cv::imread(args.image_path);
    if (image.empty()) {
      throw std::runtime_error("Cannot read image: " + args.image_path);
    }

    LetterboxInfo letterbox;
    std::vector<float> input = preprocess(image, input_h, input_w, letterbox);
    std::vector<float> output(volume(output_dims));

    void* bindings[2] = {nullptr, nullptr};
    checkCuda(cudaMalloc(&bindings[input_index], input.size() * sizeof(float)), "cudaMalloc input");
    checkCuda(cudaMalloc(&bindings[output_index], output.size() * sizeof(float)), "cudaMalloc output");

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    checkCuda(cudaMemcpyAsync(bindings[input_index], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
              "copy input to device");

    if (!context->enqueueV2(bindings, stream, nullptr)) {
      throw std::runtime_error("TensorRT enqueueV2 failed.");
    }

    checkCuda(cudaMemcpyAsync(output.data(), bindings[output_index], output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "copy output to host");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    checkCuda(cudaFree(bindings[input_index]), "cudaFree input");
    checkCuda(cudaFree(bindings[output_index]), "cudaFree output");

    const auto detections = decodeYolov8(output, output_dims, image.size(), letterbox, args.conf_threshold, args.iou_threshold);

    for (const auto& det : detections) {
      const std::string label = det.class_id >= 0 && det.class_id < static_cast<int>(labels.size())
                                  ? labels[det.class_id]
                                  : std::to_string(det.class_id);
      std::cout << label << " score=" << det.score << " box=["
                << det.box.x << "," << det.box.y << ","
                << det.box.width << "," << det.box.height << "]\n";

      cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
      cv::putText(image, label + " " + cv::format("%.2f", det.score),
                  cv::Point(det.box.x, std::max(0, det.box.y - 5)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite(args.output_path, image);
    std::cout << "Detections: " << detections.size() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
