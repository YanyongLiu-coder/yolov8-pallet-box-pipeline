/**
 * YOLOv8m ONNX Runtime inference REST API service (C++).
 *
 * Dependencies: ONNX Runtime (CUDA EP), OpenCV, cpp-httplib, nlohmann/json.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const char* MODEL_PATH = "/workspace/models/best.onnx";
static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;
static const float CONF_THRESHOLD = 0.25f;
static const float IOU_THRESHOLD = 0.7f;
static const int NUM_CLASSES = 1;
static const char* CLASS_NAMES[] = {"goods_stack"};

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------
static float iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// YOLOv8 Detector class
// ---------------------------------------------------------------------------
class YoloDetector {
public:
    YoloDetector(const std::string& model_path) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Try CUDA EP first
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolov8_server");
        session_ = Ort::Session(env_, model_path.c_str(), session_options);

        std::cout << "Model loaded: " << model_path << std::endl;
        std::cout << "Execution Provider: CUDA (GPU 0)" << std::endl;

        // Warmup
        cv::Mat dummy = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
        detect(dummy);
        std::cout << "Warmup complete." << std::endl;
    }

    std::vector<Detection> detect(const cv::Mat& image) {
        // Preprocess: resize with letterbox
        float scale;
        int pad_w, pad_h;
        cv::Mat input_image = preprocess(image, scale, pad_w, pad_h);

        // Prepare input tensor
        std::array<int64_t, 4> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
        size_t input_size = 1 * 3 * INPUT_HEIGHT * INPUT_WIDTH;
        std::vector<float> input_data(input_size);

        // HWC BGR -> CHW RGB normalized [0, 1]
        int img_area = INPUT_HEIGHT * INPUT_WIDTH;
        const unsigned char* ptr = input_image.data;
        for (int i = 0; i < img_area; ++i) {
            input_data[0 * img_area + i] = ptr[i * 3 + 2] / 255.0f;  // R
            input_data[1 * img_area + i] = ptr[i * 3 + 1] / 255.0f;  // G
            input_data[2 * img_area + i] = ptr[i * 3 + 0] / 255.0f;  // B
        }

        // Run inference
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};

        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Parse output: shape (1, 5, 8400) for 1 class
        // Row 0-3: x_center, y_center, width, height
        // Row 4: class_0 confidence
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_features = static_cast<int>(output_shape[1]);  // 5
        int num_boxes = static_cast<int>(output_shape[2]);     // 8400

        std::vector<Detection> detections;
        for (int i = 0; i < num_boxes; ++i) {
            // Find max class score
            float max_score = 0.0f;
            int max_class = 0;
            for (int c = 4; c < num_features; ++c) {
                float score = output_data[c * num_boxes + i];
                if (score > max_score) {
                    max_score = score;
                    max_class = c - 4;
                }
            }

            if (max_score < CONF_THRESHOLD) continue;

            float cx = output_data[0 * num_boxes + i];
            float cy = output_data[1 * num_boxes + i];
            float w  = output_data[2 * num_boxes + i];
            float h  = output_data[3 * num_boxes + i];

            // Convert from letterboxed coords to original image coords
            float x1 = (cx - w / 2.0f - pad_w) / scale;
            float y1 = (cy - h / 2.0f - pad_h) / scale;
            float x2 = (cx + w / 2.0f - pad_w) / scale;
            float y2 = (cy + h / 2.0f - pad_h) / scale;

            // Clip to image bounds
            x1 = std::max(0.0f, std::min(x1, (float)image.cols));
            y1 = std::max(0.0f, std::min(y1, (float)image.rows));
            x2 = std::max(0.0f, std::min(x2, (float)image.cols));
            y2 = std::max(0.0f, std::min(y2, (float)image.rows));

            detections.push_back({x1, y1, x2, y2, max_score, max_class});
        }

        // Apply NMS
        return nms(detections, IOU_THRESHOLD);
    }

private:
    cv::Mat preprocess(const cv::Mat& image, float& scale, int& pad_w, int& pad_h) {
        int img_w = image.cols;
        int img_h = image.rows;
        scale = std::min((float)INPUT_WIDTH / img_w, (float)INPUT_HEIGHT / img_h);
        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);
        pad_w = (INPUT_WIDTH - new_w) / 2;
        pad_h = (INPUT_HEIGHT - new_h) / 2;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h));

        cv::Mat padded = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
        padded.setTo(cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
        return padded;
    }

    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
};

// ---------------------------------------------------------------------------
// Main: HTTP Server
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Initializing YOLOv8m C++ inference server..." << std::endl;

    YoloDetector detector(MODEL_PATH);

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json resp = {{"status", "ok"}, {"model", MODEL_PATH}};
        res.set_content(resp.dump(), "application/json");
    });

    svr.Post("/detect", [&detector](const httplib::Request& req, httplib::Response& res) {
        auto start = Clock::now();

        // Parse multipart form data
        if (!req.has_file("file")) {
            json err = {{"error", "No file field in request"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        const auto& file = req.get_file_value("file");
        std::string filename = file.filename;

        // Decode image
        std::vector<unsigned char> buf(file.content.begin(), file.content.end());
        cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (image.empty()) {
            json err = {{"error", "Invalid image"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Inference
        auto infer_start = Clock::now();
        auto detections = detector.detect(image);
        auto infer_end = Clock::now();

        double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

        // Build response
        json boxes = json::array();
        for (const auto& det : detections) {
            boxes.push_back({
                {"class_id", det.class_id},
                {"class_name", CLASS_NAMES[det.class_id]},
                {"confidence", std::round(det.confidence * 10000) / 10000.0},
                {"bbox", {
                    {"x1", std::round(det.x1 * 10) / 10.0},
                    {"y1", std::round(det.y1 * 10) / 10.0},
                    {"x2", std::round(det.x2 * 10) / 10.0},
                    {"y2", std::round(det.y2 * 10) / 10.0},
                }},
            });
        }

        auto total_end = Clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - start).count();

        json response = {
            {"image_name", filename},
            {"image_size", {{"width", image.cols}, {"height", image.rows}}},
            {"num_boxes", (int)detections.size()},
            {"boxes", boxes},
            {"timing_ms", {
                {"inference", std::round(infer_ms * 100) / 100.0},
                {"total", std::round(total_ms * 100) / 100.0},
            }},
        };

        res.set_content(response.dump(), "application/json");
    });

    std::cout << "Server listening on port 8001..." << std::endl;
    svr.listen("0.0.0.0", 8001);
    return 0;
}
