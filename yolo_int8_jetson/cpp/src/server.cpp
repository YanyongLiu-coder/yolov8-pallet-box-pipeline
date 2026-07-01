/**
 * YOLOv8 TensorRT HTTP inference server.
 *
 * Reuses the same TensorRT inference logic from main.cpp but wraps it
 * in an HTTP server (cpp-httplib) for RESTful access.
 *
 * Endpoint:  POST /detect   (multipart form, field "file")
 * Response:  JSON with image_name, num_boxes, boxes[], timing_ms
 */

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;
using nvinfer1::Dims;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IRuntime;
using nvinfer1::ILogger;

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
class TrtLogger final : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << msg << '\n';
    }
};

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------
struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection {
    float x1, y1, x2, y2;
    int class_id;
    float score;
};

// ---------------------------------------------------------------------------
// TensorRT helpers
// ---------------------------------------------------------------------------
template <typename T>
struct TrtDestroy {
    void operator()(T* obj) const { if (obj) obj->destroy(); }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroy<T>>;

static void checkCuda(cudaError_t s, const std::string& msg) {
    if (s != cudaSuccess) throw std::runtime_error(msg + ": " + cudaGetErrorString(s));
}

static std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    std::vector<char> buf(f.tellg());
    f.seekg(0);
    f.read(buf.data(), buf.size());
    return buf;
}

static int64_t volume(const Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------
static std::vector<float> preprocess(const cv::Mat& img, int h, int w, LetterboxInfo& lb) {
    float scale = std::min((float)w / img.cols, (float)h / img.rows);
    int rw = (int)std::round(img.cols * scale);
    int rh = (int)std::round(img.rows * scale);
    lb.scale = scale;
    lb.pad_x = (w - rw) / 2;
    lb.pad_y = (h - rh) / 2;

    cv::Mat resized, canvas(h, w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(img, resized, cv::Size(rw, rh));
    resized.copyTo(canvas(cv::Rect(lb.pad_x, lb.pad_y, rw, rh)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> ch(3);
    cv::split(canvas, ch);
    std::vector<float> chw(3 * h * w);
    int cs = h * w;
    for (int c = 0; c < 3; ++c)
        std::memcpy(chw.data() + c * cs, ch[c].data, cs * sizeof(float));
    return chw;
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------
static float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1);
    float ub = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (ua + ub - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets, float thresh) {
    std::sort(dets.begin(), dets.end(), [](auto& a, auto& b) { return a.score > b.score; });
    std::vector<bool> sup(dets.size(), false);
    std::vector<Detection> out;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!sup[j] && iou(dets[i], dets[j]) > thresh) sup[j] = true;
    }
    return out;
}

// ---------------------------------------------------------------------------
// YOLOv8 output decode
// ---------------------------------------------------------------------------
static std::vector<Detection> decode(const std::vector<float>& out, const Dims& dims,
                                      int img_w, int img_h, const LetterboxInfo& lb,
                                      float conf_th, float iou_th) {
    int channels = dims.d[1];  // 5 for 1 class
    int anchors = dims.d[2];   // 8400
    int num_cls = channels - 4;

    std::vector<Detection> dets;
    for (int a = 0; a < anchors; ++a) {
        float best = 0; int best_c = 0;
        for (int c = 0; c < num_cls; ++c) {
            float s = out[(4 + c) * anchors + a];
            if (s > best) { best = s; best_c = c; }
        }
        if (best < conf_th) continue;
        float cx = out[0 * anchors + a], cy = out[1 * anchors + a];
        float w = out[2 * anchors + a], h = out[3 * anchors + a];
        float x1 = std::clamp((cx - w / 2 - lb.pad_x) / lb.scale, 0.f, (float)img_w);
        float y1 = std::clamp((cy - h / 2 - lb.pad_y) / lb.scale, 0.f, (float)img_h);
        float x2 = std::clamp((cx + w / 2 - lb.pad_x) / lb.scale, 0.f, (float)img_w);
        float y2 = std::clamp((cy + h / 2 - lb.pad_y) / lb.scale, 0.f, (float)img_h);
        if (x2 > x1 && y2 > y1) dets.push_back({x1, y1, x2, y2, best_c, best});
    }
    return nms(dets, iou_th);
}

// ---------------------------------------------------------------------------
// TensorRT Detector
// ---------------------------------------------------------------------------
class TrtDetector {
public:
    TrtDetector(const std::string& engine_path) {
        auto data = readBinaryFile(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        context_.reset(engine_->createExecutionContext());

        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) in_idx_ = i;
            else out_idx_ = i;
        }
        in_dims_ = engine_->getBindingDimensions(in_idx_);
        out_dims_ = engine_->getBindingDimensions(out_idx_);
        input_h_ = in_dims_.d[2];
        input_w_ = in_dims_.d[3];

        checkCuda(cudaMalloc(&bindings_[0], volume(in_dims_) * sizeof(float)), "alloc in");
        checkCuda(cudaMalloc(&bindings_[1], volume(out_dims_) * sizeof(float)), "alloc out");
        checkCuda(cudaStreamCreate(&stream_), "stream");

        // Warmup
        cv::Mat dummy = cv::Mat::zeros(input_h_, input_w_, CV_8UC3);
        detect(dummy);
        std::cout << "TensorRT engine loaded, warmup done. Input: "
                  << input_h_ << "x" << input_w_ << std::endl;
    }

    ~TrtDetector() {
        cudaFree(bindings_[0]);
        cudaFree(bindings_[1]);
        cudaStreamDestroy(stream_);
    }

    std::vector<Detection> detect(const cv::Mat& img, float conf = 0.25f, float iou_th = 0.7f) {
        LetterboxInfo lb;
        auto input = preprocess(img, input_h_, input_w_, lb);
        std::vector<float> output(volume(out_dims_));

        checkCuda(cudaMemcpyAsync(bindings_[in_idx_], input.data(),
                  input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "h2d");
        context_->enqueueV2(bindings_, stream_, nullptr);
        checkCuda(cudaMemcpyAsync(output.data(), bindings_[out_idx_],
                  output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "d2h");
        cudaStreamSynchronize(stream_);

        return decode(output, out_dims_, img.cols, img.rows, lb, conf, iou_th);
    }

private:
    TrtLogger logger_;
    TrtUniquePtr<IRuntime> runtime_{nullptr};
    TrtUniquePtr<ICudaEngine> engine_{nullptr};
    TrtUniquePtr<IExecutionContext> context_{nullptr};
    void* bindings_[2] = {nullptr, nullptr};
    cudaStream_t stream_ = nullptr;
    int in_idx_ = 0, out_idx_ = 1;
    Dims in_dims_{}, out_dims_{};
    int input_h_ = 640, input_w_ = 640;
};

// ---------------------------------------------------------------------------
// Main: HTTP Server
// ---------------------------------------------------------------------------
static const char* CLASS_NAMES[] = {"goods_stack"};

int main(int argc, char** argv) {
    std::string engine_path = "models/best.fp16.engine";
    int port = 8002;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--engine" && i + 1 < argc) engine_path = argv[++i];
        else if (k == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    }

    std::cout << "Loading TensorRT engine: " << engine_path << std::endl;
    TrtDetector detector(engine_path);

    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json r = {{"status", "ok"}, {"engine", engine_path}, {"backend", "TensorRT FP16"}};
        res.set_content(r.dump(), "application/json");
    });

    svr.Post("/detect", [&](const httplib::Request& req, httplib::Response& res) {
        auto start = Clock::now();

        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content(json({{"error", "No file field"}}).dump(), "application/json");
            return;
        }
        const auto& file = req.get_file_value("file");
        std::vector<unsigned char> buf(file.content.begin(), file.content.end());
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty()) {
            res.status = 400;
            res.set_content(json({{"error", "Invalid image"}}).dump(), "application/json");
            return;
        }

        auto infer_start = Clock::now();
        auto dets = detector.detect(img);
        double infer_ms = std::chrono::duration<double, std::milli>(Clock::now() - infer_start).count();

        json boxes = json::array();
        for (auto& d : dets) {
            boxes.push_back({
                {"class_id", d.class_id},
                {"class_name", CLASS_NAMES[d.class_id]},
                {"confidence", std::round(d.score * 10000) / 10000.0},
                {"bbox", {{"x1", std::round(d.x1*10)/10.0}, {"y1", std::round(d.y1*10)/10.0},
                          {"x2", std::round(d.x2*10)/10.0}, {"y2", std::round(d.y2*10)/10.0}}}
            });
        }

        double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        json resp = {
            {"image_name", file.filename},
            {"image_size", {{"width", img.cols}, {"height", img.rows}}},
            {"num_boxes", (int)dets.size()},
            {"boxes", boxes},
            {"timing_ms", {{"inference", std::round(infer_ms*100)/100.0},
                           {"total", std::round(total_ms*100)/100.0}}}
        };
        res.set_content(resp.dump(), "application/json");
    });

    std::cout << "TensorRT server listening on port " << port << "..." << std::endl;
    svr.listen("0.0.0.0", port);
    return 0;
}
