#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    // 必要に応じてメッセージをフィルタリング
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
} g_logger;
// ランタイムを保持するグローバル変数（寿命を延ばすため）
std::shared_ptr<nvinfer1::IRuntime> g_runtime;

// 独自のデリーター構造体
struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

// Define a unique pointer template
template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

// Manage CUDA memory allocation and deallocation
class CudaMemory {
 public:
  CudaMemory() : devicePtr_(nullptr), size_(0) {}

  // Allocate CUDA memory with specified size
  bool Allocate(size_t bytes) {
    if (device_ptr_) {
      cudaFree(device_ptr_);
      device_ptr_ = nullptr;
    }
    size_ = bytes;
    cudaError_t err = cudaMalloc(&device_ptr_, size_);
    return err == cudaSuccess;
  }

  // Copy data from host to device
  bool CopyFromHost(const void* host_data, size_t bytes) {
    if (!device_ptr_ || bytes > size_) return false;
    cudaError_t err =
        cudaMemcpy(device_ptr_, host_data, bytes, cudaMemcpyHostToDevice);
    return err == cudaSuccess;
  }

  // Copy data from device to host
  bool CopyToHost(void* host_data, size_t bytes) const {
    if (!device_ptr_ || bytes > size_) return false;
    cudaError_t err =
        cudaMemcpy(host_data, device_ptr_, bytes, cudaMemcpyDeviceToHost);
    return err == cudaSuccess;
  }

  // Get device pointer
  void* Get() const { return device_ptr_; }

  // Destructor to release memory
  ~CudaMemory() {
    if (device_ptr_) {
      cudaFree(device_ptr_);
    }
  }

 private:
  void* device_ptr_;
  size_t size_;
};

// ファイルを開く関数
bool OpenEngineFile(const std::string& engine_path, std::ifstream& engine_file) {
  engine_file.open(engine_path, std::ios::binary);
  if (!engine_file) {
    std::cerr << "エンジンファイルを開けませんでした: " << engine_path
              << std::endl;
    return false;
  }
  return true;
}

// ファイルサイズを取得する関数
size_t GetFileSize(std::ifstream& engine_file) {
  // ファイルの末尾に移動
  engine_file.seekg(0, std::ios::end);
  const size_t engine_size = engine_file.tellg();
  // 読み取り位置を先頭に戻す
  engine_file.seekg(0, std::ios::beg);
  return engine_size;
}

// ファイル内容をメモリにロードする関数
bool LoadEngineToMemory(std::ifstream& engine_file,
                        const std::string& engine_path,
                        std::vector<char>& engine_data) {
  const size_t engine_size = GetFileSize(engine_file);

  // メモリを確保してファイル内容を読み込む
  engine_data.resize(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  if (!engine_file) {
    std::cerr << "エンジンファイルの読み込みに失敗しました: " << engine_path
              << std::endl;
    return false;
  }

  return true;
}

// TensorRTエンジンをメモリブロックからデシリアライズする関数
std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(
    const std::vector<char>& engine_data, size_t engine_size,
    nvinfer1::ILogger& logger) {
  // グローバルなランタイムがまだ作成されていない場合は作成
  if (!g_runtime) {
    g_runtime.reset(nvinfer1::createInferRuntime(logger), InferDeleter());
  }
  nvinfer1::ICudaEngine* raw_engine =
      g_runtime->deserializeCudaEngine(engine_data.data(), engine_size);
  if (!raw_engine) {
    return nullptr;
  }
  return std::shared_ptr<nvinfer1::ICudaEngine>(raw_engine, InferDeleter());
}

// 画像の前処理関数
cv::Mat Preprocess(const cv::Mat& image, int target_width, int target_height) {
  cv::Mat resized, float_image;
  cv::resize(image, resized, cv::Size(target_width, target_height));
  // BGR -> RGB変換
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  // [0,255] -> [0,1] のfloat32に変換
  resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
  // 正規化（ImageNetの平均と標準偏差を使用）
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);

  // ImageNet平均値: [0.485, 0.456, 0.406], 標準偏差: [0.229, 0.224, 0.225]
  channels[0] = (channels[0] - 0.485) / 0.229;  // R
  channels[1] = (channels[1] - 0.456) / 0.224;  // G
  channels[2] = (channels[2] - 0.406) / 0.225;  // B

  cv::merge(channels, float_image);

  // NCHW形式に変換 (バッチサイズ1)
  cv::Mat blob(1, 3 * target_height * target_width, CV_32FC1);
  float* blob_data = blob.ptr<float>();

  // HWCからNCHWへの変換
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < target_height; ++h) {
      for (int w = 0; w < target_width; ++w) {
        blob_data[c * target_height * target_width + h * target_width + w] =
            float_image.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
  std::cout << "blobの行列形状: [" << blob.rows << ", " << blob.cols << "]"
            << std::endl;
  return blob;
}

// ImageNetクラスラベルを読み込む関数
bool LoadImageNetLabels(const std::string& label_file,
                        std::vector<std::string>& labels) {
  std::ifstream file(label_file);
  if (!file.is_open()) {
    std::cerr << "Failed to open label file: " << label_file << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return !labels.empty();
}

// 上位N個の予測結果を取得する関数
std::vector<std::pair<int, float>> GetTopPredictions(const float* scores,
                                                     int num_classes, int top_n) {
  std::vector<std::pair<int, float>> indexed_scores;
  for (int i = 0; i < num_classes; ++i) {
    indexed_scores.push_back(std::make_pair(i, scores[i]));
  }

  // スコアの降順でソート
  std::partial_sort(
      indexed_scores.begin(), indexed_scores.begin() + top_n, indexed_scores.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });

  // 上位N個の結果を返す
  return std::vector<std::pair<int, float>>(indexed_scores.begin(),
                                            indexed_scores.begin() + top_n);
}

// 画像を読み込む関数
cv::Mat LoadImage(const std::string& image_path) {
  // 画像の読み込み
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "画像の読み込みに失敗しました: " << image_path << std::endl;
    return cv::Mat();
  }
  return image;
}

int main() {
  std::string engine_path = "resnet50.engine";
  std::string image_path = "n01440764_10365.jpeg";
  std::string label_path = "imagenet_labels.txt";

  // Load labels
  std::vector<std::string> labels;
  if (!LoadImageNetLabels(label_path, labels)) {
    std::cerr << "Failed to load labels" << std::endl;
    // ラベルが読み込めなくても処理は続行する
  }

  std::ifstream engine_file;
  std::vector<char> engine_data;

  // Open file
  if (!OpenEngineFile(engine_path, engine_file)) {
    return 1;
  }

  // Load file content to memory
  if (!LoadEngineToMemory(engine_file, engine_path, engine_data)) {
    return 1;
  }

  // Deserialize engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine =
      DeserializeEngine(engine_data, engine_data.size(), g_logger);
  if (!engine) {
    std::cerr << "Deserialize engine failed" << std::endl;
    return 1;
  }

  // Create execution context
  UniquePtr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  if (!context) {
    std::cerr << "Create execution context failed" << std::endl;
    return 1;
  }

  // Load image
  cv::Mat image = LoadImage(image_path);
  if (image.empty()) {
    return 1;
  }

  // Preprocess image
  int input_h = 224;  // ResNet-50の標準入力サイズ
  int input_w = 224;
  cv::Mat preprocessed_image = Preprocess(image, input_w, input_h);

  // Get input binding index
  int input_index = engine->getBindingIndex("input");
  int output_index = engine->getBindingIndex("output");

  if (input_index == -1 || output_index == -1) {
    std::cerr << "Binding name not found. Check the model's input and output "
                 "layer names."
              << std::endl;
    // インデックス直指定
    // input_index = 0;
    // output_index = 1;
  }

  // Batch size, number of channels, height, width
  nvinfer1::Dims input_dims = engine->getBindingDimensions(input_index);
  nvinfer1::Dims output_dims = engine->getBindingDimensions(output_index);

  int input_size = 1;  // Batch size = 1
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_size *= input_dims.d[i];
  }

  int output_size = 1;
  for (int i = 0; i < output_dims.nbDims; i++) {
    output_size *= output_dims.d[i];
  }
  std::cout << "input_size: " << input_size << std::endl;
  std::cout << "output_size: " << output_size << std::endl;

  // Allocate memory on CPU
  std::vector<float> input_data(input_size);
  std::vector<float> output_data(output_size);

  // Preprocess image data to input_data
  std::memcpy(input_data.data(), preprocessed_image.ptr<float>(),
              input_size * sizeof(float));

  // Allocate memory on GPU
  CudaMemory input_buffer, output_buffer;
  if (!input_buffer.Allocate(input_size * sizeof(float)) ||
      !output_buffer.Allocate(output_size * sizeof(float))) {
    std::cerr << "CUDA memory allocation failed" << std::endl;
    return 1;
  }

  // Host to Device
  if (!input_buffer.CopyFromHost(input_data.data(), input_size * sizeof(float))) {
    std::cerr << "Copy input data to GPU failed" << std::endl;
    return 1;
  }

  // Set binding pointers
  void* bindings[2] = {input_buffer.Get(), output_buffer.Get()};

  bool status = context->executeV2(bindings);
  if (!status) {
    return 1;
  }

  // Device to Host
  if (!output_buffer.CopyToHost(output_data.data(), output_size * sizeof(float))) {
    std::cerr << "Copy output data to CPU failed" << std::endl;
    return 1;
  }

  // Get top 5 predictions
  int top_n = 5;
  auto top_predictions = GetTopPredictions(output_data.data(), output_size, top_n);

  // Result
  std::cout << "Top " << top_n << " predictions:" << std::endl;
  for (int i = 0; i < top_n; ++i) {
    int class_id = top_predictions[i].first;
    float score = top_predictions[i].second;
    std::string label = (class_id < labels.size()) ? labels[class_id] : "Unknown";
    std::cout << i + 1 << ". " << label << " (" << class_id
              << "): " << score * 100.0f << "%" << std::endl;
  }

  return 0;
}