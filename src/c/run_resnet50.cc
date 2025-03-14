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
} gLogger;
// ランタイムを保持するグローバル変数（寿命を延ばすため）
std::shared_ptr<nvinfer1::IRuntime> gRuntime;

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
    if (devicePtr_) {
      cudaFree(devicePtr_);
      devicePtr_ = nullptr;
    }
    size_ = bytes;
    cudaError_t err = cudaMalloc(&devicePtr_, size_);
    return err == cudaSuccess;
  }

  // Copy data from host to device
  bool CopyFromHost(const void* hostData, size_t bytes) {
    if (!devicePtr_ || bytes > size_) return false;
    cudaError_t err =
        cudaMemcpy(devicePtr_, hostData, bytes, cudaMemcpyHostToDevice);
    return err == cudaSuccess;
  }

  // Copy data from device to host
  bool CopyToHost(void* hostData, size_t bytes) const {
    if (!devicePtr_ || bytes > size_) return false;
    cudaError_t err =
        cudaMemcpy(hostData, devicePtr_, bytes, cudaMemcpyDeviceToHost);
    return err == cudaSuccess;
  }

  // Get device pointer
  void* Get() const { return devicePtr_; }

  // Destructor to release memory
  ~CudaMemory() {
    if (devicePtr_) {
      cudaFree(devicePtr_);
    }
  }

 private:
  void* devicePtr_;
  size_t size_;
};

// ファイルを開く関数
bool OpenEngineFile(const std::string& engine_path, std::ifstream& engineFile) {
  engineFile.open(engine_path, std::ios::binary);
  if (!engineFile) {
    std::cerr << "エンジンファイルを開けませんでした: " << engine_path
              << std::endl;
    return false;
  }
  return true;
}

// ファイルサイズを取得する関数
size_t GetFileSize(std::ifstream& engineFile) {
  // ファイルの末尾に移動
  engineFile.seekg(0, std::ios::end);
  const size_t engineSize = engineFile.tellg();
  // 読み取り位置を先頭に戻す
  engineFile.seekg(0, std::ios::beg);
  return engineSize;
}

// ファイル内容をメモリにロードする関数
bool LoadEngineToMemory(std::ifstream& engineFile,
                        const std::string& engine_path,
                        std::vector<char>& engineData) {
  const size_t engineSize = GetFileSize(engineFile);

  // メモリを確保してファイル内容を読み込む
  engineData.resize(engineSize);
  engineFile.read(engineData.data(), engineSize);

  if (!engineFile) {
    std::cerr << "エンジンファイルの読み込みに失敗しました: " << engine_path
              << std::endl;
    return false;
  }

  return true;
}

// TensorRTエンジンをメモリブロックからデシリアライズする関数
std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(
    const std::vector<char>& engineData, size_t engineSize,
    nvinfer1::ILogger& logger) {
  // グローバルなランタイムがまだ作成されていない場合は作成
  if (!gRuntime) {
    gRuntime.reset(nvinfer1::createInferRuntime(logger), InferDeleter());
  }
  nvinfer1::ICudaEngine* rawEngine =
      gRuntime->deserializeCudaEngine(engineData.data(), engineSize);
  if (!rawEngine) {
    return nullptr;
  }
  return std::shared_ptr<nvinfer1::ICudaEngine>(rawEngine, InferDeleter());
}

// 画像の前処理関数
cv::Mat Preprocess(const cv::Mat& image, int targetWidth, int targetHeight) {
  cv::Mat resized, floatImage;
  cv::resize(image, resized, cv::Size(targetWidth, targetHeight));
  // BGR -> RGB変換
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  // [0,255] -> [0,1] のfloat32に変換
  resized.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
  // 正規化（ImageNetの平均と標準偏差を使用）
  std::vector<cv::Mat> channels(3);
  cv::split(floatImage, channels);

  // ImageNet平均値: [0.485, 0.456, 0.406], 標準偏差: [0.229, 0.224, 0.225]
  channels[0] = (channels[0] - 0.485) / 0.229;  // R
  channels[1] = (channels[1] - 0.456) / 0.224;  // G
  channels[2] = (channels[2] - 0.406) / 0.225;  // B

  cv::merge(channels, floatImage);

  // NCHW形式に変換 (バッチサイズ1)
  cv::Mat blob(1, 3 * targetHeight * targetWidth, CV_32FC1);
  float* blobData = blob.ptr<float>();

  // HWCからNCHWへの変換
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < targetHeight; ++h) {
      for (int w = 0; w < targetWidth; ++w) {
        blobData[c * targetHeight * targetWidth + h * targetWidth + w] =
            floatImage.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
  std::cout << "blobの行列形状: [" << blob.rows << ", " << blob.cols << "]"
            << std::endl;
  return blob;
}

// ImageNetクラスラベルを読み込む関数
bool LoadImageNetLabels(const std::string& labelFile,
                        std::vector<std::string>& labels) {
  std::ifstream file(labelFile);
  if (!file.is_open()) {
    std::cerr << "Failed to open label file: " << labelFile << std::endl;
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
                                                     int numClasses, int topN) {
  std::vector<std::pair<int, float>> indexedScores;
  for (int i = 0; i < numClasses; ++i) {
    indexedScores.push_back(std::make_pair(i, scores[i]));
  }

  // スコアの降順でソート
  std::partial_sort(
      indexedScores.begin(), indexedScores.begin() + topN, indexedScores.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });

  // 上位N個の結果を返す
  return std::vector<std::pair<int, float>>(indexedScores.begin(),
                                            indexedScores.begin() + topN);
}

// 画像を読み込む関数
cv::Mat LoadImage(const std::string& imagePath) {
  // 画像の読み込み
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "画像の読み込みに失敗しました: " << imagePath << std::endl;
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

  std::ifstream engineFile;
  std::vector<char> engineData;

  // Open file
  if (!OpenEngineFile(engine_path, engineFile)) {
    return 1;
  }

  // Load file content to memory
  if (!LoadEngineToMemory(engineFile, engine_path, engineData)) {
    return 1;
  }

  // Deserialize engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine =
      DeserializeEngine(engineData, engineData.size(), gLogger);
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
  int inputH = 224;  // ResNet-50の標準入力サイズ
  int inputW = 224;
  cv::Mat preprocessedImage = Preprocess(image, inputW, inputH);

  // Get input binding index
  int inputIndex = engine->getBindingIndex("input");
  int outputIndex = engine->getBindingIndex("output");

  if (inputIndex == -1 || outputIndex == -1) {
    std::cerr << "Binding name not found. Check the model's input and output "
                 "layer names."
              << std::endl;
    // インデックス直指定
    // inputIndex = 0;
    // outputIndex = 1;
  }

  // Batch size, number of channels, height, width
  nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
  nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

  int inputSize = 1;  // Batch size = 1
  for (int i = 0; i < inputDims.nbDims; i++) {
    inputSize *= inputDims.d[i];
  }

  int outputSize = 1;
  for (int i = 0; i < outputDims.nbDims; i++) {
    outputSize *= outputDims.d[i];
  }
  std::cout << "inputSize: " << inputSize << std::endl;
  std::cout << "outputSize: " << outputSize << std::endl;

  // Allocate memory on CPU
  std::vector<float> inputData(inputSize);
  std::vector<float> outputData(outputSize);

  // Preprocess image data to inputData
  std::memcpy(inputData.data(), preprocessedImage.ptr<float>(),
              inputSize * sizeof(float));

  // Allocate memory on GPU
  CudaMemory inputBuffer, outputBuffer;
  if (!inputBuffer.Allocate(inputSize * sizeof(float)) ||
      !outputBuffer.Allocate(outputSize * sizeof(float))) {
    std::cerr << "CUDA memory allocation failed" << std::endl;
    return 1;
  }

  // Host to Device
  if (!inputBuffer.CopyFromHost(inputData.data(), inputSize * sizeof(float))) {
    std::cerr << "Copy input data to GPU failed" << std::endl;
    return 1;
  }

  // Set binding pointers
  void* bindings[2] = {inputBuffer.Get(), outputBuffer.Get()};

  bool status = context->executeV2(bindings);
  if (!status) {
    return 1;
  }

  // Device to Host
  if (!outputBuffer.CopyToHost(outputData.data(), outputSize * sizeof(float))) {
    std::cerr << "Copy output data to CPU failed" << std::endl;
    return 1;
  }

  // Get top 5 predictions
  int topN = 5;
  auto topPredictions = GetTopPredictions(outputData.data(), outputSize, topN);

  // Result
  std::cout << "Top " << topN << " predictions:" << std::endl;
  for (int i = 0; i < topN; ++i) {
    int classId = topPredictions[i].first;
    float score = topPredictions[i].second;
    std::string label = (classId < labels.size()) ? labels[classId] : "Unknown";
    std::cout << i + 1 << ". " << label << " (" << classId
              << "): " << score * 100.0f << "%" << std::endl;
  }

  return 0;
}