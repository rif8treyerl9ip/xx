#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>

// main.cc

// Error check macro
#define CHECK(status)                                    \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "CUDA failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

// TensorRT smart pointer definition
struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) obj->destroy();
  }
};
template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    const char* sev_str = "";
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        sev_str = "INTERNAL_ERROR";
        break;
      case Severity::kERROR:
        sev_str = "ERROR";
        break;
      case Severity::kWARNING:
        sev_str = "WARNING";
        break;
      case Severity::kINFO:
        sev_str = "INFO";
        break;
      case Severity::kVERBOSE:
        sev_str = "VERBOSE";
        break;
    }
    if (severity != Severity::kINFO && severity != Severity::kVERBOSE) {
      std::cerr << "[" << sev_str << "] " << msg << std::endl;
    }
  }
};

int main() {
  Logger logger;  // NvInfer.h

  // TensorRTプラグインの初期化
  initLibNvInferPlugins(&logger, "");

  // std::cout << "\n=== 方法1: 手動でdlopenを使用してプラグインをロード ==="
  //           << std::endl;
  // void* handle = dlopen("./libsimple_identity_plugin.so", RTLD_NOW);
  // if (!handle) {
  //   std::cerr << "dlopen Fail: " << dlerror() << std::endl;
  // }

  std::cout << "\n=== 方法2: TensorRTのAPIを使用してプラグインをロード ==="
  << std::endl; bool success =
  getPluginRegistry()->loadLibrary("./libsimple_identity_plugin.so"); if
  (!success) {
      std::cerr << "getPluginRegistry()->loadLibrary()失敗" << std::endl;
  } else {
      std::cout << "getPluginRegistry()->loadLibrary()成功" << std::endl;
  }

  // Look up the plugin in the registry
  auto creator =
      getPluginRegistry()->getPluginCreator("SimpleIdentityPlugin", "1");
  if (!creator) {
    std::cerr << "\nSimpleIdentityPlugin v1 not found" << std::endl;
    return 1;
  }

  // Create builder
  UniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "Builder creation failed" << std::endl;
    return 1;
  }

  // Create network definition
  UniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
      1U << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  // Define input tensor (example: 1x3x10x10)
  auto input = network->addInput("input", nvinfer1::DataType::kFLOAT,
                                 nvinfer1::Dims4(1, 3, 10, 10));

  // Create plugin (no parameters)
  nvinfer1::PluginFieldCollection fc{};
  auto plugin = creator->createPlugin("identity_instance", &fc);
  if (!plugin) {
    std::cerr << "Plugin creation failed" << std::endl;
    return 1;
  }

  // Add plugin layer to network
  auto plugin_layer = network->addPluginV2(&input, 1, *plugin);
  plugin->destroy();  // Destroyable after being copied to the network

  // Set output
  auto output = plugin_layer->getOutput(0);
  output->setName("output");
  network->markOutput(*output);

  // Build config
  UniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMaxWorkspaceSize(1 << 20);  // 1MB

  // Building engine
  UniquePtr<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    std::cerr << "Engine build failed" << std::endl;
    return 1;
  }
  std::cout << "Engine build successfully" << std::endl;

  // 実行コンテキストの作成
  UniquePtr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());

  // 入出力用のメモリ確保
  const int size = 1 * 3 * 10 * 10;
  std::vector<float> input_data(size, 1.0f);  // 入力データ（全て1.0）
  std::vector<float> output_data(size);       // 出力データ

  // CUDA用のメモリ確保
  void* d_input = nullptr;
  void* d_output = nullptr;
  CHECK(cudaMalloc(&d_input, size * sizeof(float)));
  CHECK(cudaMalloc(&d_output, size * sizeof(float)));

  // Host to Device
  CHECK(cudaMemcpy(d_input, input_data.data(), size * sizeof(float),
                   cudaMemcpyHostToDevice));

  // Inference
  void* bindings[2] = {d_input, d_output};
  bool status = context->executeV2(bindings);
  if (!status) {
    std::cerr << "Inference failed" << std::endl;
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    return 1;
  }

  // Get result
  CHECK(cudaMemcpy(outputData.data(), d_output, size * sizeof(float),
                   cudaMemcpyDeviceToHost));

  // Verify result
  bool verified = true;
  for (int i = 0; i < 10; i++) {
    std::cout << "入力[" << i << "] = " << input_data[i] << ", 出力[" << i
              << "] = " << output_data[i] << std::endl;

    if (input_data[i] != output_data[i]) {
      verified = false;
    }
  }

  if (verified) {
    std::cout << "\nIdentityプラグインの検証に成功しました（入出力が一致）"
              << std::endl;
  } else {
    std::cerr << "\n検証失敗：入出力が一致しません" << std::endl;
  }

  // リソースの解放
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));

  std::cout << "Execution completed successfully" << std::endl;
  return 0;
}