#include <cuda_runtime_api.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include <NvInfer.h>
#include <NvInferPlugin.h>

// simple_identity_plugin.cc

using namespace nvinfer1;

class SimpleIdentityPlugin : public IPluginV2DynamicExt {
 public:
  SimpleIdentityPlugin() {
    std::cout << "SimpleIdentityPlugin constructor called" << std::endl;
  }

  SimpleIdentityPlugin(const void* /*data*/, size_t /*length*/) {
    std::cout << "SimpleIdentityPlugin deserialization constructor called"
              << std::endl;
  }

  IPluginV2DynamicExt* clone() const noexcept override {
    std::cout << "SimpleIdentityPlugin clone called" << std::endl;
    return new SimpleIdentityPlugin();
  }

  int getNbOutputs() const noexcept override { return 1; }

  // IPluginV2DynamicExtの純粋仮想関数
  DimsExprs getOutputDimensions(int output_index, const DimsExprs* inputs,
                                int nb_inputs,
                                IExprBuilder& expr_builder) noexcept override {
    std::cout << "getOutputDimensions called for output " << output_index
              << std::endl;
    if (output_index != 0 || nb_inputs != 1) {
      std::cout << "Invalid input/output configuration" << std::endl;
      DimsExprs empty;
      empty.nbDims = 0;
      return empty;
    }
    return inputs[0];
  }

  bool supportsFormatCombination(int pos, const PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) noexcept override {
    assert(nb_inputs == 1 && nb_outputs == 1);
    bool supported = (in_out[pos].type == DataType::kFLOAT) &&
                     (in_out[pos].format == TensorFormat::kLINEAR);
    std::cout << "supportsFormatCombination for pos " << pos << ": "
              << (supported ? "true" : "false") << std::endl;
    return supported;
  }

  void configurePlugin(const DynamicPluginTensorDesc* inputs, int32_t nb_inputs,
                       const DynamicPluginTensorDesc* outputs,
                       int32_t nb_outputs) noexcept override {
    std::cout << "configurePlugin called with " << nb_inputs << " inputs, "
              << nb_outputs << " outputs" << std::endl;
  }

  size_t getWorkspaceSize(const PluginTensorDesc* /*inputs*/, int /*nb_inputs*/,
                          const PluginTensorDesc* /*outputs*/,
                          int /*nb_outputs*/) const noexcept override {
    return 0;
  }

  // IPluginV2DynamicExtの純粋仮想関数
  int enqueue(const PluginTensorDesc* input_desc,
              const PluginTensorDesc* /*outputDesc*/, const void* const* inputs,
              void* const* outputs, void* /*workspace*/,
              cudaStream_t stream) noexcept override {
    int volume = 1;
    for (int i = 0; i < input_desc[0].dims.nbDims; i++)
      volume *= input_desc[0].dims.d[i];
    size_t bytes = volume * sizeof(float);

    std::cout << "enqueue called, copying " << bytes << " bytes" << std::endl;
    cudaMemcpyAsync(outputs[0], inputs[0], bytes, cudaMemcpyDeviceToDevice,
                    stream);
    return 0;
  }

  DataType getOutputDataType(int /*index*/, const DataType* input_types,
                             int /*nb_inputs*/) const noexcept override {
    return input_types[0];
  }

  // IPluginV2の純粋仮想関数
  const char* getPluginType() const noexcept override {
    return "SimpleIdentityPlugin";
  }
  // IPluginV2の純粋仮想関数
  const char* getPluginVersion() const noexcept override { return "1"; }
  int initialize() noexcept override {
    std::cout << "SimpleIdentityPlugin initialize called" << std::endl;
    return 0;
  }
  void terminate() noexcept override {
    std::cout << "SimpleIdentityPlugin terminate called" << std::endl;
  }

  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void* /*buffer*/) const noexcept override {}

  void destroy() noexcept override {
    std::cout << "SimpleIdentityPlugin destroy called" << std::endl;
    delete this;
  }
  void setPluginNamespace(const char* plugin_namespace) noexcept override {
    m_namespace_ = plugin_namespace;
  }
  const char* getPluginNamespace() const noexcept override {
    return m_namespace_.c_str();
  }

 private:
  std::string m_namespace_;
};

class SimpleIdentityPluginCreator : public IPluginCreator {
 public:
  SimpleIdentityPluginCreator() {
    mFC.nbFields = 0;
    mFC.fields = nullptr;
    std::cout << "SimpleIdentityPluginCreator constructor called" << std::endl;
  }

  // IPluginV2の純粋仮想関数
  const char* getPluginName() const noexcept override {
    return "SimpleIdentityPlugin";
  }

  // IPluginV2の純粋仮想関数
  const char* getPluginVersion() const noexcept override { return "1"; }

  // IPluginV2の純粋仮想関数
  const PluginFieldCollection* getFieldNames() noexcept override {
    return &mFC;
  }

  IPluginV2* createPlugin(const char* name,
                          const PluginFieldCollection* fc) noexcept override {
    std::cout << "Creating plugin instance: " << name << std::endl;
    return new SimpleIdentityPlugin();
  }

  IPluginV2* deserializePlugin(const char* name, const void* serial_data,
                               size_t serial_length) noexcept override {
    std::cout << "Deserializing plugin instance: " << name << std::endl;
    return new SimpleIdentityPlugin(serial_data, serial_length);
  }

  void setPluginNamespace(const char* lib_namespace) noexcept override {
    m_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return m_namespace_.c_str();
  }

 private:
  PluginFieldCollection m_fc_;
  std::string m_namespace_;
};

extern "C" {

// ライブラリ内で利用するロガーをTensorRT側に設定するための関数であり、プラグイン内で発生したログメッセージをTensorRTのログシステムで管理できるようにする目的が推測
// 必要に応じてロガーファインダーを設定するための関数（ここでは何もしない）
void setLoggerFinder(nvinfer1::ILoggerFinder& finder)
{
    // 例：グローバルなロガーファインダーのポインタを設定する処理
}

// プラグインクリエイター配列の取得
// ライブラリが持つプラグインの生成（クリエーション）を行うオブジェクト群をTensorRT側に提供するための関数で、TensorRTがカスタムプラグインを動的に取得・登録するために必要です。
IPluginCreator* const* getPluginCreators(int32_t& nb_creators)
{
    // シングルトンとしてプラグインクリエイターのインスタンスを生成
    static SimpleIdentityPluginCreator plugin_creator_instance;
    static nvinfer1::IPluginCreator* plugin_creators[] = {
        &plugin_creator_instance};

    nb_creators = sizeof(plugin_creators) / sizeof(plugin_creators[0]);
    return plugin_creators;
}

} // extern "C"


// REGISTER_TENSORRT_PLUGIN(SimpleIdentityPluginCreator);