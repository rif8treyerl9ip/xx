import os
import tensorrt as trt
from pathlib import Path

def convert_onnx_to_tensorrt(
    onnx_file_path,
    engine_file_path,
    precision="fp16",
    max_batch_size=1,
    max_workspace_size=1 << 30,  # 1GB
    verbose=False
):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    
    # TensorRTビルダーの作成
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # 精度設定
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8の場合はキャリブレーションが必要です（ここでは省略）
    
    # ONNXパーサーの作成
    parser = trt.OnnxParser(network, logger)
    
    # ONNXモデルの読み込み
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ONNXモデルの解析に失敗しました。")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # プロファイルの設定（動的バッチサイズの場合）
    profile = builder.create_optimization_profile()
    # 入力テンソルの名前と形状に応じて設定が必要です
    # 以下は例です。実際のモデルに合わせて調整してください
    # profile.set_shape("input", (1, 3, 224, 224), (max_batch_size, 3, 224, 224), (max_batch_size, 3, 224, 224))
    # config.add_optimization_profile(profile)
    
    # TensorRTエンジンの構築
    print("TensorRTエンジンをビルド中...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("エンジンの構築に失敗しました。")
        return False
    
    # エンジンの保存
    with open(engine_file_path, 'wb') as f:
        f.write(engine)
    
    print(f"TensorRTエンジンを保存しました: {engine_file_path}")
    return True

if __name__ == "__main__":
    input_onnx = "resnet50.onnx"  # 変換するONNXモデルのパス
    output_engine = "resnet50.engine"  # 出力するTensorRTエンジンのパス
    
    # 変換実行
    success = convert_onnx_to_tensorrt(
        input_onnx, 
        output_engine,
        precision="fp16",  # fp16, fp32, int8から選択
        max_batch_size=1,
        # verbose=True
    )
    
    if success:
        print("変換が完了しました。")
    else:
        print("変換に失敗しました。")