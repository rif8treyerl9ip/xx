import torch
from torchvision.models import resnet50

model = resnet50(pretrained=True)

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model,
            dummy_input,
            "resnet50.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            )
