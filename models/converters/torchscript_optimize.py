from pathlib import Path
import torch
from torch import _dynamo as torchdynamo
import tvm
from ultralytics     import YOLO


parent_dir = Path(__file__).parents[0]

yolo_pt = parent_dir/"../yolo/yolov8n.pt"
yolo_jit = parent_dir/"../yolo/yolov8n.torchscript"

yolov8_yolo = YOLO(parent_dir/"../yolo/yolov8n.pt")  # load a pretrained model (recommended for training)
yolov8 = yolov8_yolo.model.eval()

input = torch.rand((1, 3, 32,32))
print("initial inference done")

out = yolov8(input)

# print("Start TVM optimisation")
# compiled_v8_tvm = torch.compile(yolov8, backend="tvm")
# out = compiled_v8_tvm(input)
# torchdynamo.reset()

# compiled_v8_inductor = torch.compile(yolov8, backend="inductor")
# out = compiled_v8_inductor(input)
# torchdynamo.reset()

# frozen_jit_inductor = torch.jit.optimize_for_inference(torch.jit.script(compiled_v8_inductor.eval()))

yolov8_yolo.export(format="torchscript")
yolov8_yolo.export(format="onnx")
frozen_jit = torch.jit.optimize_for_inference(torch.jit.load(yolo_jit).eval())
frozen_jit.save(yolo_jit)


# # tvm_jit.save(parent_dir/"../yolo/yolov8n_tvm.pt")
# frozen_jit.save(parent_dir/"../yolo/yolov8n_jit.pt")