import torch
import onnx
import caffe2.python.onnx.backend as backend
import numpy as np


class ONNXModel:
    def __init__(path, device='CUDA:0'):
        self.model = onnx.load(path)
        onnx.checker.check_model(self.model)
        self.rep = backend.prepare(self.model, device=device)

    def __repr__(self):
        onnx.helper.printable_graph(model.graph)

    def __call__(self, *args):
        return rep.run(npargs)


class TorchONNXModel(ONNXModel):
    def __call__(self, *args):
        npargs = tuple(arg.numpy().astype(np.float32) for arg in args)
        outputs = super().__call__(npargs)
        outputs = tuple(torch.from_numpy(output) for output in outputs)
        return outputs
