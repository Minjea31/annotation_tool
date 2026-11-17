"""Lazy backend imports: OpenCV and TensorRT paths do not load PyTorch."""
import ctypes
import ctypes.util
import glob
from pathlib import Path
import sys

import cv2
import numpy as np

from .constants import MEAN, STD


def preprocess(bgr, width, height):
    rgb = cv2.cvtColor(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def decode(logits, original_size, threshold=0.5):
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError('Expected finite logits [1,3,H,W]')
    values = logits[0].astype(np.float32)
    values -= values.max(axis=0, keepdims=True)
    probability = np.exp(values)
    probability /= probability.sum(axis=0, keepdims=True)
    mask = probability.argmax(axis=0).astype(np.uint8)
    mask[probability.max(axis=0) < threshold] = 0
    return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)


class OpenCVBackend:
    name = 'opencv'

    def __init__(self, config):
        self.config = config

    def predict(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, np.array(self.config['orange_lower']), np.array(self.config['orange_upper']))
        white = cv2.inRange(hsv, np.array(self.config['white_lower']), np.array(self.config['white_upper']))
        # Remove long connected solid-white marks. This is only a color/shape heuristic.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_HEIGHT] > bgr.shape[0] * self.config['white_max_height_ratio']:
                white[labels == index] = 0
        mask = np.zeros(bgr.shape[:2], np.uint8)
        mask[white > 0], mask[orange > 0] = 2, 1
        return mask


class TorchBackend:
    name = 'torch'

    def __init__(self, path, width, height, device='cuda'):
        import torch
        from .model import load_checkpoint
        self.torch, self.device = torch, torch.device(device)
        self.model, _ = load_checkpoint(path, self.device)
        self.model.eval()
        self.width, self.height = width, height

    def logits(self, rgb):
        with self.torch.inference_mode():
            return self.model(self.torch.from_numpy(rgb).to(self.device)).float().cpu().numpy()


class ONNXBackend:
    name = 'onnx'

    def __init__(self, path, width, height, device='cpu'):
        import onnxruntime as ort
        if device not in ('cpu', 'cuda'):
            raise ValueError('ONNX device must be cpu/cuda')
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA ONNX Runtime provider is unavailable; select device=cpu explicitly')
            providers.insert(0, 'CUDAExecutionProvider')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        shape = self.session.get_inputs()[0].shape
        if shape != [1, 3, height, width]:
            raise ValueError(f'ONNX shape {shape} differs from configured size')
        self.width, self.height = width, height

    def logits(self, rgb):
        return self.session.run(['logits'], {'rgb': rgb})[0]


class Cuda:
    def __init__(self):
        candidates = [ctypes.util.find_library('cudart'), 'libcudart.so']
        candidates += sorted(glob.glob('/usr/local/cuda*/lib64/libcudart.so*'))
        candidates += sorted(glob.glob('/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*'))
        candidates += sorted(glob.glob(str(Path(sys.prefix) / 'lib/libcudart.so*')))
        candidates += sorted(glob.glob(str(Path(sys.prefix) / 'lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*')))
        self.lib = None
        for path in filter(None, candidates):
            try:
                self.lib = ctypes.CDLL(path)
                break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError('libcudart not found; install/source the JetPack CUDA runtime')
        definitions = {
            'cudaSetDevice': [ctypes.c_int],
            'cudaMalloc': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            'cudaFree': [ctypes.c_void_p],
            'cudaMemcpy': [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            'cudaStreamCreate': [ctypes.POINTER(ctypes.c_void_p)],
            'cudaStreamSynchronize': [ctypes.c_void_p],
            'cudaStreamDestroy': [ctypes.c_void_p],
        }
        for name, argtypes in definitions.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = argtypes, ctypes.c_int
        self.call('cudaSetDevice', 0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f'{name} failed with CUDA error {code}')


class TensorRTBackend:
    """Static batch-1, explicit-batch TensorRT 8.5+ / 10.x engine runner."""
    name = 'tensorrt'

    def __init__(self, path, width, height, device='cuda'):
        import tensorrt as trt
        self.cuda = Cuda()
        self.allocations, self.stream = [], ctypes.c_void_p()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed; rebuild on this Jetson/TensorRT version')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create TensorRT execution context')
        # Avoid old trt.nptype's dependency on removed NumPy aliases (np.bool).
        def numpy_dtype(value):
            if value == trt.DataType.FLOAT:
                return np.float32
            if value == trt.DataType.HALF:
                return np.float16
            raise ValueError('Lane engine I/O must use float32/float16')
        self.modern = hasattr(self.engine, 'num_io_tensors') and hasattr(self.context, 'execute_async_v3')
        if self.modern:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            shape = lambda name: tuple(self.engine.get_tensor_shape(name))
            dtype = lambda name: numpy_dtype(self.engine.get_tensor_dtype(name))
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            shape = lambda name: tuple(self.engine.get_binding_shape(self.engine.get_binding_index(name)))
            dtype = lambda name: numpy_dtype(self.engine.get_binding_dtype(self.engine.get_binding_index(name)))
        if set(names) != {'rgb', 'logits'} or shape('rgb') != (1, 3, height, width) or shape('logits') != (1, 3, height, width):
            raise ValueError('Engine must have static rgb/logits tensors matching configured resolution')
        self.input = np.empty(shape('rgb'), dtype=dtype('rgb'))
        self.output = np.empty(shape('logits'), dtype=dtype('logits'))
        if self.input.dtype not in (np.float32, np.float16) or self.output.dtype not in (np.float32, np.float16):
            raise ValueError('Engine I/O must use float32/float16 logits')
        self.cuda.call('cudaStreamCreate', ctypes.byref(self.stream))
        self.pointers = {}
        for name, array in (('rgb', self.input), ('logits', self.output)):
            pointer = ctypes.c_void_p()
            self.cuda.call('cudaMalloc', ctypes.byref(pointer), array.nbytes)
            self.allocations.append(pointer)
            self.pointers[name] = pointer
            if self.modern and not self.context.set_tensor_address(name, pointer.value):
                raise RuntimeError(f'TensorRT could not bind {name}')
        self.bindings = [self.pointers[name].value for name in names]
        self.width, self.height = width, height

    def logits(self, rgb):
        np.copyto(self.input, rgb, casting='same_kind')
        self.cuda.call('cudaMemcpy', self.pointers['rgb'], self.input.ctypes.data, self.input.nbytes, 1)
        if self.modern:
            ok = self.context.execute_async_v3(stream_handle=self.stream.value)
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.value)
        if not ok:
            raise RuntimeError('TensorRT execution failed')
        self.cuda.call('cudaStreamSynchronize', self.stream)
        self.cuda.call('cudaMemcpy', self.output.ctypes.data, self.pointers['logits'], self.output.nbytes, 2)
        return self.output.astype(np.float32, copy=True)

    def close(self):
        if getattr(self, 'stream', None) and self.stream.value:
            self.cuda.call('cudaStreamSynchronize', self.stream)
        for pointer in getattr(self, 'allocations', []):
            self.cuda.call('cudaFree', pointer)
        self.allocations = []
        if getattr(self, 'stream', None) and self.stream.value:
            self.cuda.call('cudaStreamDestroy', self.stream)
            self.stream = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Interpreter shutdown/partial construction must not mask the original error.


def create_backend(kind, path, width, height, device='cuda'):
    if kind == 'auto':
        kind = {'.pt': 'torch', '.pth': 'torch', '.onnx': 'onnx', '.engine': 'tensorrt', '.plan': 'tensorrt'}.get(Path(path).suffix)
    choices = {'torch': TorchBackend, 'onnx': ONNXBackend, 'tensorrt': TensorRTBackend}
    if kind not in choices:
        raise ValueError('Choose torch/onnx/tensorrt or a recognized model suffix')
    return choices[kind](path, width, height, device)
