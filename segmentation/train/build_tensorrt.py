"""Run ON THE TARGET JETSON: ONNX → TensorRT 8.5+/10.x FP16 engine."""
import argparse
import json
from pathlib import Path


def build(onnx_path, output, workspace_mib):
    import tensorrt as trt
    major, minor = [int(x) for x in trt.__version__.split('.')[:2]]
    if major < 8 or (major == 8 and minor < 5) or major >= 11:
        raise RuntimeError('This JetPack runner supports TensorRT 8.5–10.x; TRT 11 requires explicit precision export')
    if output.exists():
        raise FileExistsError(output)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError('Lane engine needs one input and one logits output')
    image, logits = network.get_input(0), network.get_output(0)
    shape = tuple(image.shape)
    if image.name != 'rgb' or logits.name != 'logits' or shape not in ((1, 3, 480, 640), (1, 3, 288, 512)) or tuple(logits.shape) != shape:
        raise ValueError('Unexpected ONNX interface; export the lane model at a supported static size')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('FP16 hardware support is unavailable')
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT build failed; inspect log and available RAM')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    metadata = {'tensorrt': trt.__version__, 'precision': 'FP16 enabled; TensorRT may retain FP32 layers',
                'workspace_mib': workspace_mib, 'onnx': onnx_path.name,
                'input_shape': list(network.get_input(0).shape), 'build_on_target_required': True}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workspace-mib', default=1024, type=int)
    args = parser.parse_args()
    if args.workspace_mib < 1:
        parser.error('workspace must be positive')
    print(json.dumps(build(args.onnx, args.output, args.workspace_mib), indent=2))
