"""Export static batch-1 logits; verify numerical parity with PyTorch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from autorace_lane_lite.constants import CLASSES, MEAN, STD
from autorace_lane_lite.model import load_checkpoint


def export(weights, output, size, verify=True):
    import onnx
    width, height = map(int, size.split('x'))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(weights)
    model.eval()
    sample = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(31))
    torch.onnx.export(model, sample, str(output), opset_version=17, input_names=['rgb'],
                      output_names=['logits'], do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(output)))
    metadata = {'architecture': 'mobilenet_v3_small_unet', 'classes': list(CLASSES),
                'input': [1, 3, height, width], 'input_dtype': 'float32', 'color': 'RGB',
                'resize': 'stretch', 'mean': MEAN, 'std': STD,
                'training_size': checkpoint['size'], 'output': 'unnormalized logits [1,3,H,W]'}
    if verify:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            expected = model(sample).numpy()
        actual = session.run(['logits'], {'rgb': sample.numpy()})[0]
        error = float(np.abs(expected - actual).max())
        agreement = float((expected.argmax(1) == actual.argmax(1)).mean())
        if not np.allclose(expected, actual, atol=1e-4, rtol=1e-3):
            raise RuntimeError(f'ONNX verification failed: max error={error}')
        metadata['verification'] = {'max_absolute_error': error, 'mask_agreement': agreement}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--size', choices=['640x480', '512x288'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--no-verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export(args.weights, args.output, args.size, not args.no_verify), indent=2))
