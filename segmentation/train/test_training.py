import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import cv2
from PIL import Image
import torch

import _bootstrap  # noqa: F401
from autorace_lane_lite.model import CLASSES, LaneUNet
from autorace_lane_lite.runtime import preprocess
from convert_roboflow import convert, polygon_mask
from data import LaneDataset
from losses import DiceCrossEntropy, iou_from_confusion


def fixture(root):
    for split in ('train', 'valid'):
        path = root / split
        path.mkdir(parents=True)
        for index in range(2):
            Image.fromarray(np.full((48, 64, 3), index + (10 if split == 'valid' else 0), np.uint8)).save(path / f'{index}.png')
        annotations = [
            {'image_id': 0, 'category_id': 31, 'segmentation': [[5, 5, 30, 5, 30, 40, 5, 40]]},
            {'image_id': 0, 'category_id': 7, 'segmentation': [[20, 5, 40, 5, 40, 40, 20, 40]]},
        ]
        data = {'images': [{'id': i, 'file_name': f'{i}.png', 'width': 64, 'height': 48} for i in range(2)],
                'categories': [{'id': 7, 'name': 'white_dashed_lane'}, {'id': 31, 'name': 'orange_lane'}],
                'annotations': annotations}
        (path / '_annotations.coco.json').write_text(json.dumps(data))


class DataTests(unittest.TestCase):
    def test_coco_mapping_overlap_negative_and_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / 'source')
            metadata = convert(root / 'source', root / 'converted')
            self.assertEqual(metadata['classes'], list(CLASSES))
            rows = [row for row in metadata['samples'] if row['split'] == 'train']
            with Image.open(root / 'converted' / rows[0]['mask']) as image:
                mask = np.array(image)
            self.assertEqual((mask[10, 10], mask[10, 25], mask[10, 35]), (1, 255, 2))
            with Image.open(root / 'converted' / rows[1]['mask']) as image:
                self.assertEqual(np.array(image).sum(), 0)
            rgb, target = LaneDataset(root / 'converted', 'val', 512, 288)[0]
            self.assertEqual(tuple(rgb.shape), (3, 288, 512))
            self.assertTrue(set(target.unique().tolist()) <= {0, 1, 2, 255})
            row = next(row for row in metadata['samples'] if row['split'] == 'val')
            bgr = cv2.imread(str(root / 'converted' / row['image']))
            np.testing.assert_allclose(rgb.numpy(), preprocess(bgr, 512, 288)[0], atol=1e-6)

    def test_duplicate_images_across_splits_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / 'source')
            (root / 'source/valid/0.png').write_bytes((root / 'source/train/0.png').read_bytes())
            with self.assertRaises(ValueError):
                convert(root / 'source', root / 'converted')
            self.assertFalse((root / 'converted').exists())

    def test_keypoint_or_rle_is_not_silently_treated_as_polygon(self):
        for annotation in ({'keypoints': [1, 2, 2]}, {'segmentation': {'counts': 'rle'}}):
            with self.assertRaises(ValueError):
                polygon_mask(annotation, 48, 64)


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_resolution_outputs_and_parameter_budget(self):
        model = LaneUNet().eval()
        self.assertLess(sum(p.numel() for p in model.parameters()), 2_000_000)
        with torch.inference_mode():
            for width, height in ((640, 480), (512, 288)):
                self.assertEqual(tuple(model(torch.zeros(1, 3, height, width)).shape), (1, 3, height, width))

    def test_dice_ce_gradients_and_ignore_index(self):
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 3, (2, 32, 32))
        target[:, :4] = 255
        loss = DiceCrossEntropy()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[:, :, :4].abs().sum().item(), 0)
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_empty_supervision_and_absent_classes(self):
        logits = torch.randn(1, 3, 16, 16, requires_grad=True)
        loss = DiceCrossEntropy()(logits, torch.full((1, 16, 16), 255))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertIsNone(iou_from_confusion(torch.zeros(3, 3))['lane_miou'])


if __name__ == '__main__':
    unittest.main()
