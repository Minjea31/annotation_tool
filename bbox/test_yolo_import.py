"""Verify external class mapping, saved labels and split preservation."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from bbox import app as review
from bbox.import_yolo import build_dataset, extract_archive, read_dataset, relocate_paths


class YoloImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'data.yaml').write_text('nc: 2\nnames: [bump, slow]\n')
        for split, count in (('train', 6), ('valid', 2), ('test', 2)):
            for folder in ('images', 'labels'):
                (self.source/split/folder).mkdir(parents=True)
            for i in range(count):
                Image.new('RGB', (100, 80), (len(split)*10, i*20, count*20)).save(self.source/split/'images'/f'{i}.png')
                (self.source/split/'labels'/f'{i}.txt').write_text('1 0.5 0.5 0.4 0.5\n0 0.2 0.2 0.1 0.1\n')
        self.archive = self.root/'source.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for path in self.source.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(self.source))

    def test_class_mapping_saved_labels_splits_and_installed_paths(self):
        records, names = read_dataset(self.source)
        box = records[0]['boxes'][0]
        self.assertEqual(box['class_id'], review.NAMES.index('slow'))
        self.assertEqual([box[key] for key in ('x1','y1','x2','y2')], [30,20,70,60])
        stage, destination = self.root/'stage', self.root/'dataset'
        report = build_dataset(self.source, stage, records, names, self.archive)
        self.assertEqual(report['counts'], {'train':6, 'test':2, 'eval':2})
        self.assertEqual(report['snapshot']['counts'], {'train':6, 'val':2, 'test':2})
        relocate_paths(stage, destination)
        stage.rename(destination)
        old_root, old_work = review.ROOT, review.WORK
        review.ROOT, review.WORK = destination, destination/'_store/review'
        try:
            state = review.state()
            self.assertEqual(len(state['images']), 10)
            self.assertTrue(all(row['status']=='approved' for row in state['images']))
            for row in state['images']:
                self.assertTrue(Path(row['path']).is_file())
                self.assertEqual(review.boxes(row['current'])[0]['class_id'], 0)
            expected = {row['id']:row['split'] for row in report['samples']}
            snapshot = review.snapshot(seed=97)
            manifest = json.loads((Path(snapshot['path'])/'manifest.json').read_text())
            actual = {row['id']:{'train':'train','val':'test','test':'eval'}[row['split']] for row in manifest['samples']}
            self.assertEqual(actual, expected)
            view = json.loads((destination/'manifest.json').read_text())
            for row in view['samples']:
                self.assertEqual(row['split'], expected[row['id']])
                self.assertTrue((destination/row['image']).is_file())
                self.assertTrue((destination/row['label']).read_text().startswith('0 '))
        finally:
            review.ROOT, review.WORK = old_root, old_work

    def test_missing_labels_are_rejected(self):
        (self.source/'train/labels/0.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            read_dataset(self.source)

    def test_archive_extraction_and_path_traversal_rejection(self):
        output = extract_archive(self.archive, self.root/'extracted')
        records, _ = read_dataset(output)
        self.assertEqual(len(records), 10)
        bad = self.root/'bad.zip'
        with zipfile.ZipFile(bad, 'w') as archive:
            archive.writestr('../outside.txt', 'not allowed')
        with self.assertRaisesRegex(ValueError, 'Unsafe ZIP'):
            extract_archive(bad, self.root/'bad-output')
        self.assertFalse((self.root/'outside.txt').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
