"""YOLO adapter. Class names, never raw source IDs, define label semantics."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
NAMES = [s.strip() for s in (HERE/'roboflow_classes.csv').read_text().splitlines() if s.strip()]

def load(path):
    from ultralytics import YOLO
    m = YOLO(str(path))
    names = list(m.names.values())
    if m.task != 'detect' or not names or len(set(names)) != len(names) or not set(names).issubset(NAMES):
        raise ValueError('Expected a detection model using the CSV classes')
    config=m.model.yaml
    if config.get('scale')!='n' or 'yolo26' not in Path(config.get('yaml_file','')).stem:
        raise ValueError('BBox requires YOLO26 Nano (YOLO26n)')
    return m

def reorder(source, target):
    import torch
    m = load(source)
    source_names = [m.names[i] for i in range(len(m.names))]
    if source_names != NAMES:
        head = m.model.model[-1]
        with torch.no_grad():
            for attr in ('cv3', 'one2one_cv3'):
                branches = getattr(head, attr, None)
                if branches is None: continue
                for level, branch in enumerate(branches):
                    conv = branch[-1]
                    if conv.out_channels != len(source_names) or conv.kernel_size != (1, 1):
                        raise ValueError('Unsupported classification head')
                    expanded = torch.nn.Conv2d(conv.in_channels, len(NAMES), 1).to(conv.weight)
                    expanded.weight.zero_()
                    import math
                    expanded.bias.fill_(math.log(5 / len(NAMES) / (640 / float(head.stride[level])) ** 2))
                    for old_id, name in enumerate(source_names):
                        new_id = NAMES.index(name)
                        expanded.weight[new_id].copy_(conv.weight[old_id])
                        expanded.bias[new_id].copy_(conv.bias[old_id])
                    branch[-1] = expanded
        head.nc = len(NAMES)
        head.no = head.nc + head.reg_max * 4
        m.model.yaml['nc'] = len(NAMES)
    m.model.names = dict(enumerate(NAMES))
    m.save(str(target))
    return target

if __name__ == '__main__':
    import json
    payload = json.loads(Path(sys.argv[1]).read_text())
    output = Path(payload['output']); output.mkdir(parents=True, exist_ok=True)
    weights = reorder(payload['weights'], output/'start.pt')
    m = load(weights)
    def epoch(trainer):
        record = {'epoch':trainer.epoch+1, 'metrics':{k:float(v) for k,v in trainer.metrics.items()}}
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    m.add_callback('on_fit_epoch_end', epoch)
    import torch
    m.train(data=payload['data'], epochs=payload['epochs'], batch=payload['batch_size'],
            lr0=payload['lr'], seed=payload['seed'], imgsz=640, workers=0,
            device=0 if torch.cuda.is_available() else 'cpu', project=str(output), name='train', exist_ok=True,
            amp=False, plots=False, optimizer="AdamW")
    best=output/'train/weights/best.pt'
    metrics=load(best).val(data=payload['data'], split='test', imgsz=640, workers=0,
                           project=str(output), name='eval', exist_ok=True, plots=False)
    (output/'result.json').write_text(json.dumps({'best':str(best), 'eval':metrics.results_dict}))
