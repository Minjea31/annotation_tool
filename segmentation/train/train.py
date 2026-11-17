"""Train/evaluate the same U-Net at either supported resolution."""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from autorace_lane_lite.model import CLASSES, LaneUNet, load_checkpoint
from data import LaneDataset
from losses import DiceCrossEntropy, iou_from_confusion


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=True):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((3, 3), dtype=torch.int64, device=device)
    loss_sum, samples, updates = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for rgb, target in loader:
            rgb, target = rgb.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda' and amp):
                logits = model(rgb)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; check input images and masks')
            if training:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                updates += int(scaler.get_scale() >= scale_before)
            pred = logits.detach().argmax(1)
            keep = target != 255
            confusion += torch.bincount(3 * target[keep] + pred[keep], minlength=9).reshape(3, 3)
            loss_sum += float(loss.detach()) * len(rgb)
            samples += len(rgb)
    result = {'loss': loss_sum / samples, **iou_from_confusion(confusion.cpu())}
    if training:
        result['optimizer_steps'] = updates
        if not updates:
            raise FloatingPointError('AMP skipped every optimizer update; retry with --no-amp and check data')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/lane_512'))
    parser.add_argument('--size', choices=['640x480', '512x288'], default='512x288')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', action='store_true', help='Download torchvision ImageNet encoder weights')
    parser.add_argument('--no-amp', action='store_true', help='Disable CUDA automatic mixed precision')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--weights', type=Path, help='Initialize model weights; reset optimizer')
    parser.add_argument('--evaluate', choices=['val', 'test'])
    args = parser.parse_args()
    if args.resume and args.weights:
        parser.error('Use either --resume or --weights')
    if args.evaluate and not (args.weights or args.resume):
        parser.error('--evaluate needs --weights/--resume')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs/batch-size must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    width, height = map(int, args.size.split('x'))
    checkpoint = None
    if args.resume or args.weights:
        model, checkpoint = load_checkpoint(args.resume or args.weights, device)
    else:
        model = LaneUNet(pretrained=args.pretrained).to(device)
    criterion = DiceCrossEntropy().to(device)
    validation = DataLoader(LaneDataset(args.data, args.evaluate or 'val', width, height),
                            batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=seed_worker)
    if args.evaluate:
        print(json.dumps(run_epoch(model, validation, criterion, device, amp=not args.no_amp), indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Output is not empty; choose a new run directory or use --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    training = DataLoader(LaneDataset(args.data, 'train', width, height, augment=True),
                          batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          worker_init_fn=seed_worker)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda' and not args.no_amp, init_scale=128.0)
    start, best = 0, -1.0
    if args.resume:
        if checkpoint['size'] != [width, height]:
            parser.error('Resume size differs; use --weights for resolution fine tuning')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if scheduler.T_max != args.epochs:
            scheduler.T_max = args.epochs
            # A completed cosine run has lr=0. Recompute the current LR when
            # extending its horizon so resumed training can actually learn.
            for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
                group['lr'] = base_lr * (1 + math.cos(math.pi * scheduler.last_epoch / args.epochs)) / 2
        scaler.load_state_dict(checkpoint['scaler'])
        start, best = checkpoint['epoch'] + 1, checkpoint['best_lane_miou']
        if start >= args.epochs:
            parser.error('--epochs must exceed the completed epoch count when resuming')
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
    (args.output / 'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + '\n')
    for epoch in range(start, args.epochs):
        train_metrics = run_epoch(model, training, criterion, device, optimizer, scaler, amp=not args.no_amp)
        val_metrics = run_epoch(model, validation, criterion, device, amp=not args.no_amp)
        scheduler.step()
        score = val_metrics['lane_miou']
        improved = score is not None and score > best
        if improved:
            best = score
        state = {'model': model.state_dict(), 'classes': CLASSES, 'size': [width, height],
                 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'best_lane_miou': best,
                 'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                 'python_rng': random.getstate(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        torch.save(state, args.output / 'last.pt')
        if improved:
            torch.save(state, args.output / 'best.pt')
        row = {'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics}
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
