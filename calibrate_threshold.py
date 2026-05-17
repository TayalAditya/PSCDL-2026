"""
Calibrate detection threshold on PSCD_pers dataset.
Usage: python calib_pscd_pers.py --data_dir /tmp/pscd_pers_calib/PSCD_pers
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def foreground_mask(frame, bg, threshold):
    frame_f = frame.astype(np.float32)
    bg_f    = bg.astype(np.float32)
    mean_frame = frame_f.mean() + 1e-6
    mean_bg    = bg_f.mean()    + 1e-6
    frame_adj = np.clip(frame_f * (mean_bg / mean_frame), 0, 255).astype(np.uint8)
    lab_frame = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_bg    = cv2.cvtColor(bg,        cv2.COLOR_BGR2Lab).astype(np.float32)
    diff      = np.abs(lab_frame - lab_bg)
    score     = 0.4 * diff[:,:,0] + 0.8 * diff[:,:,1] + 0.8 * diff[:,:,2]
    score     = cv2.GaussianBlur(score, (5, 5), 0)
    _, fg     = cv2.threshold(score, threshold, 1, cv2.THRESH_BINARY)
    return fg.astype(np.uint8)

def f1(pred, gt):
    pred_b = pred.astype(bool)
    gt_b   = (gt > 127).astype(bool)
    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)

def eval_pair(args):
    t0_path, t1_path, gt_path, thresh = args
    t0 = cv2.imread(str(t0_path))
    t1 = cv2.imread(str(t1_path))
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if t0 is None or t1 is None or gt is None:
        return None
    if t0.shape != t1.shape:
        t1 = cv2.resize(t1, (t0.shape[1], t0.shape[0]))
    if gt.shape[:2] != t0.shape[:2]:
        gt = cv2.resize(gt, (t0.shape[1], t0.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = foreground_mask(t1, t0, thresh)
    return f1(pred, gt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/tmp/pscd_pers_calib/PSCD_pers')
    parser.add_argument('--thresh_min', type=int, default=10)
    parser.add_argument('--thresh_max', type=int, default=70)
    parser.add_argument('--thresh_step', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    data = Path(args.data_dir)
    t0_dir  = data / 't0'
    t1_dir  = data / 't1'
    gt_dir  = data / 'label_t1_integ'

    stems = sorted(p.stem for p in t0_dir.glob('*.png'))
    print(f'Found {len(stems)} pairs in {data}')

    thresholds = list(range(args.thresh_min, args.thresh_max + 1, args.thresh_step))
    best_thresh, best_f1 = 0, 0.0

    for thresh in thresholds:
        tasks = []
        for stem in stems:
            t0_p  = t0_dir  / f'{stem}.png'
            t1_p  = t1_dir  / f'{stem}.png'
            gt_p  = gt_dir  / f'{stem}.png'
            if t0_p.exists() and t1_p.exists() and gt_p.exists():
                tasks.append((t0_p, t1_p, gt_p, thresh))

        scores = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in as_completed([ex.submit(eval_pair, t) for t in tasks]):
                v = res.result()
                if v is not None:
                    scores.append(v)

        mean_f1 = float(np.mean(scores)) if scores else 0.0
        marker = ' <-- best' if mean_f1 > best_f1 else ''
        print(f'thresh={thresh:3d}  F1={mean_f1:.4f}  (n={len(scores)}){marker}')
        if mean_f1 > best_f1:
            best_f1, best_thresh = mean_f1, thresh

    print(f'\nBest threshold: {best_thresh}  F1={best_f1:.4f}')

if __name__ == '__main__':
    main()
