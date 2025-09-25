from ultralytics import YOLO
import numpy as np
import cv2
import os
from math import sqrt
from datetime import datetime

KEYPOINT_NAMES = ["back-left", "front", "back-right"]

# Load your trained model
model = YOLO('runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt')

print("POSE ESTIMATION QUALITY METRICS")
print("=" * 50)

# 1. Standard YOLO validation metrics
print("\n 1. STANDARD YOLO METRICS:")
results = model.val(
    data="./datasets/moorebot_v5/data.yaml",
    split='test',
    imgsz=1280,
    batch=4,
    verbose=False
)

# Extract pose-specific metrics
if hasattr(results, 'pose') and results.pose is not None:
    print(f" Pose mAP50:     {results.pose.map50:.4f}")
    print(f" Pose mAP50-95:  {results.pose.map:.4f}")
    print(f" Pose mAP75:     {results.pose.map75:.4f}")
else:
    print("  Pose metrics not available in results object")

# Print box metrics for person detection
if hasattr(results, 'box') and results.box is not None:
    print(f" Box mAP50:      {results.box.map50:.4f}")
    print(f" Box mAP50-95:   {results.box.map:.4f}")

print("\n 2. POSE-SPECIFIC QUALITY METRICS:")

# Custom pose evaluation
def calculate_pose_metrics(model, test_dir, labels_dir):
    """Calculate custom pose estimation metrics"""
    
    # Get test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    total_images = 0
    total_keypoints_detected = 0
    total_keypoints_gt = 0
    correct_keypoints = 0
    pck_distances = []  # For PCK calculation
    
    print(f"   Analyzing {len(test_images)} test images...")
    
    for img_name in test_images[:10]:  # Limit to first 10 for speed
        img_path = os.path.join(test_dir, img_name)
        
        # Get predictions
        results = model(img_path, verbose=False)
        
        # Load ground truth (simplified - would need actual label parsing)
        total_images += 1
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                # Count detected keypoints
                for person_kpts in keypoints:
                    for kpt_conf in confidences:
                        for conf in kpt_conf:
                            if conf > 0.5:  # Confidence threshold
                                total_keypoints_detected += 1
    
    return {
        'total_images': total_images,
        'avg_keypoints_per_image': total_keypoints_detected / max(total_images, 1),
        'detection_rate': total_keypoints_detected / max(total_images * 3, 1)  # Assuming 3 keypoints max
    }


# ---------------------------
# PCK (Percentage of Correct Keypoints)
# ---------------------------
def _read_yolo_pose_labels(label_path, img_w, img_h):
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            # denormalize bbox to pixels
            bw, bh = w * img_w, h * img_h
            bx, by = cx * img_w, cy * img_h
            x1, y1 = bx - bw / 2, by - bh / 2
            x2, y2 = bx + bw / 2, by + bh / 2
            # keypoints triplets follow
            kpt_vals = list(map(float, parts[5:]))
            kpts = []
            for i in range(0, len(kpt_vals), 3):
                if i + 2 >= len(kpt_vals):
                    break
                kx, ky, v = kpt_vals[i], kpt_vals[i+1], kpt_vals[i+2]
                kpts.append((kx * img_w, ky * img_h, int(v)))
            gts.append({
                'cls': cls,
                'bbox': (x1, y1, x2, y2),
                'kpts': kpts,
            })
    return gts


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def calculate_pck(model, test_dir, labels_dir, alphas=(0.1, 0.2), iou_thresh=0.5, limit=None):
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    if limit:
        images = images[:limit]

    totals = {alpha: 0 for alpha in alphas}
    corrects = {alpha: 0 for alpha in alphas}
    per_kpt_totals = {}
    per_kpt_corrects = {}
    matched_pairs = 0
    total_gts = 0

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        gts = _read_yolo_pose_labels(label_path, w, h)
        if not gts:
            continue

        total_gts += len(gts)

        preds = model(img_path, verbose=False)[0]
        if preds is None or preds.boxes is None or preds.keypoints is None:
            continue

        pred_boxes = preds.boxes.xyxy.cpu().numpy()
        pred_kpts = preds.keypoints.xy.cpu().numpy()  # shape: (N, K, 2)

        # Build IoU matrix
        iou_mat = np.zeros((len(pred_boxes), len(gts)), dtype=float)
        for i, pb in enumerate(pred_boxes):
            for j, gt in enumerate(gts):
                iou_mat[i, j] = _iou_xyxy(tuple(pb.tolist()), gt['bbox'])

        # Greedy matching by IoU
        pred_used = set()
        gt_used = set()
        while True:
            max_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            max_iou = iou_mat[max_idx]
            if max_iou < iou_thresh:
                break
            pi, gi = int(max_idx[0]), int(max_idx[1])
            if pi in pred_used or gi in gt_used:
                iou_mat[pi, gi] = -1.0
                continue
            pred_used.add(pi)
            gt_used.add(gi)
            iou_mat[pi, :] = -1.0
            iou_mat[:, gi] = -1.0

            matched_pairs += 1

            # Compute PCK for this pair
            x1, y1, x2, y2 = gts[gi]['bbox']
            ref = max(x2 - x1, y2 - y1)
            if ref <= 0:
                continue
            gt_k = gts[gi]['kpts']
            pred_k = pred_kpts[pi]
            k_len = min(len(gt_k), pred_k.shape[0])
            for k in range(k_len):
                gx, gy, gv = gt_k[k]
                if gv <= 0:  # skip unlabeled keypoints
                    continue
                px, py = pred_k[k]
                dist = sqrt((px - gx) ** 2 + (py - gy) ** 2)
                for alpha in alphas:
                    totals[alpha] += 1
                    per_kpt_totals.setdefault(alpha, {}).setdefault(k, 0)
                    per_kpt_totals[alpha][k] += 1
                    if dist <= alpha * ref:
                        corrects[alpha] += 1
                        per_kpt_corrects.setdefault(alpha, {}).setdefault(k, 0)
                        per_kpt_corrects[alpha][k] += 1

    pck = {alpha: (corrects[alpha] / totals[alpha]) if totals[alpha] > 0 else 0.0 for alpha in alphas}
    return {
        'pck': pck,
        'totals': totals,
        'corrects': corrects,
        'per_keypoint_pck': {alpha: {k: (per_kpt_corrects.get(alpha, {}).get(k, 0) / per_kpt_totals.get(alpha, {}).get(k, 1)) for k in per_kpt_totals.get(alpha, {})} for alpha in alphas},
        'per_keypoint_totals': per_kpt_totals,
        'per_keypoint_corrects': per_kpt_corrects,
        'matched_pairs': matched_pairs,
        'total_gts': total_gts,
    }


def _kpt_name(idx, names):
    if isinstance(names, (list, tuple)) and idx < len(names):
        return str(names[idx])
    return f"kpt {idx}"


def plot_per_keypoint_pck(pck_metrics, keypoint_names=None, out_dir="runs/pose/pck", alphas=(0.1, 0.2)):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  Skipping PCK plot (matplotlib not available): {e}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # Union of keypoint indices across alphas
    kpt_set = set()
    for a in alphas:
        kpt_set.update(pck_metrics['per_keypoint_totals'].get(a, {}).keys())
    kpt_ids = sorted(kpt_set)
    if not kpt_ids:
        print("  No per-keypoint data to plot.")
        return None

    labels = [_kpt_name(k, keypoint_names) for k in kpt_ids]
    x = np.arange(len(kpt_ids))
    n = max(1, len(alphas))
    width = min(0.8 / n, 0.35)

    fig, ax = plt.subplots(figsize=(max(6, len(kpt_ids) * 0.6), 4.5))
    for i, a in enumerate(alphas):
        vals = [pck_metrics['per_keypoint_pck'].get(a, {}).get(k, 0.0) for k in kpt_ids]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=f"PCK@{a:.1f}")

    ax.set_ylabel('PCK')
    ax.set_title('Per-Keypoint PCK')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_path = os.path.join(out_dir, f"pck_per_keypoint_{ts}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Per-keypoint PCK plot saved: {out_path}")
    return out_path

# Calculate custom metrics (fix dataset path)
test_dir = "./datasets/moorebot_v5/test/images"
labels_dir = "./datasets/moorebot_v5/test/labels"

if os.path.exists(test_dir):
    custom_metrics = calculate_pose_metrics(model, test_dir, labels_dir)
    
    print(f" Images analyzed:           {custom_metrics['total_images']}")
    print(f" Avg keypoints per image:   {custom_metrics['avg_keypoints_per_image']:.2f}")
    print(f" Keypoint detection rate:   {custom_metrics['detection_rate']:.2f}")
else:
    print("  Test directory not found")

# Compute PCK metrics
if os.path.exists(test_dir) and os.path.exists(labels_dir):
    pck_metrics = calculate_pck(model, test_dir, labels_dir, alphas=(0.1, 0.2), iou_thresh=0.5, limit=None)
    print("\n PCK METRICS:")
    for alpha, val in pck_metrics['pck'].items():
        print(f"   PCK@{alpha:.1f}: {val:.3f}  ({pck_metrics['corrects'][alpha]}/{pck_metrics['totals'][alpha]} keypoints)")
    print(f"   Matched predictions/GT pairs: {pck_metrics['matched_pairs']}/{pck_metrics['total_gts']}")
    # Per-keypoint breakdown
    for alpha, per_k in pck_metrics['per_keypoint_pck'].items():
        if not per_k:
            continue
        print(f"   Per-keypoint PCK@{alpha:.1f}:")
        for k_idx in sorted(per_k.keys()):
            corr = pck_metrics['per_keypoint_corrects'].get(alpha, {}).get(k_idx, 0)
            tot = pck_metrics['per_keypoint_totals'].get(alpha, {}).get(k_idx, 0)
            val = per_k[k_idx]
            print(f"     - {_kpt_name(k_idx, KEYPOINT_NAMES)}: {val:.3f} ({corr}/{tot})")

    # Plot per-keypoint PCK bar chart
    try:
        plot_per_keypoint_pck(pck_metrics, keypoint_names=KEYPOINT_NAMES, alphas=(0.1, 0.2))
    except Exception as e:
        print(f"  Failed to plot per-keypoint PCK: {e}")

# print("\n3. KEY POSE ESTIMATION METRICS EXPLAINED:")
# print("""
# POSE mAP50:
#    - Measures keypoint detection accuracy at IoU threshold 0.5
#    - Range: 0.0-1.0 (higher is better)
#    - Good: >0.7, Excellent: >0.9

# POSE mAP50-95:
#    - Average mAP across IoU thresholds 0.5-0.95
#    - More strict metric (harder to achieve high scores)
#    - Good: >0.5, Excellent: >0.8

# PCK (Percentage of Correct Keypoints):
#    - Percentage of keypoints within threshold distance from ground truth
#    - Usually measured as percentage of head size or torso size
#    - Good: >80%, Excellent: >95%

# OKS (Object Keypoint Similarity):
#    - Similar to IoU but for keypoints
#    - Accounts for keypoint visibility and importance
#    - Used in COCO pose evaluation

# Keypoint Confidence:
#    - How confident the model is about each keypoint
#    - Higher confidence = more reliable detection
#    - Threshold: typically 0.5 or 0.3
# """)

# print("\n💡 INTERPRETATION GUIDE:")
# print("""
# For Robot Pose Detection:
#    - mAP50 > 0.8:  Excellent detection
#    - mAP50 > 0.6:  Good detection  
#    - mAP50 > 0.4:  Acceptable for some applications
#    - mAP50 < 0.4:  Needs improvement

# Focus Areas:
#    - If Box mAP is high but Pose mAP is low: Good at finding robots, bad at keypoints
#    - If both are low: Need more training data or different model
#    - Check confidence scores in individual predictions
# """)

# print(f"\n💾 Detailed results saved in: runs/pose/val/")
# print("📁 Check these files for visual analysis:")
# print("   - val_batch*_pred.jpg (predictions)")
# print("   - val_batch*_labels.jpg (ground truth)")
# print("   - PosePR_curve.png (precision-recall curve)")
# print("   - PoseF1_curve.png (F1 score curve)")
