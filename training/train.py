"""Train, calibrate, evaluate and export the mask-state classifier. See training/README.md.

The held-out test split is used exactly once, at the end. Everything that is *chosen* (epoch, temperature)
is chosen on the validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from mask_detection.detector import YuNetDetector
from mask_detection.mask import DEFAULT_CROP_MARGIN, INPUT_SIZE, LABELS, preprocess, square_crop

SEED = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
YUNET_SHA256 = (
    "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"  # models/face_detection_yunet_2026may.onnx
)
LABEL_IDX = {name: i for i, name in enumerate(LABELS)}
SIZE_BUCKETS = ((0, 24, "<24 px"), (24, 32, "24-31 px"), (32, 64, "32-63 px"), (64, 10_000, ">=64 px"))


@dataclass(frozen=True)
class Face:
    image: str
    box: tuple[float, float, float, float]  # x, y, w, h
    label: int

    @property
    def min_side(self) -> float:
        return min(self.box[2], self.box[3])


def load_faces(root: Path) -> dict[str, list[Face]]:
    by_image: dict[str, list[Face]] = {}
    for xml in sorted((root / "annotations").glob("*.xml")):
        tree = ET.parse(xml).getroot()  # noqa: S314 - trusted, pinned, checksum-verified dataset
        name = tree.find("filename").text  # type: ignore[union-attr]
        faces = []
        for obj in tree.findall("object"):
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (int(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))  # type: ignore[union-attr,arg-type]
            faces.append(Face(name, (x1, y1, x2 - x1, y2 - y1), LABEL_IDX[obj.find("name").text]))  # type: ignore[union-attr,index]
        by_image[name] = faces
    return by_image


def split_images(by_image: dict[str, list[Face]]) -> dict[str, list[str]]:
    """70/15/15 by *image*, reshuffling the seed deterministically until every split has every class."""
    names = sorted(by_image)
    for attempt in range(200):
        rng = random.Random(SEED + attempt)
        order = names[:]
        rng.shuffle(order)
        n = len(order)
        parts = {
            "train": order[: int(0.7 * n)],
            "val": order[int(0.7 * n) : int(0.85 * n)],
            "test": order[int(0.85 * n) :],
        }
        if all({f.label for im in v for f in by_image[im]} == set(range(len(LABELS))) for v in parts.values()):
            return parts
    raise RuntimeError("could not find a split where every split contains every class")


class Images:
    """All images decoded once and kept in memory (853 small images)."""

    def __init__(self, root: Path, names: list[str]) -> None:
        self.data = {n: cv2.imread(str(root / "images" / n)) for n in names}


def augment(img: np.ndarray, face: Face, rng: random.Random) -> np.ndarray:
    x, y, w, h = face.box
    dx, dy = rng.uniform(-0.10, 0.10) * w, rng.uniform(-0.10, 0.10) * h
    s = rng.uniform(0.85, 1.15)
    box = (x + dx - (s - 1) * w / 2, y + dy - (s - 1) * h / 2, w * s, h * s)
    crop = square_crop(img, box, rng.uniform(1.15, 1.5))
    if rng.random() < 0.5:  # simulate low resolution (half of all faces in the data are under 24 px)
        low = rng.randint(14, INPUT_SIZE)
        crop = cv2.resize(cv2.resize(crop, (low, low), interpolation=cv2.INTER_AREA), (INPUT_SIZE, INPUT_SIZE))
    if rng.random() < 0.5:
        crop = cv2.flip(crop, 1)
    a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    crop = cv2.convertScaleAbs(crop, alpha=a, beta=b)
    if rng.random() < 0.2:
        crop = cv2.GaussianBlur(crop, (5, 5), 0)
    return crop


def fixed_crop(img: np.ndarray, face: Face) -> np.ndarray:
    return square_crop(img, face.box, DEFAULT_CROP_MARGIN)


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([preprocess(c) for c in crops], axis=0))


def build_model() -> nn.Module:
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, len(LABELS))
    return m


class Normalised(nn.Module):
    """ImageNet normalisation (+ optional softmax with temperature) around the backbone, so the ONNX takes RGB in [0, 1]."""

    def __init__(self, backbone: nn.Module, temperature: float | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone((x - self.mean) / self.std)
        return logits if self.temperature is None else F.softmax(logits / self.temperature, dim=1)


@torch.no_grad()
def logits_for(model: nn.Module, crops: list[np.ndarray], batch: int = 128) -> torch.Tensor:
    model.eval()
    dev = next(model.parameters()).device
    return torch.cat([model(to_tensor(crops[i : i + batch]).to(dev)).cpu() for i in range(0, len(crops), batch)])


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(len(LABELS)):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - r, c + r)


def fit_temperature(val_logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter > 0 else 0.0


def bucket(min_side: float) -> str:
    return next(name for lo, hi, name in SIZE_BUCKETS if lo <= min_side < hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--report", type=Path, default=Path("docs/EVALUATION.md"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--yunet", type=Path, default=Path("models/face_detection_yunet_2026may.onnx"))
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    by_image = load_faces(args.data)
    splits = split_images(by_image)
    imgs = Images(args.data, sorted(by_image))
    faces = {k: [f for n in v for f in by_image[n]] for k, v in splits.items()}
    counts = {k: np.bincount([f.label for f in v], minlength=len(LABELS)).tolist() for k, v in faces.items()}
    print(
        "images:",
        {k: len(v) for k, v in splits.items()},
        "| faces per class (with, without, incorrect):",
        counts,
        flush=True,
    )

    train_faces = faces["train"]
    class_n = np.bincount([f.label for f in train_faces], minlength=len(LABELS))
    weights = torch.tensor((class_n.sum() / (len(LABELS) * class_n)) ** 0.5, dtype=torch.float32)
    model = Normalised(build_model()).to(DEVICE)
    print("device:", DEVICE, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train_faces) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)
    rng = random.Random(SEED)

    val_crops = [fixed_crop(imgs.data[f.image], f) for f in faces["val"]]
    val_y = np.array([f.label for f in faces["val"]])
    best, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_faces)))
        rng.shuffle(order)
        t0, total = time.time(), 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            x = to_tensor([augment(imgs.data[train_faces[j].image], train_faces[j], rng) for j in idx]).to(DEVICE)
            y = torch.tensor([train_faces[j].label for j in idx]).to(DEVICE)
            loss = F.cross_entropy(model(x), y, weight=weights.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        vlog = logits_for(model, val_crops)
        vf1 = macro_f1(val_y, vlog.argmax(1).numpy())
        vacc = float((vlog.argmax(1).numpy() == val_y).mean())
        print(
            f"epoch {epoch:2d} loss {total / len(order):.4f} val_acc {vacc:.4f} val_macroF1 {vf1:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if vf1 > best:
            best, best_state = vf1, {k: v.clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    # ---- calibration on validation only ----
    vlog = logits_for(model, val_crops)
    temp = fit_temperature(vlog, torch.tensor(val_y))
    val_pre = F.softmax(vlog, 1).numpy()
    val_post = F.softmax(vlog / temp, 1).numpy()
    print(f"temperature {temp:.3f} | val ECE {ece(val_pre, val_y):.4f} -> {ece(val_post, val_y):.4f}", flush=True)

    # ---- held-out test (touched once) ----
    test_crops = [fixed_crop(imgs.data[f.image], f) for f in faces["test"]]
    test_y = np.array([f.label for f in faces["test"]])
    tlog = logits_for(model, test_crops)
    probs = F.softmax(tlog / temp, 1).numpy()
    pred = probs.argmax(1)
    acc = float((pred == test_y).mean())
    lo, hi = wilson(int((pred == test_y).sum()), len(test_y))
    cm = np.zeros((3, 3), int)
    for t, p in zip(test_y, pred, strict=True):
        cm[t, p] += 1
    per_class = {}
    for c, name in enumerate(LABELS):
        tp = cm[c, c]
        per_class[name] = {
            "support": int(cm[c].sum()),
            "precision": float(tp / max(1, cm[:, c].sum())),
            "recall": float(tp / max(1, cm[c].sum())),
        }
    by_size = {}
    for _, _, name in SIZE_BUCKETS:
        m = np.array([bucket(f.min_side) == name for f in faces["test"]])
        if m.any():
            k = int((pred[m] == test_y[m]).sum())
            l2, h2 = wilson(k, int(m.sum()))
            by_size[name] = {"n": int(m.sum()), "accuracy": k / int(m.sum()), "ci95": [l2, h2]}
    test_ece_pre, test_ece_post = ece(F.softmax(tlog, 1).numpy(), test_y), ece(probs, test_y)
    conf = probs.max(1)
    high = conf >= 0.7
    metrics = {
        "test_faces": int(len(test_y)),
        "test_images": len(splits["test"]),
        "accuracy": acc,
        "accuracy_ci95": [lo, hi],
        "macro_f1": macro_f1(test_y, pred),
        "confusion": cm.tolist(),
        "per_class": per_class,
        "by_face_size": by_size,
        "ece_uncalibrated": test_ece_pre,
        "ece_calibrated": test_ece_post,
        "coverage_at_0.7": float(high.mean()),
        "accuracy_when_confident": float((pred[high] == test_y[high]).mean()) if high.any() else None,
    }
    print(
        json.dumps(
            {k: metrics[k] for k in ("accuracy", "accuracy_ci95", "macro_f1", "ece_uncalibrated", "ece_calibrated")}
        ),
        flush=True,
    )

    # ---- end-to-end with the real detector on the test images ----
    det = YuNetDetector(args.yunet, expected_sha256=YUNET_SHA256, max_side=960, upscale_to=1024)  # the service defaults
    matched = correct = fp = gt_total = 0
    by_size_e2e = {n: [0, 0] for _, _, n in SIZE_BUCKETS}
    cal = Normalised(model.backbone, temperature=temp).to(DEVICE)
    cal.eval()
    for name in splits["test"]:
        img = imgs.data[name]
        gts = by_image[name]
        dets = det.detect(img)
        used: set[int] = set()
        gt_total += len(gts)
        for g in gts:
            best_i, best_iou = -1, 0.5
            for i, d in enumerate(dets):
                v = iou(g.box, (d.x, d.y, d.width, d.height))
                if i not in used and v >= best_iou:
                    best_i, best_iou = i, v
            b = bucket(g.min_side)
            by_size_e2e[b][1] += 1
            if best_i >= 0:
                used.add(best_i)
                d = dets[best_i]
                matched += 1
                by_size_e2e[b][0] += 1
                p = int(
                    logits_for(cal, [square_crop(img, (d.x, d.y, d.width, d.height), DEFAULT_CROP_MARGIN)])[0].argmax()
                )
                correct += p == g.label
        fp += len(dets) - len(used)
    e2e = {
        "gt_faces": gt_total,
        "detected": matched,
        "detection_recall": matched / gt_total,
        "false_positive_detections": fp,
        "label_accuracy_on_detected": correct / max(1, matched),
        "detection_recall_by_size": {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in by_size_e2e.items()},
    }
    print("end-to-end:", json.dumps({k: v for k, v in e2e.items() if k != "detection_recall_by_size"}), flush=True)

    # ---- export ONNX (normalisation + calibrated softmax inside the graph) and verify parity ----
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "mask_classifier.onnx"
    cal = cal.cpu()
    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        cal,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = to_tensor(test_crops[:200])
    with torch.no_grad():
        torch_out = cal(sample).numpy()
    ort_out = sess.run(None, {"input": sample.numpy()})[0]
    parity = float(np.abs(torch_out - ort_out).max())
    assert parity < 1e-4, f"ONNX/torch parity failed: {parity}"
    print(f"ONNX parity max abs diff {parity:.2e}", flush=True)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    meta = {
        "labels": list(LABELS),
        "input_size": INPUT_SIZE,
        "crop_margin": DEFAULT_CROP_MARGIN,
        "temperature": temp,
        "dataset": {
            "repo": "hmnshudhmn24/face-mask-detection",
            "revision": "681b307920cad305bdbc042254d933dbb18d1d28",
            "license": "CC0-1.0",
        },
        "seed": SEED,
        "epochs": args.epochs,
        "train_faces_per_class": counts["train"],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnx_sha256": digest,
        "onnx_parity_max_abs_diff": parity,
        "test_metrics": metrics,
        "end_to_end": e2e,
        "splits": {k: len(v) for k, v in splits.items()},
    }
    (args.out / "mask_classifier.json").write_text(json.dumps(meta, indent=2))
    print("sha256", digest, flush=True)


if __name__ == "__main__":
    main()
