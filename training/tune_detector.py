"""Choose face-detector settings on the VALIDATION images only (never the test split).

Sweeps YuNet's score threshold and the small-image upscaling target, reporting face recall (IoU >= 0.5 with the
ground truth) and false-positive detections per image. Uses the same image-level split as train.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from train import Images, bucket, iou, load_faces, split_images  # noqa: E402

from mask_detection.detector import YuNetDetector  # noqa: E402

YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


def main() -> None:
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/maskdata")
    yunet = Path(sys.argv[2] if len(sys.argv) > 2 else "models/face_detection_yunet_2026may.onnx")
    by_image = load_faces(data)
    names = split_images(by_image)["val"]
    imgs = Images(data, names)
    print(f"validation images: {len(names)}, faces: {sum(len(by_image[n]) for n in names)}")
    print(f"{'score':>6} {'upscale_to':>10} {'recall':>7} {'recall>=24px':>13} {'FP/img':>7}")
    for score in (0.5, 0.6, 0.7):
        for up in (0, 640, 1024, 1600):
            det = YuNetDetector(
                yunet, expected_sha256=YUNET_SHA256, score_threshold=score, max_side=2048, upscale_to=up
            )
            hit = tot = hit24 = tot24 = fp = 0
            for n in names:
                dets = det.detect(imgs.data[n])
                used: set[int] = set()
                for g in by_image[n]:
                    best_i, best = -1, 0.5
                    for i, d in enumerate(dets):
                        v = iou(g.box, (d.x, d.y, d.width, d.height))
                        if i not in used and v >= best:
                            best_i, best = i, v
                    tot += 1
                    tot24 += g.min_side >= 24
                    if best_i >= 0:
                        used.add(best_i)
                        hit += 1
                        hit24 += g.min_side >= 24
                fp += len(dets) - len(used)
            print(f"{score:>6} {up:>10} {hit / tot:>7.3f} {hit24 / max(1, tot24):>13.3f} {fp / len(names):>7.2f}")


if __name__ == "__main__":
    _ = (cv2, bucket)
    main()
