import os, sys, cv2, time, numpy as np
from pathlib import Path
from functools import partial

import torch
_original_load = torch.load
torch.load = partial(_original_load, weights_only=False)

try:
    from ultralyticsplus import YOLO
except ImportError:
    print("[!] Installing ultralyticsplus …")
    os.system(f"{sys.executable} -m pip install ultralyticsplus==0.0.23 ultralytics==8.0.21 -q")
    from ultralyticsplus import YOLO

INPUT_DIR     = Path("images")
OUTPUT_DIR    = Path("output_images")
MODEL_PATH    = "best.pt"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF_THRESH   = 0.10
IOU_THRESH    = 0.45
MAX_DET       = 300

BOX_COLOR  = (0, 0, 255)
MASK_COLOR = (30, 30, 220)
TEXT_COLOR = (255, 255, 255)
LABEL_BG   = (0, 0, 180)
BOX_THICK  = 3
FONT       = cv2.FONT_HERSHEY_SIMPLEX

def load_model() -> YOLO:
    if not Path(MODEL_PATH).exists():
        print(f"  [!] '{MODEL_PATH}' not found. Run download_model.py first.")
        sys.exit(1)
    print(f"  Loading pothole model  : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.overrides["conf"]         = CONF_THRESH
    model.overrides["iou"]          = IOU_THRESH
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"]      = MAX_DET
    print("  [Done] Model ready  (mAP@0.5 = 0.995)\n")
    return model

def annotate(image: np.ndarray, result) -> tuple[np.ndarray, int]:
    out   = image.copy()
    count = 0

    if result.boxes is not None:
        for idx, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)

            lbl = f"Pothole  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.65, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), LABEL_BG, -1)
            cv2.putText(out, lbl, (x1 + 4, max(0, y1 - 5)), FONT, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)
            count += 1

            if result.masks is not None and hasattr(result.masks, 'segments'):
                if idx < len(result.masks.segments):
                    pts = result.masks.segments[idx]
                    arr = np.array(pts, dtype=np.float32)
                    if arr.size > 0:
                        if np.max(arr) <= 1.0:
                            h, w = out.shape[:2]
                            arr[:, 0] *= w
                            arr[:, 1] *= h
                        arr = arr.astype(np.int32)
                        
                        overlay = out.copy()
                        cv2.fillPoly(overlay, [arr], MASK_COLOR)
                        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

    badge = f"  Potholes: {count}  "
    (bw2, _), _ = cv2.getTextSize(badge, FONT, 0.75, 2)
    cv2.rectangle(out, (8, 8), (16 + bw2, 46), (15, 15, 15), -1)
    cv2.putText(out, badge, (12, 38), FONT, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    return out, count

def process(model: YOLO, path: Path) -> tuple[np.ndarray, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    results = model.predict(str(path), verbose=False)
    return annotate(img, results[0])

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXT)

    if not files:
        print(f"\n[!] No images in '{INPUT_DIR}/'. Add road images and re-run.\n")
        return

    print(f"\n{'='*58}")
    print(f"  Pothole Detection System  |  YOLOv8n  |  best.pt")
    print(f"  Input  : {len(files)} image(s) in '{INPUT_DIR}/'")
    print(f"  Output : '{OUTPUT_DIR}/'")
    print(f"{'='*58}\n")

    model       = load_model()
    total, t0   = 0, time.time()

    for i, p in enumerate(files, 1):
        print(f"  [{i:02d}/{len(files)}] {p.name}", end="  →  ", flush=True)
        try:
            annotated, n = process(model, p)
            out = OUTPUT_DIR / p.name
            cv2.imwrite(str(out), annotated)
            print(f"{n} pothole(s) detected  |  saved: {out.name}")
            total += n
        except Exception as e:
            print(f"ERROR — {e}")

    print(f"\n{'='*58}")
    print(f"  Done! {len(files)} image(s) in {time.time()-t0:.1f}s")
    print(f"  Total potholes detected : {total}")
    print(f"  Results saved in        : '{OUTPUT_DIR}/'")
    print(f"{'='*58}\n")

if __name__ == "__main__":
    main()
