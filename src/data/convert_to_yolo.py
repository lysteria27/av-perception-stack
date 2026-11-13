"""Convert BDD100K detection labels to a YOLO-format subset."""
import argparse, json, random, shutil
from pathlib import Path
from tqdm import tqdm

CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
CLS_ID = {name: i for i, name in enumerate(CLASSES)}
W, H = 1280, 720  # BDD100K image size


def convert_frame(fr):
    lines = []
    for lab in fr.get("labels") or []:
        box, cat = lab.get("box2d"), lab.get("category")
        if box is None or cat not in CLS_ID:
            continue
        x1 = max(0.0, min(box["x1"], box["x2"]))
        x2 = min(float(W), max(box["x1"], box["x2"]))
        y1 = max(0.0, min(box["y1"], box["y2"]))
        y2 = min(float(H), max(box["y1"], box["y2"]))
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        lines.append(f"{CLS_ID[cat]} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
    return lines


def build_split(labels_json, img_root, out_root, split, n, seed):
    with open(labels_json) as f:
        frames = json.load(f)
    index = {p.name: p for p in Path(img_root).rglob("*.jpg")}
    present = [fr for fr in frames if fr["name"] in index]
    print(f"[{split}] labeled: {len(frames)} | images on disk: {len(index)} | both: {len(present)}")

    random.Random(seed).shuffle(present)
    chosen = present[:n]
    img_out, lbl_out = out_root / "images" / split, out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for fr in tqdm(chosen, desc=split):
        shutil.copy2(index[fr["name"]], img_out / fr["name"])
        (lbl_out / (Path(fr["name"]).stem + ".txt")).write_text("\n".join(convert_frame(fr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--val_labels", required=True)
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--out", default="/kaggle/working/datasets/bdd")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    build_split(a.train_labels, a.train_images, out, "train", a.n_train, a.seed)
    build_split(a.val_labels, a.val_images, out, "val", a.n_val, a.seed)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "bdd.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print("wrote", out / "bdd.yaml")


if __name__ == "__main__":
    main()