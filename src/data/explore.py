"""Phase 1: explore BDD100K detection labels and class balance."""
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # draw to files, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import wandb


def load_frames(label_path):
    with open(label_path, "r") as f:
        return json.load(f)


def to_dataframes(frames):
    """Flatten nested JSON into two tables: one row per image, one per box."""
    img_rows, box_rows = [], []
    for fr in frames:
        attrs = fr.get("attributes") or {}
        n_boxes = 0
        for lab in fr.get("labels") or []:
            box = lab.get("box2d")
            if box is None:          # skip lanes, drivable area, etc.
                continue
            n_boxes += 1
            w = box["x2"] - box["x1"]
            h = box["y2"] - box["y1"]
            lab_attrs = lab.get("attributes") or {}
            box_rows.append({
                "image": fr["name"],
                "category": lab["category"],
                "w": w, "h": h, "area": w * h,
                "occluded": lab_attrs.get("occluded"),
                "truncated": lab_attrs.get("truncated"),
            })
        img_rows.append({
            "image": fr["name"],
            "weather": attrs.get("weather"),
            "scene": attrs.get("scene"),
            "timeofday": attrs.get("timeofday"),
            "n_boxes": n_boxes,
        })
    return pd.DataFrame(img_rows), pd.DataFrame(box_rows)


def class_summary(boxes):
    s = boxes.groupby("category").agg(
        count=("area", "size"),
        median_area=("area", "median"),
        pct_small=("area", lambda a: (a < 32 * 32).mean() * 100),
    )
    s["pct_of_total"] = s["count"] / s["count"].sum() * 100
    return s.sort_values("count", ascending=False).round(2).reset_index()


def make_plots(images, boxes, summary, out_dir):
    paths = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["category"], summary["count"])
    ax.set_yscale("log")
    ax.set_title("Boxes per class (log scale)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    paths["class_counts"] = out_dir / "class_counts.png"
    fig.savefig(paths["class_counts"], dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["weather", "scene", "timeofday"]):
        images[col].value_counts().plot.bar(ax=ax, title=col)
    fig.tight_layout()
    paths["attributes"] = out_dir / "attributes.png"
    fig.savefig(paths["attributes"], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    images["n_boxes"].plot.hist(bins=40, ax=ax, title="Boxes per image")
    fig.tight_layout()
    paths["boxes_per_image"] = out_dir / "boxes_per_image.png"
    fig.savefig(paths["boxes_per_image"], dpi=120)
    plt.close(fig)
    return paths


def draw_samples(frames, image_index, out_dir, n=4):
    rng = random.Random(0)  # fixed seed: same samples every run
    out_paths = []
    for fr in rng.sample(frames, n):
        path = image_index.get(fr["name"])
        if path is None:
            continue
        img = cv2.imread(str(path))  # OpenCV loads as BGR
        for lab in fr.get("labels") or []:
            b = lab.get("box2d")
            if b is None:
                continue
            p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, lab["category"], (p1[0], max(p1[1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out = out_dir / f"sample_{fr['name']}"
        cv2.imwrite(str(out), img)
        out_paths.append(out)
    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="path to detection labels JSON")
    ap.add_argument("--images", default=None, help="folder containing images")
    ap.add_argument("--out", default="results/eda")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(project="av-perception", job_type="eda",
                     config={"dataset": "bdd100k", "split": args.split})

    frames = load_frames(args.labels)
    images, boxes = to_dataframes(frames)
    summary = class_summary(boxes)
    print(f"{len(images)} images, {len(boxes)} boxes")
    print(summary.to_string(index=False))

    plots = make_plots(images, boxes, summary, out_dir)
    log = {"class_summary": wandb.Table(dataframe=summary),
           "num_images": len(images), "num_boxes": len(boxes)}
    log.update({k: wandb.Image(str(v)) for k, v in plots.items()})

    if args.images:
        index = {p.name: p for p in Path(args.images).rglob("*.jpg")}
        samples = draw_samples(frames, index, out_dir)
        log["samples"] = [wandb.Image(str(p)) for p in samples]

    wandb.log(log)
    run.finish()


if __name__ == "__main__":
    main()