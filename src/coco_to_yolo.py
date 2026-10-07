"""
coco_to_yolo.py -- converts the real SIDD COCO annotations (all 4 scenes:
sky/city/mountain/sea) into YOLO training format, so a modern detector
(YOLOv8/YOLO11) can be fine-tuned on the EXACT same dataset the CA-LSTM
paper's Table 6 (Yolov5 0.871/0.503 vs Improved Yolov5+CBAM 0.906/0.519)
was measured on.

Single class: "uav" (category_id 0 in all 4 scene jsons).
"""
import json
import os
import shutil

REPO = "/home/claude/sidd_repo"
SCENES = {
    "sky": f"{REPO}/sky/coco格式/sky2coco",
    "city": f"{REPO}/city/coco格式/cococity",
    "mountain": f"{REPO}/moutain/coco格式/moutain2coco",
    "sea": f"{REPO}/sea/coco格式/sea2coco",
}
OUT = "/home/claude/uav_yolo_dataset"


def convert_split(scene, scene_dir, split):
    img_out = f"{OUT}/images/{split}"
    lbl_out = f"{OUT}/labels/{split}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    with open(f"{scene_dir}/annotations/instances_{split}2017.json") as f:
        d = json.load(f)
    img_by_id = {im["id"]: im for im in d["images"]}
    anns_by_img = {}
    for ann in d["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n = 0
    for img_id, im in img_by_id.items():
        fname = im["file_name"].split("/")[-1]
        src = f"{scene_dir}/{split}2017/{fname}"
        if not os.path.exists(src):
            continue
        W, H = im["width"], im["height"]
        dst_name = f"{scene}_{fname}"
        shutil.copy(src, f"{img_out}/{dst_name}")

        lines = []
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(f"{lbl_out}/{dst_name.replace('.jpg', '.txt')}", "w") as f:
            f.write("\n".join(lines))
        n += 1
    return n


if __name__ == "__main__":
    total = {"train": 0, "val": 0}
    for scene, scene_dir in SCENES.items():
        for split in ["train", "val"]:
            n = convert_split(scene, scene_dir, split)
            total[split] += n
            print(f"{scene}/{split}: {n} images")

    yaml_content = f"""path: {OUT}
train: images/train
val: images/val
names:
  0: uav
"""
    with open(f"{OUT}/data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\nTotal: {total['train']} train, {total['val']} val images -> {OUT}/data.yaml")
