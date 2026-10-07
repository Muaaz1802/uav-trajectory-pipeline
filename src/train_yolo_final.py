"""
train_yolo_final.py -- run this on a GPU (Google Colab free T4 is enough)
to get final, defensible numbers for your "beat the paper" claim.

Runtime: ~15-20 min on a Colab T4 for 50 epochs at imgsz=640, vs. multiple
hours on a CPU-only machine. Don't try to run the full version locally on a
laptop without a GPU -- use Colab.

WHAT THIS REPRODUCES: Table 6 of the CA-LSTM paper (Dang et al., 2023),
on the exact same SIDD dataset (github.com/Dang-zy/SIDD):
    Yolov5 (their baseline):          mAP@0.5 = 0.871, mAP@0.5:0.95 = 0.503
    Improved Yolov5+CBAM (their method): mAP@0.5 = 0.906, mAP@0.5:0.95 = 0.519

Colab setup:
    !pip install ultralytics -q
    !git clone https://github.com/Dang-zy/SIDD.git
    # then run coco_to_yolo.py (included in this package) pointed at the
    # cloned repo, then run this script.
"""
from ultralytics import YOLO

DATA_YAML = "/content/uav_yolo_dataset/data.yaml"  # adjust path on Colab

if __name__ == "__main__":
    # YOLO11n: newer architecture than the paper's 2023-era YOLOv5, same
    # "lightweight, fast" spirit -- a fair, defensible upgrade to compare.
    model = YOLO("yolo11n.pt")  # pretrained on COCO, fine-tuned below

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,          # match the paper's stated 640x640 input
        batch=16,
        device=0,           # GPU
        patience=20,
        project="runs",
        name="sidd_yolo11n",
        plots=True,
    )

    metrics = model.val(data=DATA_YAML, imgsz=640)
    print("\n=== FINAL RESULT (compare directly against paper Table 6) ===")
    print(f"mAP@0.5     = {metrics.box.map50:.3f}   (paper's best: 0.906)")
    print(f"mAP@0.5:0.95 = {metrics.box.map:.3f}   (paper's best: 0.519)")

    # Optional: also fine-tune a genuine YOLOv5n for an apples-to-apples
    # "same architecture family, more epochs/modern training tricks" story,
    # in case a reviewer wants strict architecture parity with the paper:
    # model2 = YOLO("yolov5n.pt")
    # model2.train(data=DATA_YAML, epochs=50, imgsz=640, batch=16, device=0)
