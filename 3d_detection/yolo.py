import cv2
import numpy as np
from ultralytics import YOLOE


class YOLOEWrapper:
    def __init__(self, model_path="yoloe-11l-seg.pt"):
        """
        初始化 YOLOE 模型（不再固定类别）
        """
        self.model = YOLOE(model_path)

        # 模型内置类别名（可能是 COCO 或自定义训练数据）
        self.default_class_names = self.model.names

        # 当前 active 类别（如果不设置，则用默认类别）
        self.active_class_names = None

    # -----------------------------
    # ⭐ 动态设置类别（关键功能）
    # -----------------------------
    def set_classes(self, class_names):
        """
        动态设置 YOLOE 的检测类别
        """
        # class_names 必须为列表（即使只有一个元素）
        if isinstance(class_names, str):
            class_names = [class_names]

        self.active_class_names = class_names

        # 使用 CLIP 文本编码
        text_pe = self.model.get_text_pe(class_names)
        self.model.set_classes(class_names, text_pe)

        print(f"[YOLO] Active classes set to: {class_names}")

    # -----------------------------
    # 执行推理
    # -----------------------------
    def predict(self, image_path):
        """
        执行推理
        """
        # 如果没有设置 active classes，就使用默认训练类别
        if self.active_class_names is None:
            print("[YOLO] Using default model classes")
        else:
            print(f"[YOLO] Detecting only: {self.active_class_names}")

        return self.model.predict(image_path)

    # -----------------------------
    # 解析 bbox
    # -----------------------------
    def extract_bboxes(self, results):
        parsed = []

        for r in results:
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            # 如果你 set_classes，则 cls 索引是新类别表的索引
            names = (
                self.active_class_names
                if self.active_class_names is not None
                else self.default_class_names
            )

            for box, c, score in zip(xyxy, cls, conf):
                x1, y1, x2, y2 = box
                label = names[c]

                parsed.append({
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "label": label,
                    "conf": float(score)
                })

        return parsed

    def get_bbox_centers(self, parsed_bboxes):
        centers = []
        for obj in parsed_bboxes:
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            centers.append({
                "label": obj["label"],
                "center": (cx, cy),
                "bbox": obj["bbox"],
                "conf": obj["conf"]
            })
        return centers

    def save_visualization(self, results, save_path):
        results[0].save(filename=save_path)
        print(f"[✔] Saved YOLO visualization to {save_path}")


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    img_path = "data/0.jpg"

    # =========================================
    # 示例 1：不传自定义类别，用默认类别
    # =========================================
    print("\n================ 示例 1：默认类别 ================")
    yolo_default = YOLOEWrapper("yoloe-11l-seg.pt")

    # ---- Step 1: 直接用默认类别推理 ----
    results_def = yolo_default.predict(img_path)
    parsed_def = yolo_default.extract_bboxes(results_def)

    print("\n=== Detected Boxes (default classes) ===")
    for obj in parsed_def:
        print(obj)

    centers_def = yolo_default.get_bbox_centers(parsed_def)
    print("\n=== BBox Centers (default classes) ===")
    for c in centers_def:
        print(f"label={c['label']}, center={c['center']}, conf={c['conf']:.3f}")

    yolo_default.save_visualization(results_def, "single_output/yolo_default_result.jpg")

    # ---- Step 2: 后期再加单个标签 [orange]，重新检测 ----
    print("\n---- Now set classes to ['orange'] and run again ----")
    yolo_default.set_classes(["orange"])

    results_orange = yolo_default.predict(img_path)
    parsed_orange = yolo_default.extract_bboxes(results_orange)

    print("\n=== Detected Boxes (after set_classes(['orange'])) ===")
    for obj in parsed_orange:
        print(obj)

    centers_orange = yolo_default.get_bbox_centers(parsed_orange)
    print("\n=== BBox Centers (after set_classes(['orange'])) ===")
    for c in centers_orange:
        print(f"label={c['label']}, center={c['center']}, conf={c['conf']:.3f}")

    yolo_default.save_visualization(results_orange, "single_output/yolo_orange_result.jpg")

    # =========================================
    # 示例 2：动态设置自定义类别（水果）
    # =========================================
    print("\n================ 示例 2：自定义类别 ================")

    names = ["watermelon", "melon", "apple", "banana", "orange",
             "grape", "pineapple", "strawberry", "blueberry", "kiwi"]

    yolo_fruits = YOLOEWrapper("yoloe-11l-seg.pt")

    # ⭐ 动态设置类别（关键）
    yolo_fruits.set_classes(names)

    results_fruit = yolo_fruits.predict(img_path)
    parsed_fruit = yolo_fruits.extract_bboxes(results_fruit)

    print("\n=== Detected Boxes (custom fruit classes) ===")
    for obj in parsed_fruit:
        print(obj)

    centers_fruit = yolo_fruits.get_bbox_centers(parsed_fruit)
    print("\n=== BBox Centers (custom fruit classes) ===")
    for c in centers_fruit:
        print(f"label={c['label']}, center={c['center']}, conf={c['conf']:.3f}")

    yolo_fruits.save_visualization(results_fruit, "single_output/yolo_fruit_result.jpg")

    print("\nAll demos done ✅")
