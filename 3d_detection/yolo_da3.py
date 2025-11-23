import os
import cv2
import numpy as np
from da3 import DepthAnything3Wrapper
from yolo import YOLOEWrapper


def attach_depth_to_yolo_centers(centers, depth, orig_h, orig_w):
    """
    centers: YOLOEWrapper.get_bbox_centers 的输出列表
    depth:  DA3 输出的 depth 图, shape = [Hd, Wd]
    orig_h, orig_w: 原始图尺寸

    把每个 center 映射到 depth 分辨率，取一个近似的 depth 值。
    """
    Hd, Wd = depth.shape
    sx = Wd / orig_w
    sy = Hd / orig_h

    out = []
    for obj in centers:
        cx, cy = obj["center"]  # 原图坐标

        # 映射到 depth 图坐标
        u = int(round(cx * sx))  # x -> 列
        v = int(round(cy * sy))  # y -> 行

        u = max(0, min(Wd - 1, u))
        v = max(0, min(Hd - 1, v))

        d = float(depth[v, u])

        new_obj = obj.copy()
        new_obj["depth"] = d        # 深度值
        new_obj["u"] = u            # depth 图上的坐标（可选信息）
        new_obj["v"] = v
        out.append(new_obj)

    return out


def make_iphone13promax_intrinsics(width, height):
    """
    近似 iPhone 13 Pro Max 主摄在给定分辨率下的内参 K
    假设：
        真实焦距 f ≈ 5.7 mm
        传感器 1/1.65"，对角线 ≈ 9.7 mm，4:3
    """
    diag_sensor = 16.0 / 1.65   # mm, 1" 约 16mm
    sensor_width = 4.0 / 5.0 * diag_sensor  # 4:3 传感器下宽度
    f_mm = 5.7

    fx = f_mm / sensor_width * width
    fy = fx  # 假设像素是正方形

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    return K


def make_dummy_phone_intrinsics(width, height):
    fx = fy = 1000.0
    cx = width / 2
    cy = height / 2
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ], dtype=float)


def backproject_pixel_to_3d(u, v, depth, K):
    """
    用相机内参把像素点 (u, v) + depth 反投影到相机坐标系
    注意：这里的 (u, v) 是【原图坐标】，不是 depth 图坐标
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (u - cx) / fx * depth
    Y = (v - cy) / fy * depth
    Z = depth

    return X, Y, Z


if __name__ == "__main__":
    img_path = "data/0.jpg"

    # 1. 初始化 YOLOE
    # names = ["watermelon", "melon", "apple", "banana", "orange",
    #          "grape", "pineapple", "strawberry", "blueberry", "kiwi"]

    yolo = YOLOEWrapper(
        model_path="yoloe-11l-seg.pt",
    )

    # 2. 初始化 DepthAnything3
    da3 = DepthAnything3Wrapper(device="cuda")

    # 例如：你只想查找 "orange"
    target_label = "orange"
    print(f"\n=== Re-run YOLO with custom classes: [{target_label}] ===")
    if isinstance(target_label, str):
        target_label = [target_label]
    yolo.set_classes(target_label)  # ⭐ 动态设置成只找这个类别

    # 3. YOLO 检测
    results = yolo.predict(img_path)
    parsed = yolo.extract_bboxes(results)
    centers = yolo.get_bbox_centers(parsed)

    print("\n=== YOLO BBox Centers ===")
    for c in centers:
        print(f"label={c['label']}, center={c['center']}, conf={c['conf']:.3f}")

    # 4. DA3 depth 预测
    rgb, depth, conf, K_da3, E_da3 = da3.predict(img_path)

    # 5. 原图尺寸（840x630）
    img_bgr = cv2.imread(img_path)
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"\nOriginal image size: {orig_w} x {orig_h}")
    print(f"Depth map size: {depth.shape[1]} x {depth.shape[0]}")

    # 6. iPhone 13 Pro Max dummy 内参（基于 840x630）
    K_iphone = make_iphone13promax_intrinsics(orig_w, orig_h)
    print("\nApprox iPhone13 Pro Max intrinsics (for this resolution):\n", K_iphone)

    # 7. center -> depth
    centers_with_depth = attach_depth_to_yolo_centers(
        centers, depth, orig_h, orig_w
    )

    # 8. 利用中心点 + depth + K_iphone 求 3D
    print("\n=== YOLO centers with depth + 3D (iPhone13 Pro Max intrinsics) ===")
    for obj in centers_with_depth:
        label = obj["label"]
        (cx, cy) = obj["center"]   # 原图坐标
        d = obj["depth"]
        conf = obj["conf"]

        X, Y, Z = backproject_pixel_to_3d(cx, cy, d, K_iphone)

        print(f"{label:10s} | pixel=({cx:.1f},{cy:.1f}) | depth={d:.3f} "
              f"| 3D=({X:.3f}, {Y:.3f}, {Z:.3f}) | conf={conf:.3f}")

    # 9. 可选：保存可视化
    os.makedirs("single_output", exist_ok=True)
    yolo.save_visualization(results, "single_output/yolo_result.jpg")
    da3.save_all(depth, rgb, out_dir="single_output")

    print("\nAll done ✅")
