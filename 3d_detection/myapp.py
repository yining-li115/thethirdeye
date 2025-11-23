from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import cv2
import numpy as np
import tempfile

# ======= 你的两个 wrapper 类 =======
from da3 import DepthAnything3Wrapper
from yolo import YOLOEWrapper


# ======= 工具函数：附加 depth、内参、反投影 =======
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


def backproject_pixel_to_3d(u, v, depth, K):
    """
    用相机内参把像素点 (u, v) + depth 反投影到相机坐标系
    这里的 (u, v) 是原图坐标
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = (u - cx) / fx * depth
    Y = (v - cy) / fy * depth
    Z = depth

    return X, Y, Z


# ======= 初始化 FastAPI =======
app = FastAPI()

# ======= 全局初始化模型（只加载一次，避免每次请求都重新 load） =======
# NAMES = ["watermelon", "melon", "apple", "banana", "orange",
#          "grape", "pineapple", "strawberry", "blueberry", "kiwi"]

YOLO_MODEL_PATH = "yoloe-11l-seg.pt"

# 初始化 YOLO
yolo = YOLOEWrapper(
    model_path=YOLO_MODEL_PATH,
    # custom_classes=NAMES
)

# 初始化 DepthAnything3
da3 = DepthAnything3Wrapper(device="cuda")  # 没有 GPU 就改 "cpu"


# ======= 核心函数：bytes -> 推理 -> 结构化结果 =======
def run_model_on_image(image_bytes: bytes):
    """
    输入: 图片的二进制数据
    输出: 列表，每个元素是一个物体的信息：
      {
        "label": str,
        "center_pixel": [cx, cy],
        "depth": d,
        "xyz_camera": [X, Y, Z],
        "conf": conf,
        "depth_uv": [u, v]
      }
    """

    # 1. 把 bytes 临时写成一个文件
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # 2. YOLO 检测
        results = yolo.predict(tmp_path)
        parsed = yolo.extract_bboxes(results)
        centers = yolo.get_bbox_centers(parsed)  # label / center / conf

        if not centers:
            # 没有检测到任何目标，这里也可以选择保存原图或空结果
            return []

        # 3. DepthAnything3 深度
        rgb, depth, conf_map, K_da3, E_da3 = da3.predict(tmp_path)

        # 4. 原图大小
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise RuntimeError("Failed to read image back from temp file.")
        orig_h, orig_w = img_bgr.shape[:2]

        # ====== 4.5 保存 YOLO 可视化结果 + Depth 输出 ======
        # 用临时文件名的一部分作为 id，避免覆盖
        base_name = os.path.splitext(os.path.basename(tmp_path))[0]
        out_dir = os.path.join("app_outputs", base_name)
        os.makedirs(out_dir, exist_ok=True)

        # 保存 YOLO 检测可视化
        try:
            yolo_vis_path = os.path.join(out_dir, "yolo_result.jpg")
            yolo.save_visualization(results, yolo_vis_path)
            print(f"[SAVE] YOLO visualization saved to {yolo_vis_path}")
        except Exception as e:
            print(f"[SAVE] Failed to save YOLO visualization: {e}")

        # 保存 depth / rgb / 其他（沿用你之前的 da3.save_all）
        try:
            da3.save_all(depth, rgb, out_dir=out_dir)
            print(f"[SAVE] DepthAnything3 outputs saved to {out_dir}")
        except Exception as e:
            print(f"[SAVE] Failed to save DA3 outputs: {e}")
        # ====== 保存结束 ======

        # 5. iPhone 13 Pro Max 内参（近似）
        K_iphone = make_iphone13promax_intrinsics(orig_w, orig_h)

        # 6. center -> depth
        centers_with_depth = attach_depth_to_yolo_centers(
            centers, depth, orig_h, orig_w
        )

        # 7. 反投影到 3D
        objects_out = []
        for obj in centers_with_depth:
            label = obj["label"]
            (cx, cy) = obj["center"]
            d = obj["depth"]
            conf = obj["conf"]

            X, Y, Z = backproject_pixel_to_3d(cx, cy, d, K_iphone)

            objects_out.append({
                "label": label,
                "center_pixel": [float(cx), float(cy)],
                "depth": float(d),
                "xyz_camera": [float(X), float(Y), float(Z)],
                "conf": float(conf),
                "depth_uv": [int(obj["u"]), int(obj["v"])],
            })

        return objects_out

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ======= 原来的 /detect：返回所有 objects，方便调试 =======
@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        objects = run_model_on_image(image_bytes)

        result = {
            "filename": file.filename,
            "num_objects": len(objects),
            "objects": objects,
        }
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ======= 新增 /locate：对标你朋友的接口格式 =======
@app.post("/locate")
def locate_target(
    image: UploadFile = File(...),          # 对应 -F "image=@test_frame.jpg"
    target_object: str = Form(...)          # 对应 -F "target_object=apple"
):
    """
    接口约定（和你朋友一致）：
      POST /locate
      -F "image=@xxx.jpg"
      -F "target_object=apple"

    返回（始终 HTTP 200）：
    {
      "found": true/false,
      "target_position": { "x": ..., "y": ..., "z": ... },
      "camera_position": { "x": 0.0, "y": 0.0, "z": 0.0 },
      "camera_orientation": { "yaw": 0.0, "pitch": 0.0 }
    }
    """
    try:
        image_bytes = image.file.read()

        # 打印收到的内容，方便你看
        print(
            f"[LOCATE] received: file={image.filename}, "
            f"size={len(image_bytes)} bytes, "
            f"target_object={target_object}",
            flush=True
        )

        # ⭐ 关键：先告诉 YOLO 你想找什么
        yolo.set_classes(target_object)

        objects = run_model_on_image(image_bytes)

        # 1. 过滤出 label 等于 target_object 的物体
        candidates = [obj for obj in objects if obj["label"] == target_object]

        if not candidates:
            # ❗ 没找到目标：仍然返回 200 + 固定格式 + 标记 found=false
            result = {
                "found": False,
                "target_position": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                },
                "camera_position": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                },
                "camera_orientation": {
                    "yaw": 0.0,
                    "pitch": 0.0,
                },
            }
            print(
                f"[LOCATE] target '{target_object}' NOT found, "
                f"response={result}",
                flush=True
            )
            return JSONResponse(result)   # 不再写 status_code，默认就是 200

        # 2. 找到了：选择置信度最高的那个
        best = max(candidates, key=lambda o: o["conf"])
        X, Y, Z = best["xyz_camera"]

        # 现在先保留你原来的 target_position = (0,0,0)
        # 以后想用真实坐标时，把 0.0 改成 X/Y/Z 即可
        result = {
            "found": True,
            "target_position": {
                "x": float(X),
                "y": float(Y),
                "z": float(Z),
            },
            "camera_position": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            "camera_orientation": {
                "yaw": 0.0,
                "pitch": 0.0,
            },
        }

        print(f"[LOCATE] target '{target_object}' found, best object: {best}")
        print(f"[LOCATE] response JSON: {result}", flush=True)

        return JSONResponse(result)

    except Exception as e:
        print(f"[LOCATE] ERROR: {e}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
