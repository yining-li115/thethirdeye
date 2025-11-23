import os
import torch
import numpy as np
import cv2
import open3d as o3d 
import glob
import matplotlib.pyplot as plt
from depth_anything_3.api import DepthAnything3


class DepthAnything3Wrapper:
    """
    一个整合的 DepthAnything3 单图推理类
    提供：
        - 模型加载
        - 单图推理
        - 保存 depth.npy / colormap / rgb
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        print(f"[DA3] Loading DepthAnything3 model on {self.device} ...")
        self.model = DepthAnything3.from_pretrained(
            "depth-anything/DA3NESTED-GIANT-LARGE"
        ).to(self.device)
        print("[DA3] Model loaded.")

    def predict(self, image_path):
        """
        对单张图片做推理
        返回：
            rgb_proc : [H, W, 3] uint8
            depth    : [H, W]    float32
            conf     : [H, W]    float32
            K        : [3, 3]
            E        : [3, 4]
        """
        prediction = self.model.inference([image_path])

        rgb_proc = prediction.processed_images[0]
        depth = prediction.depth[0]
        conf = prediction.conf[0]
        K = prediction.intrinsics[0]
        E = prediction.extrinsics[0]

        return rgb_proc, depth, conf, K, E
    
    # ----------------------------- 多图推理 -----------------------------
    def predict_seperate_images(self, image_paths):
        pred = self.model.inference(image_paths)

        rgbs = pred.processed_images
        depths = pred.depth
        confs = pred.conf
        Ks = pred.intrinsics
        Es = pred.extrinsics

        return rgbs, depths, confs, Ks, Es

    @torch.no_grad()
    def reconstruct_from_seperate_images(
        self,
        image_paths,
        save_path="reconstruction/scene",
        visualize=True,
    ):
        """
        多张图一起输入 DA3，做一次 any-view 推理 + 重建一个 3D 场景

        image_paths: list[str]
            - 同一个场景的多帧 / 多视角
            - 一定要一起传进去，DA3 才能做多视角一致性估计
        """
        # ⭐⭐ 关键：多张图一次性传给 DA3
        prediction = self.model.inference(image_paths)

        rgbs   = prediction.processed_images   # [N, H, W, 3] uint8
        depths = prediction.depth              # [N, H, W]    float32
        Ks     = prediction.intrinsics         # [N, 3, 3]
        Es     = prediction.extrinsics         # [N, 3, 4]

        # 利用上面的 reconstruct_scene 做点云融合
        return self.reconstruct_scene(
            rgbs=rgbs,
            depths=depths,
            Ks=Ks,
            Es=Es,
            save_path=save_path,
            visualize=visualize,
        )
    
    # 原来的单图、多图 predict 可以保留，这里重点是统一 3D 重建：
    def reconstruct_from_images(
        self,
        image_paths,
        export_dir="reconstruction",
        export_format="gs_ply",   # 或 "glb", "npz", "mini_npz", "gs_ply", "gs_video"
        visualize=True,
    ):
        os.makedirs(export_dir, exist_ok=True)

        # 关键：让 DA3 自己做多视角一致 3D 场景重建并导出
        _ = self.model.inference(
            image_paths,
            export_dir=export_dir,
            export_format=export_format,
            infer_gs=("gs" in export_format),  # 如果你用 gs_ply / gs_video 才需要 True
        )

        # 下面这部分是我们**读取导出的 ply**，在 python 里直接用 o3d 看
        # 官方 export 的文件命名规则大概类似：scene.ply / point_cloud.ply 之类，
        # 这里你可以根据实际 export_dir 中的文件名改一下
        ply_candidates = [
            f for f in os.listdir(export_dir) if f.lower().endswith(".ply")
        ]
        if not ply_candidates:
            print("[DA3] No .ply found in export_dir, check export_format & path.")
            return

        ply_path = os.path.join(export_dir, ply_candidates[0])
        print(f"[DA3] Loading exported PLY: {ply_path}")

        pcd = o3d.io.read_point_cloud(ply_path)

        if visualize:
            print("[DA3] Visualizing fused 3D scene from multiple views ...")
            o3d.visualization.draw_geometries([pcd])

        return pcd

    def save_depth_npy(self, depth, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, depth)
        print(f"[DA3] depth npy saved: {save_path}")

    def save_depth_colormap(self, depth, save_path, cmap="turbo"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(depth, cmap=cmap)
        plt.colorbar()
        plt.title("Depth")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[DA3] depth colormap saved: {save_path}")

    def save_rgb(self, rgb_proc, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        bgr = cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
        print(f"[DA3] rgb saved: {save_path}")

    def save_all(self, depth, rgb, out_dir="output"):
        """
        一次性保存所有文件：depth.npy，depth.png，rgb.png
        """
        os.makedirs(out_dir, exist_ok=True)
        self.save_depth_npy(depth, os.path.join(out_dir, "depth.npy"))
        self.save_depth_colormap(depth, os.path.join(out_dir, "depth_colormap.png"))
        self.save_rgb(rgb, os.path.join(out_dir, "rgb.png"))
        print(f"[DA3] All results saved to {out_dir}")

    # ------------------------------------------------------------------
    # 深度 -> 相机坐标系点云
    # ------------------------------------------------------------------
    @staticmethod
    def depth_to_pointcloud(depth, K, device="cuda"):
        """
        depth: [H, W]  numpy 或 torch
        K    : [3, 3]
        返回: pts_cam [N, 3]
        """
        depth_t = torch.as_tensor(depth, device=device, dtype=torch.float32)
        K_t = torch.as_tensor(K, device=device, dtype=torch.float32)

        H, W = depth_t.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        fx, fy = K_t[0, 0], K_t[1, 1]
        cx, cy = K_t[0, 2], K_t[1, 2]

        Z = depth_t
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy

        pts = torch.stack([X, Y, Z], dim=-1)  # [H, W, 3]
        return pts.reshape(-1, 3)  # [N, 3]

    # ------------------------------------------------------------------
    # 相机坐标 -> 世界坐标
    # ------------------------------------------------------------------
    @staticmethod
    def cam_to_world(pts_cam, E, device="cuda"):
        """
        pts_cam: [N, 3] (torch, device 同上)
        E      : [3, 4]
        返回: pts_world [N, 3]
        """
        E_t = torch.as_tensor(E, device=device, dtype=torch.float32)
        R = E_t[:, :3]  # [3, 3]
        t = E_t[:, 3]   # [3]

        pts_world = pts_cam @ R.T + t  # [N, 3]
        return pts_world

    def reconstruct_scene(
        self,
        rgbs,
        depths,
        Ks,
        Es,
        save_path="reconstruction/scene",
        visualize=True,
    ):
        """
        rgbs, depths, Ks, Es 为多个视角的列表（顺序要对应）
        保存:
            - <save_path>.npy        : 所有世界坐标点 (N, 3)
            - <save_path>_rgb.npy    : 所有点的颜色 (N, 3)，范围 [0, 1]
            - <save_path>.ply        : 带颜色的点云 (Open3D)
        如果 visualize=True，会调用 Open3D 弹出窗口显示点云
        """
        device = str(self.device)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        all_points = []
        all_colors = []

        for rgb, depth, K, E in zip(rgbs, depths, Ks, Es):
            # 1) 深度 -> 相机坐标点云
            pts_cam = self.depth_to_pointcloud(depth, K, device=device)  # [N, 3]

            # 2) 相机坐标 -> 世界坐标
            pts_world = self.cam_to_world(pts_cam, E, device=device)     # [N, 3]
            all_points.append(pts_world.cpu())

            # 3) 颜色展平，和点一一对应
            # rgb: [H, W, 3] uint8 -> float32 [0, 1]
            rgb_arr = torch.as_tensor(rgb, device=device, dtype=torch.float32) / 255.0
            colors = rgb_arr.view(-1, 3)   # [N, 3]，和 pts_cam 展平顺序一致
            all_colors.append(colors.cpu())

        # 拼接所有视角
        all_points = torch.cat(all_points, dim=0).numpy()  # [N_total, 3]
        all_colors = torch.cat(all_colors, dim=0).numpy()  # [N_total, 3]

        # 保存为 npy
        np.save(save_path + ".npy", all_points)
        np.save(save_path + "_rgb.npy", all_colors)
        print(f"[DA3] merged point cloud saved: {save_path}.npy")
        print(f"[DA3] merged point colors saved: {save_path}_rgb.npy")

        # 保存为带颜色的 ply
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(save_path + ".ply", pcd)
        print(f"[DA3] merged colored point cloud saved: {save_path}.ply")

        # 可视化
        if visualize:
            print("[DA3] Visualizing point cloud in Open3D window ...")
            o3d.visualization.draw_geometries([pcd])

        return all_points, all_colors

# ======================================================================
#                                 DEMO
# ======================================================================

if __name__ == "__main__":
    da3 = DepthAnything3Wrapper(device="cuda")

    # ---------------------------- 单图推理 ----------------------------
    # img_path = "data/0.jpg"
    # rgb, depth, conf, K, E = da3.predict(img_path)

    # print("rgb:", rgb.shape, rgb.dtype)
    # print("depth:", depth.shape, depth.dtype)
    # print("conf:", conf.shape, conf.dtype)
    # print("K:\n", K)
    # print("E:\n", E)

    # da3.save_all(depth, rgb, out_dir="single_output")
    # print("Single image depth prediction done ✓")

    # ---------------------------- 多图分别推理 ----------------------------
    # image_list = sorted([
    #     "data/rgb0.png",
    #     "data/rgb1.png",
    #     # 可以继续加: "data/rgb2.png",
    # ])

    # rgbs, depths, confs, Ks, Es = da3.predict_seperate_images(image_list)

    # # 保存每张图的结果
    # for i, (rgb_i, depth_i) in enumerate(zip(rgbs, depths)):
    #     da3.save_all(depth_i, rgb_i, out_dir=f"multi_output/view_{i}")

    # print("Multi-image depth prediction done ✓")

    # ------------------------ 多视角分别 3D 点云重建（带颜色 + 可视化） ------------------------
    # pts, colors = da3.reconstruct_from_seperate_images(
    #     image_list,
    #     save_path="reconstruction/room_demo",
    #     visualize=True,
    # )

    # print("3D Reconstruction Done ✓", pts.shape)

    # ------------------------ 多视角 Any-View 重建（直接用 DA3 内置接口） ------------------------
    image_list = sorted(glob.glob("data/multi_view/*.png"))

    pcd = da3.reconstruct_from_images(
        image_list,
        export_dir="reconstruction/soh_scene",
        export_format="glb",   # 想要 3DGS 就用 "gs_ply" 或 "gs_video"
        visualize=True,
    )