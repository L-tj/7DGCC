# TJL 
# 离线裁减点云，保存原始场景点云
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from os import path

ws_path = path.dirname(path.dirname(path.abspath(__file__)))     

class RealsenseCamera:
    def __init__(self):
        # 配置相机
        self.pipeline = rs.pipeline()    # 定义流程pipeline
        config = rs.config()        # 定义配置config
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
        profile = self.pipeline.start(config)    # 启动相机
        self.align = rs.align(rs.stream.color)   #深度图像向彩色对齐


    def get_aligned_images(self):
        while True:
            frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
            aligned_frames = self.align.process(frames)  # 获取对齐帧
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
        
            # depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
            color_image = np.asanyarray(aligned_color_frame.get_data())  # RGB图
            if color_image.any():
                break
        ############### 相机参数的获取 #######################
        # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

        # 返回相机深度参数、彩色图、深度图、齐帧中的depth帧
        return color_image, aligned_depth_frame


    def save_pointcloud(self, color_image, aligned_depth_frame):
        color_image1 = color_image.reshape(-1,3)
        pc = rs.pointcloud()                # 创建点云数据转换器
        pc.map_to(aligned_depth_frame)
        pointcloud = pc.calculate(aligned_depth_frame)

        vtx = np.asanyarray(pointcloud.get_vertices()).view(np.float32).reshape(-1, 3)
        color_rgb = color_image1[:, [2, 1, 0]]/255
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(vtx)
        point_cloud.colors = o3d.utility.Vector3dVector(color_rgb)
        o3d.visualization.draw_geometries([point_cloud])   

        if input("是否保存（y/n）") == 'y':
            name = input('文件id：')
            path = ws_path + f"/data/points_bag/scene{name}"

            if not os.path.exists(path):
                os.makedirs(path)
            depth_image = np.asanyarray(aligned_depth_frame.get_data())  
            cv2.imwrite(f"{path}/depth.png", depth_image)
            cv2.imwrite(f"{path}/scene.png", color_image)
            o3d.io.write_point_cloud(f"{path}/scene.ply", point_cloud)



if __name__ == '__main__':
    # 启动相机
    camera = RealsenseCamera()
    cv2.namedWindow('Scene')
    try:
        while True:
            rgb_image, aligned_depth_frame = camera.get_aligned_images()        # 获取对齐的图像与相机内参

            cv2.imshow('Scene', rgb_image)

            if cv2.waitKey(1) & 0xFF == 27:
                depth_image = np.asanyarray(aligned_depth_frame.get_data())     # 深度图（默认16位）
                camera.save_pointcloud(rgb_image, aligned_depth_frame)
                break

    finally:
        cv2.destroyAllWindows()
