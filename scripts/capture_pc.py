# TJL 
# 在线裁减点云，保存裁减后的目标点云，非实时相机运行，有点云发布
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from os import path
import rospy
from sensor_msgs.msg import PointCloud2, PointField


ws_path = path.dirname(path.dirname(path.abspath(__file__)))        #pointnet-gpd-ros
#创建一个全局变量，并设置鼠标事件的回调函数
pt1 = (0, 0)
pt2 = (0, 0)
top_left_clicked = False
bottom_right_clicked = False


class RealsenseCamera:
    def __init__(self):
        self.point_cloud_pub = rospy.Publisher('/detection/camera/points', PointCloud2, queue_size=10)  # 创建一个点云发布者
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
        
            depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
            color_image = np.asanyarray(aligned_color_frame.get_data())  # RGB图
            if depth_image.any() and color_image.any():
                break
        self.publish_pointcloud(color_image, aligned_depth_frame)
        return color_image, aligned_depth_frame


    def publish_pointcloud(self, color_image, aligned_depth_frame):
        color_image_32 = np.uint32(color_image).reshape(-1,3)
        pc = rs.pointcloud()                # 创建点云数据转换器
        pc.map_to(aligned_depth_frame)
        pointcloud = pc.calculate(aligned_depth_frame)

        vtx = np.asanyarray(pointcloud.get_vertices()).view(np.float32).reshape(-1, 3)
        points_rgb = np.zeros( (vtx.shape[0], 1), \
            dtype={ "names": ( "x", "y", "z", "rgb" ), 
                    "formats": ( "f4", "f4", "f4", "u4" )} )
        points_rgb["x"] = vtx[:, 0].reshape((-1, 1))
        points_rgb["y"] = vtx[:, 1].reshape((-1, 1))
        points_rgb["z"] = vtx[:, 2].reshape((-1, 1))
        points_rgb["rgb"] = ((color_image_32[:, 2] << 16) | (color_image_32[:, 1] << 8) | (color_image_32[:, 0])).reshape((-1, 1))
        # Filter out invalid points
        mask = np.logical_and(points_rgb["z"] > 0.05, points_rgb["z"] < 2)
        points_rgb = points_rgb[mask]
        self.pointcloud2_encode(points_rgb)


    def pointcloud2_encode(self, points):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "camera"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        points = points.reshape(-1)
        msg.data = points.tobytes()

        self.point_cloud_pub.publish(msg)
    

    def small_generate_pointcloud(self, color_image, aligned_depth_frame, camera_xyz, left_top_xyz):
        cube_center = np.array(camera_xyz)
        cube_vertex = np.array(left_top_xyz)

        color_rgb = color_image.reshape(-1,3)
        color_rgb = color_rgb[:, [2, 1, 0]]/255
        pc = rs.pointcloud()                # 创建点云数据转换器
        pc.map_to(aligned_depth_frame)
        pointcloud = pc.calculate(aligned_depth_frame)
        vtx = np.asanyarray(pointcloud.get_vertices()).view(np.float32).reshape(-1, 3)
        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(vtx)
        all_pcd.colors = o3d.utility.Vector3dVector(color_rgb)

        # 放大ROI区域
        f_factor = 1.8                   # 缩放比例f_factor = (x1-c)/(x-c)
        f_vertex = cube_center - f_factor * (cube_center - cube_vertex)
        f_vertex = np.minimum(f_vertex, cube_center - np.array([0.1,0.1,0.08]))
        mask = np.all(vtx > f_vertex, axis=1) & np.all(vtx < (2*cube_center - f_vertex),axis=1)
        f_pcd = all_pcd.select_by_index(np.where(mask)[0])
        f_points = np.asarray(f_pcd.points)

        r_factor = 0.9                  # 缩小比例
        r_vertex = cube_center - r_factor * (cube_center - cube_vertex) - np.array([0,0,0.04])
        mask = np.all(f_points > r_vertex, axis=1) & np.all(f_points < (2*cube_center - r_vertex),axis=1)
        r_pcd = f_pcd.select_by_index(np.where(mask)[0])
            
        return all_pcd,f_pcd, r_pcd, r_vertex, cube_center



#创建一个鼠标事件的回调函数，用于处理鼠标事件：
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, top_left_clicked, bottom_right_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        if top_left_clicked and bottom_right_clicked:
            pt1 = (0, 0)
            pt2 = (0, 0)
            top_left_clicked = False
            bottom_right_clicked = False

        if not top_left_clicked:
            pt1 = (x, y)
            top_left_clicked = True

        elif not bottom_right_clicked:
            pt2 = (x, y)
            bottom_right_clicked = True


def cap_pc():
    # 启动相机
    camera = RealsenseCamera()
    
    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', draw_rectangle)
    while True:
        rgb_image, aligned_depth_frame = camera.get_aligned_images()  # 获取对齐的图像与相机内参
        color_image = rgb_image.copy()

        if top_left_clicked:
            cv2.circle(color_image, center=pt1, radius=5, color=(0, 255, 0), thickness=-1)

        if top_left_clicked and bottom_right_clicked:
            cv2.rectangle(color_image, pt1, pt2, (0, 255, 0), 2)

        cv2.imshow('Select ROI', color_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()     # 关闭OpenCV窗口
    # print(f'ROI coordinates: ({pt1[0]}, {pt1[1]}), ({pt2[0]}, {pt2[1]})')
    ux = int((pt1[0]+pt2[0])/2)  # 计算像素坐标系下物体中心点的x
    uy = int((pt1[1]+pt2[1])/2)  # 计算像素坐标系下物体中心点的y

    dis = aligned_depth_frame.get_distance(ux, uy)
    if dis==0.0:
        print('中心点深度为零')
        return 
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz 中心点
    left_top_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, pt1, dis)  # 左上角xy
    # f_points, r_points = camera.generate_pointcloud(rgb_image, aligned_depth_frame, camera_xyz, left_top_xyz)
    all_pcd,f_points, r_pcd, r_vertex, cube_center = camera.small_generate_pointcloud(rgb_image, aligned_depth_frame, camera_xyz, left_top_xyz)
    return all_pcd,f_points, r_pcd, r_vertex, cube_center


if __name__ == '__main__':
    cap_pc()

    