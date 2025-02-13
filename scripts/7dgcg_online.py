#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TJL 
"""在线生成抓取配置并发布ROS"""

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
from rm_msgs.msg import GraspArray, Grasp
from visualization_msgs.msg import MarkerArray, Marker
# import argparse
import time
from os import path
import numpy as np
from scipy.spatial.transform import Rotation
from pc_type_conversion import pointcloud2_to_array, split_rgb
from gripper import GraspSampler, get_show_hand
import open3d as o3d
import copy  
# from capture_pc import cap_pc


# global config:
ws_path = path.dirname(path.dirname(path.abspath(__file__)))        #7DGCG



def show_pcd(pcd): 
    show_pcd = copy.deepcopy(pcd)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.add_geometry(pcd)
    R1 = show_pcd.get_rotation_matrix_from_xyz((0,np.pi,np.pi))     # x,y屏幕上方,z垂直屏幕向外，即左下角最小
    show_pcd.rotate(R1,center = (0.0,0.0,0.0))                      # 指定旋转中心
    vis.add_geometry(show_pcd)
    # 运行可视化窗口
    vis.run()
    vis.destroy_window()


def ros_grasp_msg(real_good_grasp_, n=1):
    "发布前n个抓取位姿"
    grasp_pose_array = GraspArray()
    grasp_pose_array.header.stamp = rospy.Time.now()
    grasp_pose_array.header.frame_id = "camera"

    for g in real_good_grasp_[:n]:
        grasp_matrix = np.vstack([g[1][1], g[1][2], g[1][0]]).T                # 换加爪坐标系
        g_quat=Rotation.from_matrix(grasp_matrix).as_quat()                    # 将旋转矩阵换为四元数

        grasp = Grasp()
        grasp.pose.position.x = g[0][0]
        grasp.pose.position.y = g[0][1]
        grasp.pose.position.z = g[0][2]
        grasp.pose.orientation.x = g_quat[0]
        grasp.pose.orientation.y = g_quat[1]
        grasp.pose.orientation.z = g_quat[2]
        grasp.pose.orientation.w = g_quat[3]
        grasp.open = g[2]

        grasp_pose_array.grasps.append(grasp)
    # print("grasp_pose_array:", grasp_pose_array)
    grasp_pub.publish(grasp_pose_array)


def show_marker(marker_array_, pos_, ori_, frame_id, scale_, color_, lifetime_):
    marker_ = Marker()
    marker_.header.frame_id = frame_id
    marker_.header.stamp = rospy.Time.now()
    marker_.type = marker_.CUBE
    marker_.action = marker_.ADD

    marker_.pose.position.x = pos_[0]
    marker_.pose.position.y = pos_[1]
    marker_.pose.position.z = pos_[2]
    marker_.pose.orientation.x = ori_[0]
    marker_.pose.orientation.y = ori_[1]
    marker_.pose.orientation.z = ori_[2]
    marker_.pose.orientation.w = ori_[3]

    marker_.lifetime = rospy.Duration.from_sec(lifetime_)
    marker_.scale.x = scale_[0]
    marker_.scale.y = scale_[1]
    marker_.scale.z = scale_[2]
    marker_.color.a = 0.5
    marker_.color.r = color_[0]
    marker_.color.g = color_[1]
    marker_.color.b = color_[2]
    marker_array_.markers.append(marker_)

    
def show_all_grasp_marker(real_grasp_, lifetime_, selected=0):
    "显示所有并高亮选中抓取"
    hh = 0.01
    fw = 0.01
    hod = 0.11
    hd = 0.04

    marker_array = MarkerArray()
    frame_id = "camera"

    for i, grasp_ in enumerate(real_grasp_):
        rotation = np.vstack([grasp_[1], grasp_[2], grasp_[3]]).T
        qua = Rotation.from_matrix(rotation).as_quat()
        marker_bottom_pos = grasp_[0] - hh * 0.5 * grasp_[1]
        marker_left_pos = grasp_[0] - grasp_[2] * (hod - fw) * 0.5 + hd * 0.5 * grasp_[1]
        marker_right_pos = grasp_[0] + grasp_[2] * (hod - fw) * 0.5 + hd * 0.5 * grasp_[1]

        if i == selected:
            color_ = [1, 0, 0]
        else:
            color_ = [0, 1, 0]
        show_marker(marker_array, marker_bottom_pos, qua, frame_id, [hh, hod, hh], color_, lifetime_)
        show_marker(marker_array, marker_left_pos, qua, frame_id, [hd, fw, hh], color_, lifetime_)
        show_marker(marker_array, marker_right_pos, qua, frame_id, [hd, fw, hh], color_, lifetime_)

    id_ = 0
    for m in marker_array.markers:
        m.id = id_
        id_ += 1
    vis_grasp_pub.publish(marker_array)


def show_single_grasp_marker(real_grasp_, gripper_, color_, lifetime_):
    "显示单个抓取"
    hh = gripper_.hand_height
    fw = gripper_.real_finger_width
    hod = gripper_.hand_outer_diameter
    hd = gripper_.real_hand_depth

    approach = real_grasp_[1]
    binormal = real_grasp_[2]
    minor_pc = real_grasp_[3]
    grasp_bottom_center = real_grasp_[0]

    marker_array = MarkerArray()
    frame_id = "camera"
    rotation = np.vstack([approach, binormal, minor_pc]).T
    qua = Rotation.from_matrix(rotation).as_quat()

    marker_bottom_pos = grasp_bottom_center - hh * 0.5 * approach
    marker_left_pos = grasp_bottom_center - binormal * (hod - fw) * 0.5 + hd * 0.5 * approach
    marker_right_pos = grasp_bottom_center + binormal * (hod - fw) * 0.5 + hd * 0.5 * approach
    show_marker(marker_array, marker_bottom_pos, qua, frame_id, [hh, hod, hh], color_, lifetime_)
    show_marker(marker_array, marker_left_pos, qua, frame_id, [hd, fw, hh], color_, lifetime_)
    show_marker(marker_array, marker_right_pos, qua, frame_id, [hd, fw, hh], color_, lifetime_)

    id_ = 0
    for m in marker_array.markers:
        m.id = id_
        id_ += 1
    vis_grasp_pub.publish(marker_array)


def show_grasp_marker(gripper_, color_, lifetime_):
    "显示加爪模型"
    hh = gripper_.hand_height       # 0.03
    fw = gripper_.real_finger_width
    hod = gripper_.hand_outer_diameter
    hd = gripper_.real_hand_depth
    marker_array = MarkerArray()
    frame_id = "grasp"
    qua = [0,0,0,1]
    
    marker_bottom_pos = [- hh * 0.5, 0, 0]
    marker_left_pos = [hd*0.5, -(hod - fw)*0.5, 0]
    marker_right_pos = [hd*0.5, (hod - fw)*0.5, 0]
    show_marker(marker_array, marker_bottom_pos, qua, frame_id, [hh, hod, hh], color_, lifetime_)
    show_marker(marker_array, marker_left_pos, qua, frame_id, [hd, fw, hh], color_, lifetime_)
    show_marker(marker_array, marker_right_pos, qua, frame_id, [hd, fw, hh], color_, lifetime_)

    id_ = 0
    for m in marker_array.markers:
        m.id = id_
        id_ += 1
    vis_grasp_pub.publish(marker_array)


def show_grasp_tf(real_good_grasp_):
    "发布显示单个相机到夹爪的tf"
    broadcaster = tf2_ros.TransformBroadcaster()
    grasp_matrix = np.vstack([real_good_grasp_[1], real_good_grasp_[2], real_good_grasp_[3]]).T
    grasp_quat=Rotation.from_matrix(grasp_matrix).as_quat()            #将旋转矩阵换为欧拉角

    tfs = TransformStamped()
    tfs.header.frame_id = "camera"
    tfs.header.stamp = rospy.Time.now()
    tfs.child_frame_id = "grasp"
    tfs.transform.translation.x = real_good_grasp_[0][0]
    tfs.transform.translation.y = real_good_grasp_[0][1]
    tfs.transform.translation.z = real_good_grasp_[0][2]
    tfs.transform.rotation.x = grasp_quat[0]
    tfs.transform.rotation.y = grasp_quat[1]
    tfs.transform.rotation.z = grasp_quat[2]
    tfs.transform.rotation.w = grasp_quat[3]
    broadcaster.sendTransform(tfs)    


def subscription_pc():
    print("rospy is waiting for message: /detection/object_in_camera")
    yolo_detect = rospy.wait_for_message("/detection/object_in_camera", Pose)   # 接收目标检测后目标物体定位信息
    print("rospy is waiting for pointcloud message")
    rs_data = rospy.wait_for_message("/detection/camera/points", PointCloud2)   # 接收场景点云
    print("got yolo detect message")
    cube_center = np.array([yolo_detect.position.x, yolo_detect.position.y, yolo_detect.position.z])
    cube_vertex = np.array([yolo_detect.orientation.x, yolo_detect.orientation.y, yolo_detect.position.z])
    points_arry = pointcloud2_to_array(rs_data)
    vtx, colors = split_rgb(points_arry)

    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(vtx)
    all_pcd.colors = o3d.utility.Vector3dVector(colors)

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
    
    return all_pcd, f_pcd, r_pcd, r_vertex, cube_center



if __name__ == '__main__':
    max_generate_nums = 12

    rospy.init_node('grasp_generation', anonymous=True)
    grasp_pub = rospy.Publisher('/detect_grasps/clustered_grasps', GraspArray, queue_size=1)    #发布检测到的抓取，用于执行抓取
    # vis_grasp_pub = rospy.Publisher('gripper_vis', MarkerArray, queue_size=1)                 #创建发布器，用于rviz显示抓取
    rate = rospy.Rate(5)

    ags = GraspSampler(ws_path, 'robotiq_85')
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    while not rospy.is_shutdown():
        rate.sleep()
        start_gcg = rospy.get_param('start_gcg', False)
        if not start_gcg:
            continue

        all_pcd,fd_pcd, r_pcd, r_vertex, cube_center = subscription_pc()
        # all_pcd,fd_pcd, r_pcd, r_vertex, cube_center = cap_pc()
        start_time = time.time()
        grasp_sampled = ags.sample_grasps2(r_pcd, fd_pcd,max_generate_nums, cube_center, r_vertex)
        # ags.cube_center, ags.r_vertex = cube_center, r_vertex
        # grasp_sampled = ags.sample_grasps3(r_pcd, fd_pcd,4)

        if len(grasp_sampled)==0:
            print("No results, trying again!")
            continue
        grasp_sampled = np.array(grasp_sampled, dtype=object)
        grasp_sampled[:,-1] += 0.12*grasp_sampled[:,-2]/np.max(grasp_sampled[:,-2])
        sorted_grasp_sampled = sorted(grasp_sampled, key=lambda x: x[-1], reverse=True) 

        execution_time = time.time() - start_time 
        print(f"代码运行时间: {execution_time}秒")
        # print("Capture configuration generation completed, generated {} grasps.".format(len(grasp_sampled)))
        
        grasp_sampled1 = np.array(sorted_grasp_sampled, dtype=object)
        graspers = ags.get_show_hand1(grasp_sampled1[:3])           # 显示前3个抓取配置
        o3d.visualization.draw_geometries([fd_pcd,FOR1,graspers])
        for j in range(5):
            ros_grasp_msg(grasp_sampled1, 3)                       # 发布前3个抓取配置
            rate.sleep()
        print('ok')

        # ags.show_highlight_grasps1(all_pcd, grasp_sampled)
        # show_all_grasp_marker(sorted_grasp_sampled, marker_life_time)

        # ros_grasp_msg(sorted_grasp_sampled, 1)
        # show_grasp_tf(sorted_grasp_sampled[0])
        # show_grasp_marker(gripper, [0, 1, 0], marker_life_time)

        # rospy.loginfo(" Publishing grasp pose to rviz using marker array and good grasp pose")

        
