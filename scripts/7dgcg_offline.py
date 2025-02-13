#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TJL 
"""离线生成在抓取配置并保存"""

from os import path
import numpy as np
import open3d as o3d
import copy  
# from capture_pc import cap_pc
from load_pc import load_select_area1
from gripper import GraspSampler
from scipy.spatial.transform import Rotation


# global config:
ws_path = path.dirname(path.dirname(path.abspath(__file__)))        #7DGCG



def front_show_pcd(*objects):  
    show_objects = [copy.deepcopy(i) for i in objects]
    rot = Rotation.from_euler('XYZ', [0, 180, 180], degrees=True).as_matrix()
    for obj in show_objects:  
        obj.rotate(rot, center=(0.0, 0.0, 0.0))  
    o3d.visualization.draw_geometries(show_objects)


# def show_highlight_grasps(ags, all_points, grasps_for_show, selected=0):
#     mlab.clf()
#     for i, grasp_ in enumerate(grasps_for_show):
#         # hand_points = ags.get_hand_points(grasp_[0], grasp_[1], grasp_[2], grasp_[3])
#         hand_points = ags.get_min_hand_points(grasp_)
#         if i == selected:
#             ags.show_grasp_3d(hand_points, color=(1, 0, 0))
#         else:
#             ags.show_grasp_3d(hand_points)
#     ags.show_points(all_points, scale_factor=0.006)
#     mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
#     mlab.show()



if __name__ == '__main__':
    data_path = ws_path + '/data/points_bag/object2'        
    nameids = [1, 2]      
    max_generate_nums = 8

    ags = GraspSampler(ws_path, 'robotiq_85')
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    for id in nameids:
        all_pcd,fd_pcd, r_pcd, r_vertex, cube_center = load_select_area1(data_path, nameid=id)
        grasp_sampled = ags.sample_grasps2(r_pcd, fd_pcd, max_generate_nums, cube_center, r_vertex)
        if len(grasp_sampled)==0:
            print("No results, trying again!")
            continue
        # print("Grasp sampler finish, generated {} grasps.".format(len(grasp_sampled)))
        grasp_sampled = np.array(grasp_sampled, dtype=object)
        grasp_sampled[:,-1] += 0.12*grasp_sampled[:,-2]/np.max(grasp_sampled[:,-2])
        sorted_grasp_sampled = sorted(grasp_sampled, key=lambda x: x[-1], reverse=True) 

        graspers = ags.get_show_hand1(sorted_grasp_sampled[:6])        # 显示前6个抓取配置
        # o3d.visualization.draw_geometries([fd_pcd,graspers])
        front_show_pcd(all_pcd,FOR1,graspers)

        # np.savez(data_path+'/7DGCG_result.npz', sorted_grasp_sampled=sorted_grasp_sampled)
        # if path.exists(data_path+'/result.npy'):
        #     existing_data = np.load(data_path+'/result.npy',allow_pickle=True)
        #     all_grasp_sampled = np.concatenate((existing_data, all_grasp_sampled))
        # np.save(data_path+'/result.npy', all_grasp_sampled)     # 保存结果
        
