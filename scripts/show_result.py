from gripper import get_simplified_hand, get_one_show_hand
import numpy as np
import open3d as o3d
from random import uniform

# PointNetGPD = 'PointNetGPD_result'
# Ours = '7DGCG_result'

def get_show_hand1(grasp_sampled):
    # open3d 简化圆柱显示， 显示颜色渐变
    color = np.array([uniform(0,0.2), uniform(0,0.2), 1])
    d_color = (0.9 - color[:2].max())/len(grasp_sampled)
    graspers = o3d.geometry.TriangleMesh()

    hand = get_simplified_hand(grasp_sampled[0][2], 0.003)
    hand.paint_uniform_color([1,0.1,0.1])
    hand.translate(grasp_sampled[0][0] - grasp_sampled[0][1][0] * 0.01)     #self.grasp_cfg['center_height'])
    a = np.vstack([grasp_sampled[0][1][1],grasp_sampled[0][1][2],grasp_sampled[0][1][0]])
    hand.rotate(a.T)
    graspers += hand
    for grasp_ in grasp_sampled[1:]:
        hand = get_simplified_hand(grasp_[2], 0.002)
        hand.paint_uniform_color(color)
        color = color + np.array([d_color, d_color, 0])

        hand.translate(grasp_[0] - grasp_[1][0] * 0.01 )         #self.grasp_cfg['center_height'])
        a = np.vstack([grasp_[1][1],grasp_[1][2],grasp_[1][0]])
        hand.rotate(a.T)
        graspers += hand
    return graspers


def get_show_hand(grasp_sampled):
    color = np.array([uniform(0,0.2), uniform(0,0.2), 1])
    d_color = (0.9 - color[:2].max())/len(grasp_sampled)
    graspers = o3d.geometry.TriangleMesh()

    hand = get_simplified_hand(0.085, 0.003)
    hand.paint_uniform_color([1,0.1,0.2])
    hand.translate(grasp_sampled[0][0])    
    a = np.vstack([grasp_sampled[0][2],grasp_sampled[0][3],grasp_sampled[0][1]])
    hand.rotate(a.T)
    graspers += hand
    for grasp_ in grasp_sampled[1:]:
        hand = get_simplified_hand(0.085, 0.002)
        hand.paint_uniform_color(color)
        color = color + np.array([d_color, d_color, 0])

        hand.translate(grasp_[0]) 
        a = np.vstack([grasp_[2],grasp_[3],grasp_[1]])
        hand.rotate(a.T)
        graspers += hand
    return graspers

# data_path = '/home/lsm/my_code/6D-grasp/data/points_bag/scene2'
data_path = '/home/lsm/my_code/6D-grasp/data/points_bag/scene6'
data = np.load(data_path + '/7DGCG_result.npz', allow_pickle=True)
# data = np.load(data_path + '/PointNetGPD_result.npz', allow_pickle=True)
keys = data.files
if 'sorted_grasp_sampled' in keys:    
    sorted_grasp_sampled = data['sorted_grasp_sampled']
if 'execution_time' in keys:    
    execution_time = data['execution_time'].item()
a = sorted_grasp_sampled[:,-2]
b = np.max(sorted_grasp_sampled[:,-2])
sorted_grasp_sampled[:,-1] += 0.1*sorted_grasp_sampled[:,-2]/a

    
FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
all_pcd = o3d.io.read_point_cloud(data_path+'/scene.ply')      # 读取场景点云
graspers = get_show_hand(sorted_grasp_sampled)        #显示所有抓取配置
o3d.visualization.draw_geometries([all_pcd,FOR1,graspers])

# for i in range(len(grasp_sampled)):
#     graspers = get_show_hand(grasp_sampled[[i]])        #显示所有抓取配置
#     o3d.visualization.draw_geometries([all_pcd,FOR1,graspers])
#     command = input("实机执行（y/n/q）:")
#     if command == 'n':
#         continue
#     elif command == 'q':
#         break
#     for j in range(15):
#         ros_grasp_msg1(grasp_sampled[[i]], 1)         # 发布话题
#         rate.sleep()

print('finish')
