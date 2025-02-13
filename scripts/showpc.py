# TJL 
# RGBD转点云并可视化
import cv2
import numpy as np
import open3d as o3d

cam_K = np.array([[605.7191772460938, 0, 426.1901550292969],[0, 605.8008422851562, 260.697998046875],[0, 0, 1]])

def depth2pc(depth, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where((depth > 0 )& (depth < 4))
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - cam_K[0,2])#平移
    normalized_y = (y.astype(np.float32) - cam_K[1,2])

    world_x = normalized_x * depth[y, x] / cam_K[0,0]
    world_y = normalized_y * depth[y, x] / cam_K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([point_cloud])    # open3d和mayavi有版本冲突，不能一同用于显示

    # return (pc, rgb)

path = '/media/lsr/KINGSTON/scene4'
color_image = cv2.imread(f'{path}/scene.png')
color_image = color_image[:,:,[2,1,0]]/255

# depth_image1 = cv2.imread(f'{path}/completion_depth1.png', cv2.IMREAD_UNCHANGED) # 读取深度图
# depth_image1 = depth_image1 * 0.001  
# depth2pc(depth_image1, color_image)

depth_image1 = cv2.imread(f'{path}/completion_depth.png', cv2.IMREAD_UNCHANGED) # 读取深度图
depth_image1 = depth_image1 * 0.001  
depth2pc(depth_image1, color_image)

depth_image = cv2.imread(f'{path}/depth.png', cv2.IMREAD_UNCHANGED) # 读取深度图
depth_image = depth_image * 0.001  
depth2pc(depth_image, color_image)


