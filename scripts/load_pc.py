# TJL 
# 自选区域，离线裁加载点云
import cv2
import numpy as np
import open3d as o3d
# from os import path

# ws_path = path.dirname(path.dirname(path.abspath(__file__)))        #pointnet-gpd-ros

#创建一个全局变量，并设置鼠标事件的回调函数
pt1 = (0, 0)
pt2 = (0, 0)
top_left_clicked = False
bottom_right_clicked = False


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


def pixel_to_point(pixel, depth):
    xx = ((pixel[0] - 426.1901550292969) / 605.7191772460938) * depth           # 相机内参
    yy = ((pixel[1] - 260.697998046875) / 605.8008422851562) * depth
    return np.array([xx, yy, depth])


def zoom_box(factor, x, c):
    x_left = factor * x[0] + (1-factor) * c[0];
    x_right = -factor * x[0] + (1+factor) * c[0];
    y_left = factor * x[1] + (1-factor) * c[1];
    y_right = -factor * x[1] + (1+factor) * c[1];
    return (x_left,x_right), (y_left,y_right)


def load_select_area(path = '/home/lsr/lsr_code/ltj/6D-grasp/data/points_bag/scene1'): 
    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', draw_rectangle)

    # 读取图片，手选ROI区域
    color_image = cv2.imread(f'{path}/scene.png')
    assert color_image is not None, "Image loading failed"
    while True:
        if top_left_clicked:
            cv2.circle(color_image, center=pt1, radius=5, color=(0, 255, 0), thickness=-1)

        if top_left_clicked and bottom_right_clicked:
            cv2.rectangle(color_image, pt1, pt2, (0, 255, 0), 2)

        cv2.imshow('Select ROI', color_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    # print(f'ROI coordinates: ({pt1[0]}, {pt1[1]}), ({pt2[0]}, {pt2[1]})')
    ux = int((pt1[0]+pt2[0])/2)  # 计算像素坐标系下物体中心点的x
    uy = int((pt1[1]+pt2[1])/2)  # 计算像素坐标系下物体中心点的y
   
    depth_image = cv2.imread(f'{path}/depth.png', cv2.IMREAD_UNCHANGED) # 读取深度图
    depth_img = depth_image[pt1[1]:pt2[1]:5, pt1[0]:pt2[0]:5].reshape(-1, 1)
    dis = np.mean(depth_img[depth_img[:,0]>0]) * 0.001          # 求得中心点深度
    if depth_image[uy, ux] > 0:
        dis = dis * 0.4 + depth_image[uy, ux] * 0.6 * 0.001

    camera_xyz = pixel_to_point((ux, uy), dis)       # 计算相机坐标系的xyz 中心点
    left_top_xyz = pixel_to_point(pt1, dis)          # 左上角xy

    # 放大ROI区域
    z_max = camera_xyz[2] + 0.1
    z_min = camera_xyz[2] - 0.08
    f_factor = 2                    # 缩放比例f_factor = (x1-c)/(x-c)
    left_top = (min(left_top_xyz[0], camera_xyz[0]-0.08), min(left_top_xyz[1], camera_xyz[1]-0.08), left_top_xyz[2])
    (f_x_left, f_x_right),(f_y_up, f_y_down) = zoom_box(f_factor, left_top, camera_xyz)

    np.set_printoptions(suppress=True) # 取消默认科学计数法，open3d无法读取科学计数法表示
    all_pcd = o3d.io.read_point_cloud(f'{path}/scene.ply')      # 读取场景点云
    # o3d.visualization.draw_geometries([all_pcd])    # 可视化加载的点云
    all_points = np.asarray(all_pcd.points)
    mask1 = (all_points[:, 0] > f_x_left) & (all_points[:, 0] < f_x_right)
    mask2 = (all_points[:, 1] > f_y_up) & (all_points[:, 1] < f_y_down)
    mask3 = (all_points[:, 2] > z_min) & (all_points[:, 2] < z_max)
    mask = mask1 & mask2 & mask3
    f_pcd = all_pcd.select_by_index(np.where(mask)[0])
    # o3d.visualization.draw_geometries([f_pcd])    

    f_points = np.asarray(f_pcd.points)
    mr = f_points.shape[0]**0.5/((f_points.max()-f_points.min())*200000)
    downpcd = f_pcd.voxel_down_sample(mr)      # 下采样
    fd_points = np.asarray(downpcd.points)
    # o3d.visualization.draw_geometries([downpcd])    # 可视化下采样后的点云

    # 缩小ROI区域
    z_max = camera_xyz[2] + 0.04
    z_min = camera_xyz[2] - 0.05
    r_factor = 0.6                  # 缩小比例
    (r_x_left, r_x_right),(r_y_up, r_y_down) = zoom_box(r_factor, left_top_xyz, camera_xyz)
    mask1 = (f_points[:, 0] > r_x_left) & (f_points[:, 0] < r_x_right)
    mask2 = (f_points[:, 1] > r_y_up) & (f_points[:, 1] < r_y_down)
    mask3 = (f_points[:, 2] > z_min) & (f_points[:, 2] < z_max)
    mask = mask1 & mask2 & mask3
    r_pcd = f_pcd.select_by_index(np.where(mask)[0])
    # o3d.visualization.draw_geometries([pcd])    # open3d和mayavi有版本冲突，不能一同用于显示

    return all_pcd, fd_points, r_pcd


def load_select_area1(path = '/home/lsr/lsr_code/ltj/6D-grasp/data/points_bag/object6', nameid=0): 
    # 加载图片，手选ROI区域
    color_image = cv2.imread(f'{path}/scene{nameid}.png')
    assert color_image is not None, "Image loading failed"

    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', draw_rectangle)
    while True:
        if top_left_clicked:
            cv2.circle(color_image, center=pt1, radius=5, color=(0, 255, 0), thickness=-1)

        if top_left_clicked and bottom_right_clicked:
            cv2.rectangle(color_image, pt1, pt2, (0, 255, 0), 2)

        cv2.imshow('Select ROI', color_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    # print(f'ROI coordinates: ({pt1[0]}, {pt1[1]}), ({pt2[0]}, {pt2[1]})')
    ux = int((pt1[0]+pt2[0])/2)     # 计算像素坐标系下物体中心点的x
    uy = int((pt1[1]+pt2[1])/2)     # 计算像素坐标系下物体中心点的y
   
    depth_image = cv2.imread(f'{path}/depth{nameid}.png', cv2.IMREAD_UNCHANGED)     # 读取深度图
    depth_img = depth_image[pt1[1]:pt2[1]:5, pt1[0]:pt2[0]:5].reshape(-1, 1)
    dis = np.mean(depth_img[depth_img[:,0]>0]) * 0.001                              # 求得中心点深度
    if depth_image[uy, ux] > 0:
        dis = dis * 0.4 + depth_image[uy, ux] * 0.6 * 0.001
        # dis = depth_image[uy, ux] * 0.001

    cube_center = pixel_to_point((ux, uy), dis)         # 计算相机坐标系的xyz 中心点
    cube_vertex = pixel_to_point(pt1, dis)              # 左上角xy

    np.set_printoptions(suppress=True)                  # 取消默认科学计数法，open3d无法读取科学计数法表示
    all_pcd = o3d.io.read_point_cloud(f'{path}/scene{nameid}.ply')      # 读取场景点云
    # o3d.visualization.draw_geometries([all_pcd])      # 可视化加载的点云
    all_points = np.asarray(all_pcd.points)
    # 放大ROI区域
    f_factor = 2                    # 缩放比例f_factor = (x1-c)/(x-c)
    f_vertex = cube_center - f_factor * (cube_center - cube_vertex)
    f_vertex = np.minimum(f_vertex, cube_center - np.array([0.1,0.1,0.08]))
    mask = np.all(all_points > f_vertex, axis=1) & np.all(all_points < (2*cube_center - f_vertex),axis=1)
    f_pcd = all_pcd.select_by_index(np.where(mask)[0])
    # all_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    # cube = show_cube(cube_center, f_vertex)
    # o3d.visualization.draw_geometries([all_pcd,f_pcd]+cube)  

    f_points = np.asarray(f_pcd.points)
    mr = f_points.shape[0]**0.5/((f_points.max()-f_points.min())*200000)
    downpcd = f_pcd.voxel_down_sample(mr)               # 下采样
    # fd_points = np.asarray(downpcd.points)
    # o3d.visualization.draw_geometries([downpcd])      # 可视化下采样后的点云

    # 缩小ROI区域
    r_factor = 0.8                  # 缩小比例
    r_vertex = cube_center - r_factor * (cube_center - cube_vertex) - np.array([0,0,0.04])
    mask = np.all(f_points > r_vertex, axis=1) & np.all(f_points < (2*cube_center - r_vertex),axis=1)
    r_pcd = f_pcd.select_by_index(np.where(mask)[0])
    # r_pcd.paint_uniform_color([0.1, 0.2, 0.7])
    # cube = show_cube(cube_center, r_vertex)
    # o3d.visualization.draw_geometries([all_pcd,r_pcd]+cube) 

    return all_pcd, downpcd, r_pcd, r_vertex, cube_center
    # return f_pcd, downpcd, r_pcd, r_vertex, cube_center


def show_cube(c, top):
    # top = np.array([-1,-1,-1])
    # c = np.array([0,0,0])
 
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=20)
    center_sphere.compute_vertex_normals()
    center_sphere.paint_uniform_color([0.7, 0.1, 0.1])
    # center_sphere.translate(c)
    center_sphere.translate(c-np.array([0,0,0.005]))

    top_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004, resolution=20)
    top_sphere.compute_vertex_normals()
    top_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    top_sphere.translate(top)

    # o3d.visualization.draw_geometries([center_sphere,top_sphere])

    d = 2*(c-top)
    # 生成一些线段的端点坐标
    points = np.array([top, top + np.array([d[0],0,0]), top + np.array([0, d[1],0]), top + np.array([0,0,d[2]]), 
                    top + np.array([d[0],d[1],0]), top + np.array([d[0],0,d[2]]), top + np.array([0,d[1],d[2]]), top + d])  # 第三条线段的端点坐标
    lines = np.array([[0, 1],[0, 2],[0, 3],
                    [5, 1],[5, 3],[5, 7],
                    [4, 1],[4, 2],[4, 7],
                    [6, 2],[6, 3],[6, 7]])
    # 创建LineSet对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)    # 设置线段的起始点
    line_set.lines = o3d.utility.Vector2iVector(lines)      # 设置线段的连接关系
    line_set.paint_uniform_color([0.5, 0.1, 0.1])
    # o3d.visualization.draw_geometries([line_set])
    return [line_set, center_sphere, top_sphere]


if __name__ == '__main__':
    load_select_area1()
