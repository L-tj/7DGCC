# -*- coding: utf-8 -*-


from abc import ABCMeta
# import copy
import numpy as np
from scipy.spatial.transform import Rotation
try:
    from mayavi import mlab
except ImportError:
    mlab = []

from os import path
import open3d as o3d
import itertools
import yaml
from random import uniform
import multiprocessing as mp



class GraspSampler:
    __metaclass__ = ABCMeta

    def __init__(self, ws_path, gripper_name):
        with open(path.join(ws_path, 'cfg', gripper_name, 'params.yaml'), 'r', encoding='utf-8') as f:
            self.gripper_cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        with open(path.join(ws_path, 'cfg', '7Dgrasp.yaml'), 'r', encoding='utf-8') as f:
            self.grasp_cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.d_height = -self.gripper_cfg['hand_height'] + self.grasp_cfg['center_height'] + self.grasp_cfg['min_contact_height']         # -0.035
        self.d_width = self.gripper_cfg['max_width'] - 2 * self.grasp_cfg['finger_expand_width']        # threshold=0.08为减去预留后的点云最大宽度
        self.find_width = self.gripper_cfg['finger_width'] + self.grasp_cfg['finger_expand_width']         # threshold=0.0125为手指厚度
        self.finger_height = self.gripper_cfg['finger_height'] - self.grasp_cfg['center_height']         # 0.04为手指高
        self.cube_center = np.zeros(3)
        self.r_vertex = np.zeros(3)


    def show_points(self, point, color='lb', scale_factor=.002):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:       # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)


    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)


    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc, scale_factor=0.001):
        un2 = grasp_bottom_center
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
                      scale_factor=.03, line_width=0.25, color=(0, 1, 0), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
                      scale_factor=.03, line_width=0.1, color=(0, 0, 1), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
                      scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')


    def get_hand_points(self, open_w, grasp_bottom_center, approach_normal, binormal, minor_pc=None):
        grasp_bottom_center = grasp_bottom_center - approach_normal * self.grasp_cfg['center_height']
        hh = self.gripper_cfg['hand_height']       #手宽0.05
        fd = self.gripper_cfg['finger_depth']      #手指深度0.03
        fw = self.gripper_cfg['finger_width']      #手指宽度0.01
        fh = self.gripper_cfg['finger_height']     #手指长度0.04 + 0.01
        open_w = max(min(0.085, open_w), 0)               # 手抓内部最大张开度0.085
        if minor_pc is None:
            minor_pc = np.cross(approach_normal, binormal)
            minor_pc = minor_pc / np.linalg.norm(minor_pc)

        p5_p6 = minor_pc * fd * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * fd * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * fh + p5
        p2 = approach_normal * fh + p6
        p3 = approach_normal * fh + p7
        p4 = approach_normal * fh + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p
    

    def get_min_hand_points(self, open_w, grasp_bottom_center, approach_normal, binormal, minor_pc=None):
        grasp_bottom_center = grasp_bottom_center - approach_normal * 0.015
        hh = 0.005              # 手指宽0.03
        fw = 0.005              # 手指厚度0.01
        hd = 0.04               # 抓手深度0.04 + 0.01
        open_w = max(min(0.08, open_w), 0)               # 手抓内部最大张开度0.085 - 0.005

        if minor_pc is None:
            minor_pc = np.cross(approach_normal, binormal)
            minor_pc = minor_pc / np.linalg.norm(minor_pc)

        p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * hd + p5
        p2 = approach_normal * hd + p6
        p3 = approach_normal * hd + p7
        p4 = approach_normal * hd + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p


    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], triangles, color=color, opacity=0.5)
    

    def get_ind_points(self, grasp_bottom_center, approach_normal, binormal, minor_pc, all_points, p, vis=False):
        grasp_matrix = np.vstack([approach_normal, binormal, minor_pc])
        points = all_points - grasp_bottom_center.reshape(1, 3)     
        tmp = np.dot(grasp_matrix, points.T)       
        points = tmp.T

        # if way == "p_open":
        has_p, in_ind_ = self.check_collision(points, p[1], p[2], p[4], p[8])
        if not has_p:
            print("There is no point inside the gripper")
            return False, 1
        points_g = points[in_ind_]

        min_wide = min(points_g[:,1])
        max_wide = max(points_g[:,1])

        factor = 0
        percent = 0
        while percent < 0.09:
            factor += 0.05
            max_wide1 = (1-factor) * max_wide + factor * min_wide
            a = max_wide1 < points_g[:, 1]
            points_arry = points_g[a]
            percent = len(points_arry)/len(points_g)
        factor = 0
        percent = 0
        while percent < 0.09:    
            factor += 0.05
            min_wide1 = (1-factor) * min_wide + factor * max_wide
            a1 = min_wide1 > points_g[:, 1]
            points_arry = points_g[a]
            percent = len(points_arry)/len(points_g)
        dy_center = max_wide1 - min_wide1

        if vis:
            mlab.clf()
            self.show_grasp_3d(p)       #手爪
            self.show_points(points)    #旋转后的点云
            self.show_points(points_g, color='r')
            # mask = (points_g[:, 1] < max_wide1) & (points_g[:, 1] > min_wide1)
            # points_arry = points_g[mask]
            # self.show_points(points_arry, color='b')
            mlab.show()
        return dy_center, points_g


    def translation_hand(self, grasp_bottom_center, approach_normal, binormal, minor_pc, all_points, p, vis=False):
        """ 需要输入单位化处理后的方向向量"""
        # 旋转目标点云至抓取坐标系
        grasp_matrix = np.vstack([approach_normal, binormal, minor_pc])
        points = all_points - grasp_bottom_center.reshape(1, 3)     #相机坐标系下平移
        points_g = np.dot(points, grasp_matrix.T)                   #旋转
        if vis:         # 旋转后，平移缩放前的抓手坐标系下点云
            mlab.clf()
            self.show_grasp_3d(p)       #手爪
            self.show_points(points_g)  #旋转后的点云
            mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
            mlab.show()

        x_bis = self.translation_x1(points_g, p)
        if x_bis < -0.03:       # 允许关键点退至边缘0.005  threshold = self.gripper.real_hand_depth - 2*self.gripper.d_depth  # threshold=0.03为关键点允许回退的最大距离
            return False        # 手爪底部发生碰撞
        bis = self.translation_y1(points_g, p, x_bis)
        if not bis[0]: 
            return False

        bottom_center = grasp_bottom_center + np.dot(grasp_matrix.T, np.array([bis[2],bis[1],0]))      # grasp_bottom_center是相机坐标系的点
        
        if vis:     # 旋转平移缩放后的抓手坐标系下点云
            points_g = points_g - np.array([bis[2],bis[1],0])     #平移
            hand = self.get_hand_points(bis[3], np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
            
            mlab.clf()
            self.show_grasp_3d(hand)        #手爪
            self.show_points(points_g)      #旋转后的点云
            mlab.show()

        return True, bottom_center, bis[3], bis[2]
    

    def translation_hand1(self, grasp_bottom_center, grasp_matrix, all_points, p, vis=False):
        """ 需要输入单位化处理后的方向向量"""
        # 旋转目标点云至抓取坐标系
        points = all_points - grasp_bottom_center.reshape(1, 3)     #相机坐标系下平移
        points_g = np.dot(points, grasp_matrix.T)        #旋转
        if vis:         # 旋转后，平移缩放前的抓手坐标系下点云
            mlab.clf()
            self.show_grasp_3d(p)       #手爪##
            self.show_points(points_g)  #旋转后的点云
            mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))##
            mlab.show()

        x_bis = self.translation_x1(points_g, p)
        if x_bis < self.d_height:       # -0.035,允许关键点退至边缘0.005 
            return False                # 手爪底部发生碰撞
        bis = self.translation_y1(points_g, p, x_bis)
        if not bis[0]: 
            return False

        bottom_center = grasp_bottom_center + np.dot(grasp_matrix.T, np.array([bis[2],bis[1],0]))      # grasp_bottom_center是相机坐标系的点
        
        if vis:     # 旋转平移缩放后的抓手坐标系下点云
            points_g = points_g - np.array([bis[2],bis[1],0])     #平移
            hand = self.get_hand_points(bis[3], np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
            
            mlab.clf()
            self.show_grasp_3d(hand)        #手爪
            self.show_points(points_g)      #旋转后的点云
            mlab.show()

        return True, bottom_center, bis[3]
    

    def check_points_collision(self, ptg, all_points, vis=False):
        """ 需要输入单位化处理后的方向向量"""
        # 旋转目标点云
        grasp_matrix = ptg[1]
        points = all_points - ptg[0].reshape(1, 3)          #平移
        points_g = np.dot(points, grasp_matrix.T)           #旋转
        p = self.get_hand_points(ptg[2], np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        # if vis:                           # 旋转平移缩放后的抓手坐标系下点云
            # mlab.clf()
            # self.show_grasp_3d(p)         #手爪
            # self.show_points(points_g)    #旋转后的点云
            # # self.show_grasp_norm_oneside(np.zeros(3), approach_normal.T, binormal.T, minor_pc.T, scale_factor=0.001)      # 显示抓取坐标系
            # mlab.show()

        has_p= self.check_collision1(points_g, p[15], p[20])        # "p_bottom"
        if has_p:
            # print("手爪底部发生碰撞")
            return False,-4
        has_p = self.check_collision1(points_g, p[1], p[12])        # "p_left"
        if has_p:
            # print("手爪左边发生碰撞")
            return False,-2
        has_p = self.check_collision1(points_g, p[13], p[7])        # "p_right"
        if has_p:
            # print("手爪右边发生碰撞")
            return False,-3
           
        grasp_score = self.check_center(points_g, p[2], p[8])       # "p_open" # points_in_area, aim_grasp_rate, contact_height
        
        if vis:
            # print("points_in_area", len(points_in_area))
            mlab.clf()
            self.show_grasp_3d(p)           #手爪
            self.show_points(points_g)      #旋转后的点云
            self.show_points(points_g[grasp_score[0]], color='r')       #抓手内部点云
            mlab.show()
        # print("成功获取一个姿态")
        return True, grasp_score
    

    def check_collision(self, all_points, s1, s2, s4, s8):
        a1 = s1[1] < all_points[:, 1]
        a2 = s2[1] > all_points[:, 1]
        a3 = s1[2] > all_points[:, 2]
        a4 = s4[2] < all_points[:, 2]
        a5 = s4[0] > all_points[:, 0]
        a6 = s8[0] < all_points[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

        return has_p, points_in_area
    

    def check_collision1(self, all_points, p1, p2):          # 左下右上两点
        mask = np.all(all_points > p2, axis=1) & np.all(all_points < p1,axis=1)
        index_in_area = np.where(mask)[0]
        if len(index_in_area) == 0:
            return False
        else:
            return True
        

    def check_center(self, all_points, p1, p2):             # 左下右上两点
        mask = np.all(all_points > p2, axis=1) & np.all(all_points < p1,axis=1)
        index_in_area = np.where(mask)[0]
        points_in_area = all_points[index_in_area]
        contact_height = min((self.finger_height - points_in_area[:,0].min())/self.finger_height, 1)

        p2[0] = p1[0] - self.grasp_cfg['perceived_contact_height'] 
        mask = np.all(all_points > p2, axis=1) & np.all(all_points < p1,axis=1)
        index_in_area1 = np.where(mask)[0]

        return [index_in_area, index_in_area1, contact_height]
        


    def find_group(self, data):
        sorted_data = np.sort(data)
        diff = np.diff(sorted_data)
        split_indices = np.where(diff > self.find_width)[0]
        if split_indices.shape[0] == 0:
            return data
        groups = np.split(sorted_data, split_indices + 1)
        for group in groups:
            if 0 < group.max() and 0 > group.min():
                return group
        return data
        

    def translation_y(self, points, hand):
        threshold = self.gripper.max_width - 2 * self.d_width        # threshold=0.08为减去预留后的点云最大宽度

        # y方向平移
        s1, s4, s8 = hand[1], hand[4], hand[8]
        a3 = s1[2] > points[:, 2]
        a4 = s4[2] < points[:, 2]
        a5 = s4[0] > points[:, 0]
        a6 = s8[0] < points[:, 0]

        a = np.vstack([a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) < 10:
            return False, -1                # 表示内部点过少
        Internal_points = points[points_in_area][:, 1]
        lang = Internal_points.max() - Internal_points.min()
        open_w = self.gripper.max_width
        if abs(lang) > threshold:
            Internal_points = self.find_group(Internal_points)
            lang = Internal_points.max() - Internal_points.min()
            if abs(lang) > threshold:
                return False, -2            # 表示内部点过多，超过加爪最大开合度
            open_w = max(min(0.085, lang + 2*self.d_width), 0)     
       
        bis = (Internal_points.max() + Internal_points.min())*0.5
        if abs(bis) > 0.02:
            return False, -1
        return True, bis, open_w
    

    def translation_y1(self, points, hand, x_bis):
        threshold = self.d_width        # threshold=0.08为减去预留后的点云最大宽度
        
        s1, s4, s8 = hand[1], hand[4], hand[8]
        a1 = s1[2] > points[:, 2]
        a2 = s4[2] < points[:, 2]
        a3 = (s8[0]+x_bis) < points[:, 0]
        mask = a1 & a2 & a3
        points_in_area = points[mask]

        # Internal_points0 = points_in_area[(s4[0]+0.005) > points_in_area[:, 0]]

        # self.show_grasp_3d(hand)          # 手爪##
        # self.show_points(points)          # 旋转后的点云
        # self.show_points(Internal_points0, color='r')  #旋转后的点云
        # mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))##
        # mlab.show()

        open_w = 0.0
        d_x = self.d_height             # 手指尖端测试
        Internal_points0 = points_in_area[(s4[0]+d_x) > points_in_area[:, 0]][:, 1]
        if Internal_points0.shape[0]  > self.grasp_cfg['min_pc_number']:     
            lang = Internal_points0.max() - Internal_points0.min()
            if lang < threshold:
                open_w = self.gripper_cfg['max_width']
                x_bis1 = d_x
            else:
                Internal_points0 = self.find_group(Internal_points0)
                lang = Internal_points0.max() - Internal_points0.min()
                if lang < threshold:
                    open_w = min(0.085, lang + 2*self.grasp_cfg['finger_expand_width'])     # 手抓内部最大张开度0.085
                    x_bis1 = d_x
        if (open_w < 0.005) or (open_w > self.gripper_cfg['max_width']):
            return False, -1            # 内部点过多或过少
        for d_x in np.arange(x_bis, self.d_height, self.grasp_cfg['d_z']):
            points_in_area = points_in_area[(s4[0]+d_x) > points_in_area[:, 0]]
            Internal_points = points_in_area[:, 1]
            lang = Internal_points.max() - Internal_points.min()
            if lang < threshold:
                open_w = self.gripper_cfg['max_width']
                x_bis1 = d_x
                break

            Internal_points = self.find_group(Internal_points)
            lang = Internal_points.max() - Internal_points.min()
            if lang < threshold:
                open_w = min(0.085, lang + 2*self.grasp_cfg['finger_expand_width']) 
                x_bis1 = d_x
                break
       
        y_bis = (Internal_points.max() + Internal_points.min())*0.5
        if abs(y_bis) > self.grasp_cfg['y_bis']:
            return False, -2            # 偏置不满足条件
        return True, y_bis, x_bis1, open_w
    
    
    def translation_x(self, all_points, hand):
        s1, s2, s4, s8 = hand[9], hand[1], hand[10], hand[12]        # elif way == "p_left":
        s5, s6 = hand[2], hand[13]
        a1 = (s1[1] < all_points[:, 1]) | (s5[1] < all_points[:, 1])
        a2 = (s2[1] > all_points[:, 1]) | (s6[1] > all_points[:, 1])
        a3 = s1[2] > all_points[:, 2]
        a4 = s4[2] < all_points[:, 2]
        a5 = s4[0] > all_points[:, 0]
        a6 = s8[0] < all_points[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        internal_points = all_points[points_in_area][:, 0]
        if internal_points.min() < 0.005:
            return True, -0.03        # 

        return True, internal_points.min()-0.035
    

    def translation_x1(self, all_points, hand):
        # 感知手爪底部碰撞
        a1 = hand[8, 1] < all_points[:, 1]
        a2 = hand[7, 1] > all_points[:, 1]
        a3 = hand[17, 2] > all_points[:, 2]
        a4 = hand[8, 2] < all_points[:, 2]
        a5 = hand[8, 0] > all_points[:, 0]
        a6 = hand[17, 0] < all_points[:, 0]
        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        internal_points = all_points[points_in_area][:, 0]
        if internal_points.size == 0:
            return 0

        return internal_points.min()


    def show_all_grasps(self, all_points, grasps_for_show):
        mlab.clf()
        for grasp_ in grasps_for_show:
            hand_points = self.get_hand_points(grasp_[4], grasp_[0], grasp_[1], grasp_[2], grasp_[3])
            self.show_grasp_3d(hand_points)
        self.show_points(all_points, scale_factor=0.003)
        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
        mlab.show()


    def show_highlight_grasps(self, all_points, colors, grasps_for_show, selected=0):
        mlab.clf()
        for i, grasp_ in enumerate(grasps_for_show):
            # hand_points = ags.get_hand_points(grasp_[0], grasp_[1], grasp_[2], grasp_[3])
            hand_points = self.get_min_hand_points(grasp_[4], grasp_[0], grasp_[1], grasp_[2], grasp_[3])
            if i == selected:
                self.show_grasp_3d(hand_points, color=(1, 0, 0))
            else:
                self.show_grasp_3d(hand_points)
        # self.show_points(all_points, scale_factor=0.003)
        node = mlab.points3d(all_points[:, 0], all_points[:, 1], all_points[:, 2], scale_factor=.001)
        node.glyph.scale_mode = 'scale_by_vector'
        node.mlab_source.dataset.point_data.scalars = colors

        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
        mlab.show()


    def show_highlight_grasps1(self, all_pcd, grasp_sampled, selected=0):
        # open3d 网格显示
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        triangles = [[9,4,1], [4,9,10], [10,8,4], [8,10,12], [1,4,8 ], [8,5,1],[ 1,5,9], [11, 9,5],[20, 10,9], [9, 17,20], 
                 [17,19,20], [17, 18,19], [20, 19, 12], [19,16,12],[11, 15,17], [18, 17, 15], [8,7,6], (8,6,5),
                 [19,18,14], [13,14,18], [2,3,13], [14,13,3],[3, 6, 7], [6, 3, 2],[3,7,14], [16,14,7], [2, 13, 15], [15, 6,2],]       # 跟索引顺序有关
        for i, grasp_ in enumerate(grasp_sampled):
            hand_points = self.get_min_hand_points(grasp_[4], grasp_[0], grasp_[1], grasp_[2], grasp_[3])
            tri_mesh = o3d.geometry.TriangleMesh()
            tri_mesh.vertices = o3d.utility.Vector3dVector(hand_points)
            tri_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            tri_mesh.paint_uniform_color([0.0, 0.0, 1.0])
            vis.add_geometry(tri_mesh)
        vis.add_geometry(all_pcd)
        vis.run()
        vis.destroy_window()


    def show_grasps(self, all_points, grasps_for_show):
        mlab.clf()
        hand_points = self.get_hand_points(grasps_for_show[0], grasps_for_show[1], grasps_for_show[2], grasps_for_show[3])
        self.show_grasp_3d(hand_points)
        self.show_points(all_points, scale_factor=0.006)
        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
        mlab.show()


    def furthest_point_sampling(self, points, num_samples):
        num_points = points.shape[0]
        selected_indices = []
        distances = np.full(num_points, np.inf)

        first_index = np.random.randint(num_points)
        selected_indices.append(first_index)
        distances = np.minimum(distances, np.linalg.norm(points - points[first_index], axis=1))

        for _ in range(1, num_samples):
            farthest_index = np.argmax(distances)
            selected_indices.append(farthest_index)
            distances = np.minimum(distances, np.linalg.norm(points - points[farthest_index], axis=1))

        return selected_indices
    

    def sample_grasps2(self, pcd, fd_pcd, N, cube_center, r_vertex, **kwargs):    
        processed_potential_grasp = []

        # 计算点云法向量
        fd_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            self.grasp_cfg['normal_estimation']['radius'], self.grasp_cfg['normal_estimation']['max_nn']))     # 混合领域搜索,执行法线估计
        all_points = np.asarray(fd_pcd.points)
        all_normals = np.asarray(fd_pcd.normals)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            self.grasp_cfg['normal_estimation']['radius'], self.grasp_cfg['normal_estimation']['max_nn']))     # 混合领域搜索,执行法线估计
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)      # 可视化原始估计法线
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        vector_p2cam = -points              # 点云到相机的方向
        vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)   #向量单位化处理
        tmp = np.dot(vector_p2cam, normals.T).diagonal()     #计算向量与法向量点积
        wrong_dir_norm = np.where(tmp < 0)[0]
        tmp = np.ones([len(tmp), 3])
        tmp[wrong_dir_norm, :] = -1
        normals = normals * tmp             #与表面法相元素对元素相乘，作用是将"错误的"法向量的方向反向
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)              # 可视化修改方向后的估计法线


        # 获得采样点
        sampled_indices = self.furthest_point_sampling(points, self.grasp_cfg['num_samples'])

        # # 指定要生成的渐变颜色数量
        # num_colors = len(sampled_indices)
        # # 生成渐变颜色数组
        # sampled_points_colors = np.zeros((num_colors, 3))
        # sampled_points_colors[:, 0] = np.linspace(0, 1, num_colors)
        # sampled_points_colors[:, 2] = np.linspace(1, 0, num_colors)
        # pcd_sampled1 = o3d.geometry.PointCloud()
        # pcd_sampled1.points = o3d.utility.Vector3dVector(points[sampled_indices])
        # pcd_sampled1.colors = o3d.utility.Vector3dVector(sampled_points_colors)
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])          # 将采样点设置为灰色
        # o3d.visualization.draw_geometries([pcd,pcd_sampled1])   

        # 获取手爪点云
        hand_points = self.get_hand_points(self.gripper_cfg['max_width'], np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for ind in range(len(sampled_indices)-1,-1,-1):  
            num = 1    
            selected_surface = points[sampled_indices[ind]]

            M = np.zeros((3, 3))
            [num_hybrid, idx_hybrid, _] = pcd_tree.search_hybrid_vector_3d(points[sampled_indices[ind]], 
                                                                           self.grasp_cfg['coordinate_estimation']['radius'], 
                                                                           self.grasp_cfg['coordinate_estimation']['max_nn'])
            for idx in idx_hybrid[1:]:
                normal = normals[idx, :]
                normal = normal.reshape(-1, 1)
                M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour")
                continue 

            eigval, eigvec = np.linalg.eig(M)                       # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)      # z 
            minor_pc /= np.linalg.norm(minor_pc)
            if np.dot([0,1,0], minor_pc) > 0:                       # 调整z坐标系方向，向上为正
                    minor_pc = -minor_pc

            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)    # x 
            new_normal /= np.linalg.norm(new_normal)
            if np.dot(selected_surface, new_normal) > 0:            # 调整x坐标系方向
                new_normal = -new_normal
            major_pc = np.cross(minor_pc, new_normal)               # y 
            major_pc /= np.linalg.norm(major_pc)
            # mlab.clf()
            # self.show_points(selected_surface, color='g', scale_factor=.003)          # 显示抓取点
            # self.show_points(all_points, color='r', scale_factor=.002)                # 显示物体点云
            # self.show_grasp_norm_oneside(selected_surface, new_normal, major_pc, minor_pc, scale_factor=0.001)      # 显示抓取坐标系
            
            # # hand_points = self.get_hand_points(ptg[4], ptg[0], ptg[1], ptg[2], ptg[3])
            # # self.show_grasp_3d(hand_points)
            # # self.show_grasp_norm_oneside(ptg[0], ptg[1], ptg[2], ptg[3], scale_factor=0.001)                    # 显示抓取坐标系
            # # self.show_grasp_norm_oneside(np.array([0,0,0]), (1,0,0), (0,1,0), (0,0,1), scale_factor=0.001)      # 显示抓取坐标系
            # mlab.show()
            
            grasp_matrix0 = np.vstack([-new_normal, major_pc, -minor_pc])
            for ytheta, xtheta, ztheta in itertools.product(self.grasp_cfg['rotation_y'], self.grasp_cfg['rotation_x'], self.grasp_cfg['rotation_z']):
                rotation_matrix=Rotation.from_euler('xyz', [xtheta,ytheta,ztheta], degrees=True).as_matrix().T      # 将旋转矩阵换为欧拉角
                grasp_matrix = rotation_matrix.dot(grasp_matrix0)
                if np.dot(grasp_matrix[0], [0,0,1]) < 0.0:         
                    continue
                has_g =self.translation_hand1(selected_surface, grasp_matrix, all_points, hand_points, vis=self.grasp_cfg['debug_vis'])
                if not has_g:
                    continue
                ptg = [has_g[1], grasp_matrix, has_g[2]]
                is_collide, grasp_score =self.check_points_collision(ptg, all_points, vis=False)      # 整体碰撞检测
                if is_collide:
                    points_in_area = all_points[grasp_score[0]]
                    mask = np.all(points_in_area > r_vertex, axis=1) & np.all(points_in_area < (2*cube_center - r_vertex),axis=1)
                    index_in_aim = np.where(mask)[0]
                    aim_grasp_rate = len(index_in_aim)/len(grasp_score[0])
                    aim_grasp_rate = np.sin(0.5 * aim_grasp_rate * np.pi)
                    
                    # if aim_grasp_rate > 0.6:
                    #     aim_grasp_rate = 2*(aim_grasp_rate - 0.6)+0.2
                    # else:
                    #     aim_grasp_rate *= 1/3
                    if len(grasp_score[1])==0:
                        flatten = 0.5
                    else:
                        approach = ptg[1][1]
                        pc_normal = all_normals[grasp_score[1]]
                        tmp = np.dot(pc_normal, approach)               # 计算向量与法向量点积
                        tmp_x = np.abs(tmp).sum()/len(grasp_score[1])   
                        tmp_y = np.abs(tmp.sum())/len(grasp_score[1])   
                        flatten = tmp_x*(1-0.35*tmp_y)
                    score = 0.5*flatten + 0.08*grasp_score[2] + 0.3*aim_grasp_rate
                    # ptg += [flatten, grasp_score[2], aim_grasp_rate, len(grasp_score[0]), score]         # 越大越好
                    ptg += [len(grasp_score[0]), score]   
                    # mlab.clf()
                    # self.show_points(all_points)                      # 旋转后的点云
                    # self.show_points(all_points[ind_points_index], color='r')       # 抓手内部点云
                    # mlab.show()
                    processed_potential_grasp.append(ptg)
                    if len(processed_potential_grasp) >= N:
                        # self.show_all_grasps(all_points, processed_potential_grasp)
                        return processed_potential_grasp                 
                    if num == 0:
                        break
                    num -= 1

            # print("The grasps number got:", len(processed_potential_grasp))  
        return processed_potential_grasp
    

    def sample_grasps3(self, pcd, fd_pcd, N, **kwargs):    
        # 计算点云法向量
        fd_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            self.grasp_cfg['normal_estimation']['radius'], self.grasp_cfg['normal_estimation']['max_nn']))     # 混合领域搜索,执行法线估计

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            self.grasp_cfg['normal_estimation']['radius'], self.grasp_cfg['normal_estimation']['max_nn']))     # 混合领域搜索,执行法线估计
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)      # 可视化原始估计法线
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        vector_p2cam = -points           # 点云到相机的方向
        vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)   #向量单位化处理
        tmp = np.dot(vector_p2cam, normals.T).diagonal()     #计算向量与法向量点积
        wrong_dir_norm = np.where(tmp < 0)[0]
        tmp = np.ones([len(tmp), 3])
        tmp[wrong_dir_norm, :] = -1
        normals = normals * tmp       #与表面法相元素对元素相乘，作用是将"错误的"法向量的方向反向
        pcd.normals = o3d.utility.Vector3dVector(normals)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)          # 可视化修改方向后的估计法线


        # 获得采样点
        sampled_indices = self.furthest_point_sampling(points, self.grasp_cfg['num_samples'])

        # # 指定要生成的渐变颜色数量
        # num_colors = len(sampled_indices)
        # # 生成渐变颜色数组
        # sampled_points_colors = np.zeros((num_colors, 3))
        # sampled_points_colors[:, 0] = np.linspace(0, 1, num_colors)
        # sampled_points_colors[:, 2] = np.linspace(1, 0, num_colors)
        # pcd_sampled1 = o3d.geometry.PointCloud()
        # pcd_sampled1.points = o3d.utility.Vector3dVector(points[sampled_indices])
        # pcd_sampled1.colors = o3d.utility.Vector3dVector(sampled_points_colors)
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])          # 将采样点设置为灰色
        # o3d.visualization.draw_geometries([pcd,pcd_sampled1])   

        # 获取手爪点云
        self.hand_points = self.get_hand_points(self.gripper_cfg['max_width'], np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        
        num_workers = 2
        sampled_num = int(self.grasp_cfg['num_samples']/num_workers)
        
        queue = mp.Queue()
        # num_grasps_p_worker = int(num_grasps/num_workers)
        workers = [mp.Process(target=self.grasp_task, args=(pcd, fd_pcd, N, sampled_indices[j*sampled_num:(j+1)*sampled_num], queue)) for j in range(num_workers)]
        [i.start() for i in workers]

        processed_potential_grasp = []
        for _ in range(num_workers):
            processed_potential_grasp = processed_potential_grasp + queue.get()
        return processed_potential_grasp


    def grasp_task(self, pcd, fd_pcd, N, sampled_indices, queue_):
            ret = self.sample_grasps4(pcd, fd_pcd, N, sampled_indices)
            queue_.put(ret)


    def sample_grasps4(self, pcd, fd_pcd, N, sampled_indices, **kwargs):    
        processed_potential_grasp = []
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        all_points = np.asarray(fd_pcd.points)
        all_normals = np.asarray(fd_pcd.normals)
        for ind in range(len(sampled_indices)):  
            num = 1    
            selected_surface = points[sampled_indices[ind]]

            # 计算黑赛矩阵
            M = np.zeros((3, 3))
            [num_hybrid, idx_hybrid, _] = pcd_tree.search_hybrid_vector_3d(points[sampled_indices[ind]], 
                                                                           self.grasp_cfg['coordinate_estimation']['radius'], 
                                                                           self.grasp_cfg['coordinate_estimation']['max_nn'])
            for idx in idx_hybrid[1:]:
                normal = normals[idx, :]
                normal = normal.reshape(-1, 1)
                M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour")
                continue 

            eigval, eigvec = np.linalg.eig(M)                    # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)   # z minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)
            if np.dot([0,1,0], minor_pc) > 0:                # 调整z坐标系方向，向上为正
                    minor_pc = -minor_pc

            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # x estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)
            if np.dot(selected_surface, new_normal) > 0:         # 调整x坐标系方向
                new_normal = -new_normal
            major_pc = np.cross(minor_pc, new_normal)            # y major principal curvature
            major_pc /= np.linalg.norm(major_pc)
            # mlab.clf()
            # self.show_points(selected_surface, color='g', scale_factor=.003)    # 显示抓取点
            # self.show_points(all_points, color='r', scale_factor=.002)                # 显示物体点云
            # self.show_grasp_norm_oneside(selected_surface, new_normal, major_pc, minor_pc, scale_factor=0.001)      # 显示抓取坐标系
            
            # # hand_points = self.get_hand_points(ptg[4], ptg[0], ptg[1], ptg[2], ptg[3])
            # # self.show_grasp_3d(hand_points)
            # # self.show_grasp_norm_oneside(ptg[0], ptg[1], ptg[2], ptg[3], scale_factor=0.001)      # 显示抓取坐标系
            # # self.show_grasp_norm_oneside(np.array([0,0,0]), (1,0,0), (0,1,0), (0,0,1), scale_factor=0.001)      # 显示抓取坐标系
            # mlab.show()
            
            grasp_matrix0 = np.vstack([-new_normal, major_pc, -minor_pc])
            for ytheta, xtheta, ztheta in itertools.product(self.grasp_cfg['rotation_y'], self.grasp_cfg['rotation_x'], self.grasp_cfg['rotation_z']):
                rotation_matrix=Rotation.from_euler('xyz', [xtheta,ytheta,ztheta], degrees=True).as_matrix().T            #将旋转矩阵换为欧拉角
                grasp_matrix = rotation_matrix.dot(grasp_matrix0)
                if np.dot(grasp_matrix[0], selected_surface) < -0.5:         
                    continue
                has_g =self.translation_hand1(selected_surface, grasp_matrix, all_points, self.hand_points, vis=self.grasp_cfg['debug_vis'])
                if not has_g:
                    continue
                ptg = [has_g[1], grasp_matrix, has_g[2]]
                is_collide, grasp_score =self.check_points_collision(ptg, all_points, vis=False)      # 整体碰撞检测
                if is_collide:
                    points_in_area = all_points[grasp_score[0]]
                    mask = np.all(points_in_area > self.r_vertex, axis=1) & np.all(points_in_area < (2*self.cube_center - self.r_vertex),axis=1)
                    index_in_aim = np.where(mask)[0]
                    aim_grasp_rate = len(index_in_aim)/len(grasp_score[0])
                    aim_grasp_rate = np.sin(0.5 * aim_grasp_rate * np.pi)
                    
                    if len(grasp_score[1])==0:
                        flatten = 0.8
                    else:
                        approach = ptg[1][1]
                        pc_normal = all_normals[grasp_score[1]]
                        tmp = np.abs(np.dot(pc_normal, approach))    #计算向量与法向量点积
                        flatten = tmp.sum()/len(grasp_score[1])
                    score = 0.5*flatten + 0.2*grasp_score[2] + 0.3*aim_grasp_rate
                    ptg += [flatten, grasp_score[2], aim_grasp_rate, score]         # 越大越好
                    # mlab.clf()
                    # self.show_points(all_points)  #旋转后的点云
                    # self.show_points(all_points[ind_points_index], color='r')       #抓手内部点云
                    # mlab.show()
                    processed_potential_grasp.append(ptg)
                    if len(processed_potential_grasp) >= N:
                        # self.show_all_grasps(all_points, processed_potential_grasp)
                        return processed_potential_grasp                 
                    if num == 0:
                        break
                    num -= 1

            print("The grasps number got:", len(processed_potential_grasp))  

        # self.show_all_grasps(all_points, processed_potential_grasp)
        return processed_potential_grasp
    
    

    def get_show_hand1(self, grasp_sampled):
        # open3d 简化圆柱显示， 显示颜色渐变
        color = np.array([uniform(0,0.2), uniform(0,0.2), 1])
        d_color = (0.9 - color[:2].max())/len(grasp_sampled)
        graspers = o3d.geometry.TriangleMesh()

        hand = get_simplified_hand(grasp_sampled[0][2], 0.003)
        hand.paint_uniform_color([1,0.1,0.2])
        hand.translate(grasp_sampled[0][0] - grasp_sampled[0][1][0] * self.grasp_cfg['center_height'])
        a = np.vstack([grasp_sampled[0][1][1],grasp_sampled[0][1][2],grasp_sampled[0][1][0]])
        hand.rotate(a.T)
        graspers += hand
        for grasp_ in grasp_sampled[1:]:
            hand = get_simplified_hand(grasp_[2], 0.002)
            hand.paint_uniform_color(color)
            color = color + np.array([d_color, d_color, 0])

            hand.translate(grasp_[0] - grasp_[1][0] * self.grasp_cfg['center_height'])
            a = np.vstack([grasp_[1][1],grasp_[1][2],grasp_[1][0]])
            hand.rotate(a.T)
            graspers += hand
        return graspers
    


def get_simplified_hand(open, radius):
    height = 0.05 + radius
    open_w = max(min(0.085, open), 0) +  radius*2             # 手抓内部最大张开度0.085

    link1 = o3d.geometry.TriangleMesh.create_cylinder(radius, height=height)
    link2 = o3d.geometry.TriangleMesh.create_cylinder(radius, height=height)
    link3 = o3d.geometry.TriangleMesh.create_cylinder(radius, height=height)
    link4 = o3d.geometry.TriangleMesh.create_cylinder(radius, height=open_w)
    
    link1.translate(np.array([open_w*0.5, 0, height*0.5]))
    link2.translate(np.array([-open_w*0.5, 0, height*0.5]))
    link3.translate(np.array([0, 0, -height*0.5]))
    link4.rotate(np.array([[0, 0, -1],[0, 1, 0],[1, 0, 0]]))

    hand = o3d.geometry.TriangleMesh()
    hand += link1
    hand += link2
    hand += link3
    hand += link4
    hand.translate(np.array([0, 0, -radius]))
    # o3d.visualization.draw_geometries([link1,link2,link3,link4,FOR1])
    return hand
    

def get_show_hand(grasp_sampled, selected=0):
    # open3d 简化圆柱显示， 制定抓取显示颜色加深
    color = np.random.rand(3)
    graspers = o3d.geometry.TriangleMesh()
    for i, grasp_ in enumerate(grasp_sampled):
        if i != selected:
            hand = get_simplified_hand(grasp_[2], 0.002)
            hand.paint_uniform_color(color)
        else:
            hand = get_simplified_hand(grasp_[2], 0.003)
            hand.paint_uniform_color(np.maximum(0, color - 0.3))        # 加深

        hand.translate(grasp_[0] - grasp_[1][0] * 0.015)
        a = np.vstack([grasp_[1][1],grasp_[1][2],grasp_[1][0]])
        hand.rotate(a.T)
        graspers += hand
    return graspers


def get_one_show_hand(grasp_):
    # open3d 简化圆柱显示
    color = np.random.rand(3)
    hand = get_simplified_hand(grasp_[4], 0.002)
    hand.paint_uniform_color(color)

    hand.translate(grasp_[0] - grasp_[1] * 0.015)
    a = np.vstack([grasp_[2],grasp_[3],grasp_[1]])
    hand.rotate(a.T)
    return hand



if __name__ == '__main__':
    ws_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))        
    ags = GraspSampler(ws_path, 'robotiq_85')
    hand_points = ags.get_min_hand_points(0.06, np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    ags.show_grasp_3d(hand_points)
    mlab.show()
    hand_points = ags.get_hand_points(0.08, np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    ags.show_grasp_3d(hand_points)
    mlab.show()

