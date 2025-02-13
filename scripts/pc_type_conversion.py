#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sensor_msgs.msg import PointCloud2, PointField
import numpy as np

type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)

pftype_sizes = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8
}

def pointcloud2_to_array(msg: PointCloud2, split_rgb=False, remove_nans=True):
    offset = 0
    np_dtype_list = []
    for f in msg.fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % ('__', offset), np.uint8))
            offset += 1
        np_dtype_list.append((f.name, pftype_to_nptype[f.datatype]))
        offset += pftype_sizes[f.datatype]

    # might be extra padding between points
    while offset < msg.point_step:
        np_dtype_list.append(('%s%d' % ('__', offset), np.uint8))
        offset += 1

    cloud_arr = np.fromstring(msg.data, np_dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[[
        fname for fname, _type in np_dtype_list
        if not (fname[:len('__')] == '__')
    ]]

    if split_rgb:
        cloud_arr = split_rgb_field(cloud_arr)
        # points = np.zeros(list(cloud_rgb_arr.shape), 1, dtype={ "names": ( "x", "y", "z", "r", "g", "b"), 
        #         "formats": ( "f4", "f4", "f4", "u1", "u1", "u1" )})

        # points["x"] = cloud_arr['x']
        # points["y"] = cloud_arr['y']
        # points["z"] = cloud_arr['z']
        # points["r"] = cloud_arr['r']
        # points["g"] = cloud_arr['g']
        # points["b"] = cloud_arr['b']
        # return points

    # cloud_array = np.reshape(cloud_arr, (msg.height, msg.width))

    # if remove_nans:
    #     mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
    #     cloud_array = cloud_array[mask]

    return cloud_arr


def split_rgb_field(cloud_arr):
    """Takes an array with a named 'rgb' float32 field, and returns an array in which
    this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.

    (pcl stores rgb in packed 32 bit floats)
    """
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    r1 = np.asarray((rgb_arr >> 16) / 255, dtype=np.float16).reshape(-1, 1)
    g1 = np.asarray(((rgb_arr >> 8) & 255)/ 255, dtype=np.float16).reshape(-1, 1)
    b1 = np.asarray((rgb_arr & 255)/ 255, dtype=np.float16).reshape(-1, 1)
    combined_array = np.concatenate((r1, g1, b1), axis=1)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgb':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr

def split_rgb(cloud_arr):
    """Takes an array with a named 'rgb' float32 field, and returns an array in which
    this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.

    (pcl stores rgb in packed 32 bit floats)
    """
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) / 255, dtype=np.float16).reshape(-1, 1)      # 缩放0-1之间
    g = np.asarray(((rgb_arr >> 8) & 255)/ 255, dtype=np.float16).reshape(-1, 1)
    b = np.asarray((rgb_arr & 255)/ 255, dtype=np.float16).reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)

    x = np.asarray(cloud_arr['x'].copy(), dtype=np.float16).reshape(-1, 1)
    y = np.asarray(cloud_arr['y'].copy(), dtype=np.float16).reshape(-1, 1)
    z = np.asarray(cloud_arr['z'].copy(), dtype=np.float16).reshape(-1, 1)
    points = np.concatenate((x, y, z), axis=1)

    return points, colors
