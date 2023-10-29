#! /usr/bin/env python2

"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

# Requirements: Numpy as PIL/Pillow
import numpy as np
from PIL import Image
import glob
import json
import os
import shutil

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple. 
    
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

   
def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_write(filename, depth):
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    
    depth.astype(np.float32).tofile(f)
    f.close()


def disparity_write(filename,disparity,bitdepth=16):
    """ Write disparity to file.

    bitdepth can be either 16 (default) or 32.

    The maximum disparity is 1024, since the image width in Sintel
    is 1024.
    """
    d = disparity.copy()

    # Clip disparity.
    d[d>1024] = 1024
    d[d<0] = 0

    d_r = (d / 4.0).astype('uint8')
    d_g = ((d * (2.0**6)) % 256).astype('uint8')

    out = np.zeros((d.shape[0],d.shape[1],3),dtype='uint8')
    out[:,:,0] = d_r
    out[:,:,1] = d_g

    if bitdepth > 16:
        d_b = (d * (2**14) % 256).astype('uint8')
        out[:,:,2] = d_b

    Image.fromarray(out,'RGB').save(filename,'PNG')


def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return depth


def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def cam_write(filename, M, N):
    """ Write intrinsic matrix M and extrinsic matrix N to file. """
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    M.astype('float64').tofile(f)
    N.astype('float64').tofile(f)
    f.close()


def segmentation_write(filename,segmentation):
    """ Write segmentation to file. """

    segmentation_ = segmentation.astype('int32')
    seg_r = np.floor(segmentation_ / (256**2)).astype('uint8')
    seg_g = np.floor((segmentation_ % (256**2)) / 256).astype('uint8')
    seg_b = np.floor(segmentation_ % 256).astype('uint8')

    out = np.zeros((segmentation.shape[0],segmentation.shape[1],3),dtype='uint8')
    out[:,:,0] = seg_r
    out[:,:,1] = seg_g
    out[:,:,2] = seg_b

    Image.fromarray(out,'RGB').save(filename,'PNG')


def segmentation_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    seg_r = f_in[:,:,0].astype('int32')
    seg_g = f_in[:,:,1].astype('int32')
    seg_b = f_in[:,:,2].astype('int32')

    segmentation = (seg_r * 256 + seg_g) * 256 + seg_b
    return segmentation


def read_validation_indices(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    indices = []
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            index = line.split('frame_')[1].split('.png')[0]
            indices.append(index)
    
    return indices

def generate_transforms(dpt_path, file_path, indices, total_index = 50, train=True):
    cam_paths = glob.glob(dpt_path+"*.cam")
    # depth_paths = glob.glob("*.dpt")
    
    data = {
        "camera_angle_x": None,
        "frames": []
    }
    
    timestep = 1 / (total_index - 1)
    for index in indices:
        i = int(index)-1
        intrin, extrin = cam_read(cam_paths[i])
        
        # Transformation matrix
        T = np.zeros((4, 4))
        T[:3, :3] = extrin[:, :3]
        T[:3, 3] = extrin[:, 3]
        T[3, 3] = 1

        if (data["camera_angle_x"] is None):
            # Compute the camera angle x
            w = intrin[0, 2] * 2  # the width of the image in pixels
            f = intrin[0][0]  # the x focal length in pixels
            camera_angle_x = 2 * np.arctan2(w, 2 * f)
            data["camera_angle_x"] = camera_angle_x
            data["focal_length"] = f

        # Compute the rotation angle
        R = extrin[:3, :3]  # the rotation matrix
        theta = np.arctan2(R[0, 2], R[0, 0])  # the rotation angle around the y-axis (yaw)

        # Get file name
        name = cam_paths[i].replace(".cam","")
        
        cam = name.split("\\")[-1]
        # print(name, cam)
        mode = 'train' if train else 'validation'

        print(mode)

        frame = {
                "file_path": f"./{mode}/{cam}",
                "rotation": theta,
                "time": i*timestep,
                "transform_matrix": T.tolist()
            }
        
        data["frames"].append(frame)

        
    with open(file_path, "w") as f:
        print(file_path)
        json.dump(data, f)

def generate_npy(cam_folder, save_folder):
    cam_paths = glob.glob(cam_folder+"*.cam")
    depth_paths = glob.glob(cam_folder+"*.dpt")

    poses = np.zeros((len(depth_paths), 15))

    min_dpt = float("inf")
    max_dpt = -float("inf")


    for i in range(len(cam_paths)):
        intrin, extrin = cam_read(cam_paths[i])
        height = intrin[1, 2] * 2
        width = intrin[0, 2] * 2
        focal = (intrin[0, 0] + intrin[1, 1]) / 2
        R = extrin[:3, :3]
        R[:, 1] = -R[:, 1]
        R[:, [0, 1]] = R[:, [1, 0]]
        extrin_inv = np.linalg.inv(R)
        t_inv = -extrin_inv @ extrin[:3, 3]
        pose = intrin @ np.concatenate([extrin_inv, t_inv.reshape(3, 1), np.array([height, width, focal]).reshape(3, 1)], axis=1)
        poses[i] = pose.flatten()
        dpt = depth_read(depth_paths[i])
        if np.min(dpt) < min_dpt:
            min_dpt = np.min(dpt)
        if np.max(dpt) > max_dpt:
            max_dpt = np.max(dpt)
    a = np.concatenate([poses[0], np.array([min_dpt, max_dpt])]).reshape(1, -1).repeat(2, axis=0)
    np.save(save_folder+"\poses_bounds.npy", a)
    a = np.load(save_folder+"\poses_bounds.npy")

    

validation_indices_file = "validation_indices.md"
validation_indices = read_validation_indices(validation_indices_file)
train_indices = [f'{i:04d}' for i in range(1,51) if f'{i:04d}' not in validation_indices]
scenes = ['alley_2', "cave_2"]

for s in scenes:
    data_folder = f'mpi/albedo/{s}'
    train_folder = data_folder+'/train'
    validation_folder = data_folder+'/validation'
    try:
        os.makedirs(train_folder)
        os.makedirs(validation_folder)

        # Iterate over the files in the data folder
        for file_path in glob.glob(os.path.join(data_folder, "*")):
            file_name = os.path.basename(file_path)
            scene_name = file_name.split(".")[0]  # Extract scene name from file name
            no = scene_name[8:]
            # Check if the scene is in the validation list
            if no in validation_indices:
                shutil.move(file_path, os.path.join(validation_folder, file_name))
            else:
                shutil.move(file_path, os.path.join(train_folder, file_name))
    except OSError as er:
        print("Folders already exist.")

    cam_folder = f'mpi/depth/training/camdata_left/{s}/'
 
    train_data = generate_transforms(cam_folder,data_folder+"/transforms_train.json", train_indices)
    val_data = generate_transforms(cam_folder,data_folder+"/transforms_val.json", validation_indices, train=False)

    dpt_folder = f'mpi/depth/training/depth/{s}/'
    generate_npy(dpt_folder,data_folder)


        