import matplotlib.pyplot as plt
import open3d as o3d
import sys

img_path = sys.argv[1]
depth_path = sys.argv[2]
K_path = sys.argv[3]
pcd_path = sys.argv[4]

depth = plt.imread(depth_path)
depth = o3d.geometry.Image(depth)

image = plt.imread(img_path)
image = o3d.geometry.Image(image)

intrin_dict = {"fx": 0, "fy": 0, "cx": 0, "cy": 0}

with open(K_path) as f:
    intrinsics = f.readline()
    vals = intrinsics.split(" ")
    vals[-1] = vals[-1][:-1]
    for n, k in enumerate(intrin_dict.keys()):
        intrin_dict[k] = vals[n]

fx = intrin_dict["fx"]
fy = intrin_dict["fy"]
cx = intrin_dict["cx"]
cy = intrin_dict["cy"]


params = o3d.camera.PinholeCameraIntrinsic()
params.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image, depth, depth_scale=1, depth_trunc=500, convert_rgb_to_intensity=True)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, params)

o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud(pcd_path, pcd)

