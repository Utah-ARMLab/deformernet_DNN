import open3d
import os
import numpy as np
import pickle
import timeit
import torch

import sys

sys.path.append("../")
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors

ROBOT_Z_OFFSET = 0.25


import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument(
    "--obj_category", default="None", type=str, help="object category. Ex: boxes_10kPa"
)
args = parser.parse_args()

data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{args.obj_category}/processed_data"
data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{args.obj_category}/processed_data_w_mp"
# data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_cylinder_10kPa/single_thin_cylinder_10kPa_processed_data"
# data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_cylinder_10kPa/single_thin_cylinder_10kPa_processed_data_w_mp"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer()

start_index = 0
max_len_data = 16000

with torch.no_grad():
    for i in range(start_index, max_len_data):
        # if 11161 <= i <= 11168:
        #     continue

        if i % 50 == 0:
            print(
                "current count:",
                i,
                " , time passed:",
                timeit.default_timer() - start_time,
            )

        file_name = os.path.join(
            data_recording_path, "processed sample " + str(i) + ".pickle"
        )

        if not os.path.isfile(file_name):
            print(f"{file_name} not found")
            continue

        with open(file_name, "rb") as handle:
            data = pickle.load(handle)

        pc = data["partial pcs"][0]
        pc_goal = data["partial pcs"][1]
        # mani_point = np.array(data["mani_point"])
        mani_point = data["mani_point"]
        mp_pos = np.array(
            [-mani_point[0, 3], -mani_point[1, 3], mani_point[2, 3] + ROBOT_Z_OFFSET]
        )

        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(pc.transpose(1, 0))
        # _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
        _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
        mp_channel = np.zeros(pc.shape[1])
        mp_channel[nearest_idxs.flatten()] = 1
        modified_pc = np.vstack(
            [pc, mp_channel]
        )  # pc with 4th channel with value of 1 if near the MP point and 0 elsewhere

        # # print(modified_pc[:4,100])
        # # assert modified_pc.shape == (4,1024)
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(np.array(pc.transpose(1,0)))
        # colors = np.zeros((1024,3))
        # colors[nearest_idxs.flatten()] = [1,0,0]
        # pcd.colors =  open3d.utility.Vector3dVector(colors)
        # open3d.visualization.draw_geometries([pcd])

        pcs = (modified_pc, pc_goal)
        processed_data = {
            "partial pcs": pcs,
            "full pcs": data["full pcs"],
            "pos": data["pos"],
            "rot": data["rot"],
            "twist": data["twist"],
            "mani_point": mani_point,
            "obj_name": data["obj_name"],
        }

        with open(
            os.path.join(data_processed_path, "mp sample " + str(i) + ".pickle"), "wb"
        ) as handle:
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('/home/baothach/shape_servo_data/batch_3b(using_camera)', 'rb') as handle:
#     data1 = pickle.load(handle)


# final_point_clouds = data1["point clouds"]
# final_desired_positions = data1["positions"]
# count = 0
# modified_pcs = []
# for (pc, pc_goal) in final_point_clouds:
#     count += 1
#     print(count)
#     # if count == 2:
#     #     break

#     pcd = open3d.geometry.PointCloud()

#     pcd.points = open3d.utility.Vector3dVector(np.array(pc))
#     pcd.estimate_normals(
#         search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1,
#                                                             max_nn=10))

#     radii = [0.005, 0.01, 0.02, 0.04]
#     # radii =  [0.1, 0.2, 0.4, 0.8]
#     rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd, open3d.utility.DoubleVector(radii))

#     rec_mesh.compute_vertex_normals()
#     pcd = rec_mesh.sample_points_uniformly(number_of_points=3000)

#     pcd_2 = open3d.geometry.PointCloud()

#     pcd_2.points = open3d.utility.Vector3dVector(np.array(pc_goal))
#     pcd_2.estimate_normals(
#         search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1,
#                                                             max_nn=10))

#     radii = [0.005, 0.01, 0.02, 0.04]
#     # radii =  [0.1, 0.2, 0.4, 0.8]
#     rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd_2, open3d.utility.DoubleVector(radii))

#     rec_mesh.compute_vertex_normals()
#     pcd_2 = rec_mesh.sample_points_uniformly(number_of_points=3000)
#     modified_pcs.append((np.asarray(pcd.points), np.asarray(pcd_2.points)))


# data = {"point clouds": modified_pcs, "positions": final_desired_positions}
# with open('/home/baothach/shape_servo_data/batch_3b(using_camera)_modified', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
