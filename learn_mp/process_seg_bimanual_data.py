import open3d
import os
import numpy as np
import pickle
import timeit
from farthest_point_sampling import *
from sklearn.neighbors import NearestNeighbors


def down_sampling(pc):
    farthest_indices, _ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]
    return pc


data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/bimanual/multi_boxes_1000Pa/data"
data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/bimanual/multi_boxes_1000Pa/processed_seg_data"

start_time = timeit.default_timer()
all_chamfers = []


start_index = 0
max_len_data = 20002
file_names = sorted(os.listdir(data_recording_path))

# with open(os.path.join(data_recording_path, file_names[343]), 'rb') as handle:
#     data_2 = pickle.load(handle)

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    file_name = os.path.join(data_recording_path, file_names[i])

    if not os.path.isfile(file_name):
        print(f"{file_names[i]} not found")
        continue

    if os.path.isfile(os.path.join(data_processed_path, file_names[i])):
        print(f"{file_names[i]} already existed")
        continue

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)

    pc_resampled = down_sampling(data["partial pcs"][0])
    pc_goal_resampled = down_sampling(data["partial pcs"][1])

    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc_resampled)
    mp_channel = np.zeros(pc_resampled.shape[0])

    _, nearest_idxs_1 = neigh.kneighbors(
        np.array(list(data["mani_point_1"][0])).reshape(1, -1)
    )
    mp_channel[nearest_idxs_1.flatten()] = 1

    _, nearest_idxs_2 = neigh.kneighbors(
        np.array(list(data["mani_point_2"][0])).reshape(1, -1)
    )
    # mp_channel_2 = np.zeros(pc_resampled.shape[0])
    mp_channel[nearest_idxs_2.flatten()] = 2

    # print(data["mani_point_1"][0])
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(pc_resampled)
    # pcd_goal = open3d.geometry.PointCloud()
    # pcd_goal.points = open3d.utility.Vector3dVector(pc_goal_resampled)
    # colors = np.zeros((1024,3))
    # colors[nearest_idxs_1.flatten()] = [1,0,0]
    # colors[nearest_idxs_2.flatten()] = [0,1,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)
    # mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point_1_sphere.paint_uniform_color([0,0,1])
    # mani_point_1_sphere.translate(tuple(data["mani_point_1"][0]))
    # mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point_2_sphere.paint_uniform_color([1,0,0])
    # mani_point_2_sphere.translate(tuple(data["mani_point_2"][0]))
    # open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.2,0,0)),\
    #                                     mani_point_1_sphere, mani_point_2_sphere])

    # mp_pos = np.array(list(data["mani_point"][0]))
    # neigh = NearestNeighbors(n_neighbors=50)
    # neigh.fit(pc_resampled)
    # _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    # mp_channel = np.zeros(pc_resampled.shape[0])
    # mp_channel[nearest_idxs.flatten()] = 1

    # print(mp_channel.shape)
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print(mp_channel)

    # # pcd = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/manipulation_points/box/init_box_pc.pcd")
    # # pcd = open3d.geometry.PointCloud()
    # # pcd.points = open3d.utility.Vector3dVector(np.array(down_sampling(data_2["partial pcs"][0])))

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(pc_resampled))
    # colors = np.zeros((1024,3))
    # colors[nearest_idxs.flatten()] = [1,0,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)
    # mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point.paint_uniform_color([0,0,1])
    # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))])

    pcs = (np.transpose(pc_resampled, (1, 0)), np.transpose(pc_goal_resampled, (1, 0)))
    processed_data = {
        "partial pcs": pcs,
        "mp_labels": mp_channel,
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
