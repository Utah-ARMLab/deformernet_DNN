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


data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/resampled_mp_classifer_data"
data_processed_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/processed_mp_classifer_data"

start_time = timeit.default_timer()
all_chamfers = []
all_gt_chamfers = []


start_index = 0
file_names = sorted(os.listdir(data_recording_path))
max_len_data = len(file_names)

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    file_name = os.path.join(data_recording_path, file_names[i])

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)

    all_chamfers.append(data["chamfer"])
    all_gt_chamfers.append(data["gt chamfer"])

# positive_idxs = list(set(np.where(np.array(all_chamfers) <= 1.2*np.array(all_gt_chamfers))[0]) \
#                     | set(np.where(np.array(all_chamfers) <= 0.25)[0]))    # union of two list of indices

positive_idxs = np.where(np.array(all_chamfers) <= 1.2 * np.array(all_gt_chamfers))[0]

for i in range(start_index, max_len_data):
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    file_name = os.path.join(data_recording_path, file_names[i])

    label = 1 if i in positive_idxs else 0

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)
        pc = data["partial pcs"][0]
        pc_goal = data["partial pcs"][1]
        mp_pos = np.array(list(data["mani_point"][0]))

    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc.transpose(1, 0))
    _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc.shape[1])
    mp_channel[nearest_idxs.flatten()] = 1
    modified_pc = np.vstack(
        [pc, mp_channel]
    )  # pc with 4th channel with value of 1 if near the MP point and 0 elsewhere

    # assert modified_pc.shape == (4,1024)
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(pc.transpose(1,0)))
    # colors = np.zeros((1024,3))
    # colors[nearest_idxs.flatten()] = [1,0,0]
    # pcd.colors =  open3d.utility.Vector3dVector(colors)
    # mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # mani_point.paint_uniform_color([0,0,1])
    # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))])

    pcs = (modified_pc, pc_goal)

    processed_data = {
        "partial pcs": pcs,
        "label": label,
        "mani_point": data["mani_point"],
        "chamfer": data["chamfer"],
        "gt mani_point": data["gt mani_point"],
        "gt chamfer": data["gt chamfer"],
        "obj_name": data["obj_name"],
    }

    with open(
        os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"),
        "wb",
    ) as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(f"num possitive: {len(positive_idxs)}/{len(file_names)}")

for threshold in [0.2, 0.25, 0.3, 0.4, 0.5]:
    print(
        f"Chamfer < {threshold}m count:",
        sum(1 for chamf in all_chamfers if chamf <= threshold),
    )

print("===========")

for threshold in [0.8, 1.0]:
    print(
        f"Chamfer > {threshold}m count:",
        sum(1 for chamf in all_chamfers if chamf >= threshold),
    )
