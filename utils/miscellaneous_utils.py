# import open3d
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
from .point_cloud_utils import pcd_ize


def normalize_list(lst):
    minimum = min(lst)
    maximum = max(lst)
    value_range = maximum - minimum

    normalized_lst = [(value - minimum) / value_range for value in lst]

    return normalized_lst


def scalar_to_rgb(scalar_list, colormap="jet", min_val=None, max_val=None):
    if min_val is None:
        norm = plt.Normalize(vmin=np.min(scalar_list), vmax=np.max(scalar_list))
    else:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.cm.get_cmap(colormap)
    rgb = cmap(norm(scalar_list))
    return rgb


def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)


def read_pickle_data(data_path):
    with open(data_path, "rb") as handle:
        return pickle.load(handle)


def write_pickle_data(data, data_path, protocol=3):
    with open(data_path, "wb") as handle:
        pickle.dump(data, handle, protocol=protocol)


def print_lists_with_formatting(lists, decimals, prefix_str):
    print(prefix_str, end=" ")  # Print the prefix string followed by a space
    for lst in lists:
        print("[", end="")
        # Check if the iterable is not empty by checking its length
        if len(lst) > 0:
            for e in lst[:-1]:
                print(f"{e:.{decimals}f}" if isinstance(e, float) else e, end=", ")
            # Handle the last element to avoid a trailing comma
            print(
                f"{lst[-1]:.{decimals}f}" if isinstance(lst[-1], float) else lst[-1],
                end="] ",
            )
        else:
            print("]", end=" ")

    print("\n")


def find_knn(pc, mp_pos, num_nn):
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=num_nn)
    neigh.fit(pc)
    _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc.shape[0])
    mp_channel[nearest_idxs.flatten()] = 1

    return mp_channel, nearest_idxs.flatten()


def vis_mp(vis_pc, vis_pc_goal, vis_mp_pos, gt_mp):
    import open3d

    vis_mp_channel = find_knn(vis_pc, vis_mp_pos, num_nn=50)
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(vis_pc_goal.transpose(1, 0))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(vis_pc.transpose(1, 0))
    # pcd.colors = open3d.utility.Vector3dVector(np.array([[1,0,0] if t == 0 else [0,0,0] for t in negative_channel]))
    pcd.colors = open3d.utility.Vector3dVector([[t, 0, 0] for t in vis_mp_channel])
    # pcd.paint_uniform_color([0,0,0])

    mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mani_point.paint_uniform_color([0, 1, 0])

    gt_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    gt_mani_point.paint_uniform_color([0, 0, 1])

    open3d.visualization.draw_geometries(
        [
            pcd,
            pcd_goal.translate((0.2, 0, 0)),
            mani_point.translate(tuple(vis_mp_pos)),
            gt_mani_point.translate(tuple(gt_mp)),
        ]
    )
