import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d


def find_knn(pc, mp_pos, num_nn):
    neigh = NearestNeighbors(n_neighbors=num_nn)
    neigh.fit(pc.transpose(1, 0))
    _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc.shape[1])
    mp_channel[nearest_idxs.flatten()] = 1

    return mp_channel


def vis_mp(vis_pc, vis_pc_goal, vis_mp_pos, gt_mp):
    # print(vis_pc.shape, vis_pc_goal.shape)
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
