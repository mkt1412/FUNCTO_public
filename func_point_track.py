import os
import torch
import json
import numpy as np
import cv2
import open3d as o3d

from utils_IL.perception_utils import read_frames_from_path, backproject_3d_2d
from utils_IL.geometry_utils import compute_point_cloud
from utils_IL.params import CAM_MATRIX

def func_point_track(data_path):

    output_path = os.path.join(data_path, 'func_point_track_output')
    os.makedirs(output_path, exist_ok=True)

    # load pre-function keyframe idx
    keyframe_idx_path = os.path.join(data_path, 'key_point_track_output', 'keyframe_idx.json')
    with open(keyframe_idx_path, 'r') as file:
        keyframe_idx_dict = json.load(file)
    pre_func_frame_idx = keyframe_idx_dict['pre-func']

    # load demo video
    video_path_rgb = os.path.join(data_path, 'rgb')
    video_path_depth = os.path.join(data_path, 'depth')
    video, depth_stream = read_frames_from_path(video_path_rgb, video_path_depth)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    init_rgb_frame = video[0, 0].permute(1, 2, 0).cpu().numpy() # (720, 1280, 3)
    intial_depth_frame = depth_stream[0]  # (720, 1280)

    # transform scene point cloud
    points, colors, points_3d = compute_point_cloud(init_rgb_frame, intial_depth_frame, CAM_MATRIX, [])
    scene_point_cloud = o3d.geometry.PointCloud()
    scene_point_cloud.points = o3d.utility.Vector3dVector(points)
    scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
    with open(cam_to_target_trans_path, 'r') as file:
        cam_to_target_trans_dict = json.load(file)
    rotation_matrix = np.array(cam_to_target_trans_dict['rotation_matrix'])
    rotation_center = np.array(cam_to_target_trans_dict['rotation_center'])
    centroid = np.array(cam_to_target_trans_dict['centroid'])
    rotated_scene_point_cloud = scene_point_cloud.rotate(rotation_matrix, center=rotation_center)
    rotated_scene_point_cloud.translate(-centroid)

    # load 3d function point
    func_point_3d_path = os.path.join(data_path, 'func_point_det_output', 'func_point_3d_out.json')
    with open(func_point_3d_path, 'r') as file:
        func_point_3d_dict = json.load(file)
    func_point_3d = func_point_3d_dict['function']
   
    ############ 3D function point tracking ############
    # load the pre-computed frame transformation
    frame_transformations_path = os.path.join(data_path, 'solve_rt_output', 'frame_transformations.json')
    with open(frame_transformations_path, 'r') as json_file:
        frame_transformations = json.load(json_file)

    # compute function point in the  keyframe
    T_end_to_init = frame_transformations[str(pre_func_frame_idx)]
    T_init_to_end = np.linalg.inv(T_end_to_init)
    func_point_3d_h = np.append(func_point_3d, 1) 
    first_point_3d_h = np.dot(T_init_to_end, func_point_3d_h)
    first_point_3d = first_point_3d_h[:3]

    # backproject function point from 3d to 2d
    [u, v] = backproject_3d_2d(first_point_3d, rotation_matrix, rotation_center, centroid, CAM_MATRIX)
    print(f"2D function point in the initial keyframe computed at {[u, v]}")
    cv2.circle(init_rgb_frame, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(os.path.join(output_path, 'func_point_2d_init.jpg'), init_rgb_frame)
    func_point_2d_path = os.path.join(output_path, 'func_point_2d_init.json')
    with open(func_point_2d_path, 'w') as json_file:
        json.dump([u, v], json_file)
    
    # compute the full 3d function point trajectory
    func_point_traj_3d = [first_point_3d.tolist()]

    # each transformation is wrt the initial keyframe
    function_point_3d_hom = first_point_3d_h
    for frame_idx in range(1, video.shape[1]):
        cur_trans = np.array(frame_transformations[str(frame_idx)])
        transformed_function_point_hom = cur_trans @ function_point_3d_hom
        transformed_function_point = transformed_function_point_hom[:3]
        func_point_traj_3d.append(transformed_function_point.tolist())

    ############ 3D function point tracking visualization ############
    spheres = []
    for point in func_point_traj_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 0, 0.5]) 
        spheres.append(sphere)
    # Combine the transformed target object point cloud and the spheres
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    combined_point_cloud = rotated_scene_point_cloud + spheres_combined
    o3d.io.write_point_cloud(os.path.join(output_path, '3d_scene_w_function_point_trajectory.ply'), combined_point_cloud)

    func_point_traj_3d_full_path = os.path.join(output_path, 'func_point_traj_3d_full.json')
    with open(func_point_traj_3d_full_path, 'w') as json_file:
        json.dump(func_point_traj_3d, json_file, indent=4)
    print(f"3D function point trajectory saved to {func_point_traj_3d_full_path}")

    return None




   
    

