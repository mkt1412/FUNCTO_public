import joblib
import numpy as np
import cv2
import json
import os
from PIL import Image
import open3d as o3d
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from scipy.stats import zscore
import ruptures as rpt

from cloud_services.apis.owlv2 import OWLViT
from cloud_services.apis.sam import SAM, visualize_image

from utils_IL.perception_utils import fit_gaussian, non_maximum_suppression, convert_to_serializable
from utils_IL.geometry_utils import align_set_B_to_A_v2, compute_point_cloud, compute_3d_points
from utils_IL.params import CAM_MATRIX


def tool_center_detect(data_path):

    output_path = os.path.join(data_path, 'tool_pose_track_output')
    os.makedirs(output_path, exist_ok=True)

    init_frame_idx = 0
    init_frame_path = os.path.join(data_path, 'rgb', f'{init_frame_idx:05}.jpg')
    init_depth_frame_path = os.path.join(data_path, 'depth', f'{init_frame_idx:05}.png')

    init_frame = cv2.imread(init_frame_path)
    init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB)
    init_frame_depth_pil = Image.open(init_depth_frame_path)
    init_frame_depth = np.array(init_frame_depth_pil)

    tool_mask = np.load(os.path.join(data_path, 'detection_output', 'tool_mask.npy'))

    ############ Get tool center point from the static frame ############
    cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
    with open(cam_to_target_trans_path, 'r') as file:
        cam_to_target_trans_dict = json.load(file)
    rotation_matrix = np.array(cam_to_target_trans_dict['rotation_matrix'])
    rotation_center = np.array(cam_to_target_trans_dict['rotation_center'])
    centroid = np.array(cam_to_target_trans_dict['centroid'])

    # get the tool point cloud
    # mask = (demo_static_frame_depth == 0).astype(np.uint8)
    # demo_static_frame_depth = demo_static_frame_depth.astype(np.float32)
    # demo_static_frame_depth_inpainted = cv2.inpaint(demo_static_frame_depth, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    tool_mask_uint8 = tool_mask.astype(np.uint8)
    masked_init_frame = cv2.bitwise_and(init_frame, init_frame, mask=tool_mask_uint8)
    masked_init_frame_depth = cv2.bitwise_and(init_frame_depth, init_frame_depth, mask=tool_mask_uint8)
    points, colors, points_3d = compute_point_cloud(masked_init_frame, masked_init_frame_depth, CAM_MATRIX, [])
    tool_point_cloud = o3d.geometry.PointCloud()
    tool_point_cloud.points = o3d.utility.Vector3dVector(points)
    tool_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # remove outliers
    cl, ind = tool_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    filtered_tool_point_cloud = tool_point_cloud.select_by_index(ind)
    rotated_tool_point_cloud = filtered_tool_point_cloud.rotate(rotation_matrix, center=rotation_center)
    rotated_tool_point_cloud.translate(-centroid)
           
    # get the bounding box
    aabb = rotated_tool_point_cloud.get_axis_aligned_bounding_box()
    box_points = np.asarray(aabb.get_box_points())
    center_point = np.mean(box_points, axis=0)

    # save tool center and aabb corners 
    box_points_with_center = {}
    box_points_with_center['center'] = center_point.tolist()
    box_points_with_center['aabb'] = box_points.tolist()
    with open(os.path.join(output_path, 'tool_aabb_center_point_3d.json'), 'w') as json_file:
        json.dump(box_points_with_center, json_file, indent=4)
    print("demo tool 3d center point and bounding box computed.")

    ########### Demo tool center point visualization ############
    spheres = []
    for point in box_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0, 1, 0])  # Green color
        spheres.append(sphere)
    for point in [center_point]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.647, 0.165, 0.165])  # Red color
        spheres.append(sphere)

    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    
    combined_point_cloud = rotated_tool_point_cloud + spheres_combined
    o3d.io.write_point_cloud(os.path.join(output_path, 'tool_with_aabb_corners.ply'), combined_point_cloud)

def tool_pose_track(data_path):

    output_path = os.path.join(data_path, 'tool_pose_track_output')
    os.makedirs(output_path, exist_ok=True)

    # load the pre-computed frame transformation
    frame_transformations_path = os.path.join(data_path, 'solve_rt_output', 'frame_transformations.json')
    with open(frame_transformations_path, 'r') as json_file:
        frame_transformations = json.load(json_file)

    # load function point trajectory
    func_point_traj_3d_full_path = os.path.join(data_path, 'func_point_track_output', 'func_point_traj_3d_full.json')   
    with open(func_point_traj_3d_full_path, 'r') as json_file:
        func_point_traj_3d_full = json.load(json_file)
    func_point_trajectory = np.array(func_point_traj_3d_full)

    ############ Function frame discovery ############
    # load pre-function frame
    keyframe_dict_path = os.path.join(data_path, 'key_point_track_output', 'keyframe_idx.json')   
    with open(keyframe_dict_path, 'r') as json_file:
        keyframe_dict = json.load(json_file)
    
    pre_func_idx = keyframe_dict['pre-func']
    kf_dis_keypoints = func_point_trajectory[pre_func_idx:] 
    time_interval = 0.001 
    displacements = np.diff(kf_dis_keypoints, axis=0)
    velocities = displacements / time_interval
    algo = rpt.Dynp(model="l2").fit(velocities)
    result = algo.predict(n_bkps=2)  # [15, 30, 49]

    func_idx = pre_func_idx + result[-1] - 2
    keyframe_dict['func'] = func_idx
    print(f"detected function keyframe: {func_idx}")
    # save keyframe dict
    with open(keyframe_dict_path, 'w') as f:
        json.dump(keyframe_dict, f)

    ############ 3d center and grasp points trajectory tracking ############

    center_point_3d_path = os.path.join(data_path, 'tool_pose_track_output', 'tool_aabb_center_point_3d.json')
    with open(center_point_3d_path, 'r') as json_file:
        tool_aabb_center_points_3d = json.load(json_file)
    center_point_3d = tool_aabb_center_points_3d['center']

    grasp_point_3d_path = os.path.join(data_path, 'grasp_det_output', 'grasp_point_3d.json')
    with open(grasp_point_3d_path, 'r') as json_file:
        grasp_point_3d = json.load(json_file)
    
    func_point_3d = func_point_trajectory[0]

    # hack: center point refinement
    print("enable center point projection.")
    direction_xy = np.array(grasp_point_3d)[:2] - func_point_3d[:2]
    direction_xy_unit = direction_xy / np.linalg.norm(direction_xy)
    center_vector_xy = np.array(center_point_3d)[:2] - func_point_3d[:2]
    projection_length = np.dot(center_vector_xy, direction_xy_unit)
    refined_xy = func_point_3d[:2] + projection_length * direction_xy_unit
    center_point_3d = np.array([refined_xy[0], refined_xy[1], center_point_3d[2]])
    
    # center point trajectory
    center_point_trajectory = [center_point_3d]
    center_point_3d_hom = np.append(np.array(center_point_3d), 1)
    for frame_idx in frame_transformations.keys():
        cur_trans = np.array(frame_transformations[frame_idx])
        transformed_center_point_hom = cur_trans @ center_point_3d_hom
        transformed_center_point = transformed_center_point_hom[:3]
        center_point_trajectory.append(transformed_center_point)
    
    # grasp point trajectory
    grasp_point_trajectory = [grasp_point_3d]
    grasp_point_3d_hom = np.append(grasp_point_3d, 1)
    for frame_idx in frame_transformations.keys():
        cur_trans = np.array(frame_transformations[frame_idx])
        transformed_grasp_point_hom = cur_trans @ grasp_point_3d_hom
        transformed_grasp_point = transformed_grasp_point_hom[:3]
        grasp_point_trajectory.append(transformed_grasp_point)
   
    ############ Demo tool trajectory saving and visualization ############
    tool_track_dict = {'function': func_point_trajectory.tolist(), 'center': center_point_trajectory, 'grasp': grasp_point_trajectory}

    with open(os.path.join(output_path, 'tool_track.json'), "w") as json_file:
        json.dump(tool_track_dict, json_file, default=convert_to_serializable, indent=4)
    print("demo tool functional keypoint trajectory computed.")
    
    spheres = []
    for point in func_point_trajectory:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 0, 0.5])  
        spheres.append(sphere)
    for point in center_point_trajectory:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.647, 0.165, 0.165])  # Red color
        spheres.append(sphere)
    for point in grasp_point_trajectory:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0, 0, 1])  # Blue color
        spheres.append(sphere)
    
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        # Convert the sphere to a point cloud for easier combination
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    o3d.io.write_point_cloud(os.path.join(output_path, 'func_center_grasp_traj.ply'), spheres_combined)

    # print(func_point_trajectory[0], center_point_trajectory[0], grasp_point_trajectory[0])

    return None




