import numpy as np
import os
from PIL import Image
import cv2
import json
import open3d as o3d
import casadi as ca
from scipy.spatial.transform import Rotation as R_scipy
import sys

from utils_IL.perception_utils import convert_to_serializable
from utils_IL.geometry_utils import compute_point_cloud, calculate_rotation_matrix_z
from utils_IL.params import CAM_MATRIX

def optimize_trajectory(reference_trajectory, static_pose, contact_pose=None, contact_idx=None, n_points=100, flag=2):
       
    opti = ca.Opti()

    print(f"Solving with angular velocity constraint {flag}......\n")

    static_pos, static_ori = static_pose
    contact_pos, contact_ori = contact_pose
    
    positions = opti.variable(n_points, 3)
    euler_angles = opti.variable(n_points, 3)

    cost = 0
    for t in range(n_points):
        ref_pos, ref_euler = reference_trajectory[t]
        
        step_cost = ca.sumsqr(positions[t, :] - np.array(ref_pos).reshape((1, 3))) + \
                ca.sumsqr(ca.fmod(euler_angles[t, :] - np.array(ref_euler).reshape((1, 3)) + np.pi, np.pi*2) - np.pi)
            
        if t/n_points > 0.4:
            cost += step_cost
            # velocity constraints
            if t < n_points - 1:
                opti.subject_to(ca.sumsqr(positions[t+1, :] - positions[t, :]) <= 0.01**2)  # translation velocity

                if flag == 2:
                    opti.subject_to(ca.sumsqr(ca.fmod(euler_angles[t+1, :] - euler_angles[t, :] + np.pi, np.pi*2) - np.pi) <= (np.pi/72)**2)
                elif flag == 1:
                    opti.subject_to(ca.sumsqr(ca.fmod(euler_angles[t+1, :] - euler_angles[t, :] + np.pi, np.pi*2) - np.pi) <= (np.pi/60)**2)
                # far from contact, free to move
                else:
                    opti.subject_to(ca.sumsqr(ca.fmod(euler_angles[t+1, :] - euler_angles[t, :] + np.pi, np.pi*2) - np.pi) <= (np.pi/36)**2)
        else:
            opti.subject_to(ca.sumsqr(positions[t+1, :] - positions[t, :]) <= 0.01**2)  # translation velocity
            opti.subject_to(ca.sumsqr(ca.fmod(euler_angles[t+1, :] - euler_angles[t, :] + np.pi, np.pi*2) - np.pi) <= (np.pi / 36000)**2)

    # inital keyframe constraint
    opti.subject_to(positions[0, :] == np.array(static_pos).reshape((1, 3)))
    opti.subject_to(euler_angles[0, :] == np.array(static_ori).reshape((1, 3)))
    
    # function keyframe constraint
    opti.subject_to(positions[contact_idx, :] == np.array(contact_pos).reshape((1, 3)))
    opti.subject_to(euler_angles[contact_idx, :] == np.array(contact_ori).reshape((1, 3)))

    opti.minimize(cost)
    opti.solver('ipopt')
    try: 
        sol = opti.solve()
        optimized_trajectory = [(sol.value(positions[t, :]),  sol.value(euler_angles[t, :])) for t in range(n_points)]
        return optimized_trajectory, -2
    except RuntimeError:
        flag -= 1
        return None, flag

def compute_function_plane_frame(func_list, center_list, grasp_list):
    """
    A: function point
    B: center point
    C: grasp point 
    """
    assert len(func_list) == len(center_list) == len(grasp_list), "The lengths of three functional point lists do not match!"

    pose_list = []
    # spheres = []
    for i in range(len(func_list)):

        A = np.array(func_list[i])  # function point
        B = np.array(center_list[i])  # center point
        C = np.array(grasp_list[i])  # grasp point

        # Step 1: Compute the x-axis (A-B)
        AB = B - A
        x_axis = AB / np.linalg.norm(AB)  # Normalize the vector

        # compute the pointing direction
        real_x_axis = np.array([1, 0, 0])
        dot_product = np.dot(x_axis, real_x_axis)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)

        # Step 2: Compute the normal vector to the plane (z-axis)
        AC = C - A
        z_axis = np.cross(AB, AC)
        z_axis = z_axis / np.linalg.norm(z_axis)  # Normalize the vector

        # Step 3: Compute the y-axis using cross product of z-axis and x-axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize the vector

        # Step 4: compute rotation
        R = np.column_stack((x_axis, y_axis, z_axis))
        euler_angles = R_scipy.from_matrix(R).as_euler('xyz', degrees=False)

        # Step 6: compute pose
        pose = np.hstack((A, euler_angles)).tolist()
        pose_list.append((np.array(pose[:3], ), np.array(pose[3:], )))

    return pose_list

def pc_viz(data_path, trans_flag=True):

    static_frame_idx = 0
    demo_static_frame_path = os.path.join(data_path, 'rgb', f'{static_frame_idx:05}.jpg')
    demo_static_depth_frame_path = os.path.join(data_path, 'depth', f'{static_frame_idx:05}.png')

    demo_static_frame = cv2.imread(demo_static_frame_path)
    demo_static_frame = cv2.cvtColor(demo_static_frame, cv2.COLOR_BGR2RGB)
    # demo_static_frame_pil = Image.fromarray(demo_static_frame)
    demo_static_frame_depth_pil = Image.open(demo_static_depth_frame_path)
    demo_static_frame_depth = np.array(demo_static_frame_depth_pil)

    cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')

    with open(cam_to_target_trans_path, 'r') as file:
        cam_to_target_trans_dict = json.load(file)
    rotation_matrix = np.array(cam_to_target_trans_dict['rotation_matrix'])
    rotation_center = np.array(cam_to_target_trans_dict['rotation_center'])
    centroid = np.array(cam_to_target_trans_dict['centroid'])

    # get the test tool point cloud
    tool_mask_path = os.path.join(data_path, 'detection_output', 'tool_mask.npy')
    tool_mask = np.load(tool_mask_path)
    tool_mask_uint8 = tool_mask.astype(np.uint8)
    masked_demo_static_frame = cv2.bitwise_and(demo_static_frame, demo_static_frame, mask=tool_mask_uint8)
    masked_demo_static_frame_depth = cv2.bitwise_and(demo_static_frame_depth, demo_static_frame_depth, mask=tool_mask_uint8)
    points, colors, points_3d = compute_point_cloud(masked_demo_static_frame, masked_demo_static_frame_depth, CAM_MATRIX, [])
    tool_point_cloud = o3d.geometry.PointCloud()
    tool_point_cloud.points = o3d.utility.Vector3dVector(points)
    tool_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    cl, ind = tool_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    tool_point_cloud = tool_point_cloud.select_by_index(ind)

    if trans_flag:
        tool_point_cloud = tool_point_cloud.rotate(rotation_matrix, center=rotation_center)
        tool_point_cloud.translate(-centroid)
    points, colors, points_3d = compute_point_cloud(demo_static_frame, demo_static_frame_depth, CAM_MATRIX, [])
    scene_point_cloud = o3d.geometry.PointCloud()
    scene_point_cloud.points = o3d.utility.Vector3dVector(points)
    scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    if trans_flag:
        scene_point_cloud = scene_point_cloud.rotate(rotation_matrix, center=rotation_center)
        scene_point_cloud.translate(-centroid)

    return scene_point_cloud, tool_point_cloud

def tool_trajectory_transfer(data_path, test_data_path):

    """
    test tool trajectory generation based on established functional correspondence
    """

    output_path = os.path.join(test_data_path, 'tool_traj_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # load function keyframe idx
    keyframe_idx_path = os.path.join(data_path, 'key_point_track_output', 'keyframe_idx.json')
    with open(keyframe_idx_path, 'r') as file:
        keyframe_idx_dict = json.load(file)
    contact_idx = keyframe_idx_dict['func']

    # load demo tool functional points trajectory
    demo_tool_track_dict_path = os.path.join(data_path, 'tool_pose_track_output', 'tool_track.json')
    with open(demo_tool_track_dict_path, 'r') as file:
        demo_tool_track_dict = json.load(file)
    
    # load function keyframe test tool functional points
    test_tool_track_dict_path = os.path.join(test_data_path, 'tool_pose_transfer_output', 'test_tool_track.json')
    with open(test_tool_track_dict_path, 'r') as file:
        test_tool_track_dict = json.load(file)
    
    # transform functional keypoint trajectory to pose trajectory
    demo_func_point_track = demo_tool_track_dict['function']
    demo_center_point_track = demo_tool_track_dict['center']
    demo_grasp_point_track = demo_tool_track_dict['grasp']
    demo_tool_pose_list = compute_function_plane_frame(demo_func_point_track, demo_center_point_track, demo_grasp_point_track)

    # rotate demo tool trajectory to align with test tool
    R_z = calculate_rotation_matrix_z(
        test_tool_track_dict['function_init'],
        demo_func_point_track[0])
    rotated_demo_tool_pose_list = []
    for demo_tool_pose in demo_tool_pose_list:
        demo_translation, demo_rotation = demo_tool_pose

        demo_rotation_matrix = R_scipy.from_euler('xyz', demo_rotation).as_matrix()
        rotated_demo_rotation_matrix = R_z @ demo_rotation_matrix
        rotated_demo_rotation = R_scipy.from_matrix(rotated_demo_rotation_matrix).as_euler('xyz')

        rotated_demo_translation = R_z @ demo_translation
        rotated_demo_tool_pose_list.append((rotated_demo_translation, rotated_demo_rotation))
    
    # test tool trajectory initial and function keyframe pose constraints    
    test_tool_pose_list = compute_function_plane_frame([test_tool_track_dict['function_init'], test_tool_track_dict['function_func']], 
                                                                            [test_tool_track_dict['center_init'], test_tool_track_dict['center_func']], 
                                                                            [test_tool_track_dict['grasp_init'], test_tool_track_dict['grasp_func']])

    # test tool trajectory optimization
    # TODO: change to SO(3) optimization
    constraint_flag = 2
    while constraint_flag != -2:
        test_tool_trjectory, constraint_flag = optimize_trajectory(reference_trajectory=rotated_demo_tool_pose_list, 
                                                static_pose=test_tool_pose_list[0], 
                                                contact_pose=test_tool_pose_list[1],
                                                contact_idx=contact_idx,
                                                n_points=len(demo_tool_pose_list),
                                                flag=constraint_flag)
        if constraint_flag == -1:
            print("Solver failed with all constraints. Shutting down...")
            sys.exit("Solver failure: Could not optimize trajectory.")
    print("Test tool trajecory optimization completed.")

    # save test tool trajectory
    with open(os.path.join(output_path, 'test_tool_trajectory.json'), "w") as json_file:
        json.dump(test_tool_trjectory, json_file, default=convert_to_serializable, indent=4)

    # test tool trajectory visualization
    test_scene_pc, test_tool_pc = pc_viz(test_data_path)
    test_tool_trajectory_pc = test_tool_pc + test_scene_pc
    for i in range(1, len(test_tool_trjectory)):

        pose_1 = test_tool_trjectory[i-1]
        pose_2 = test_tool_trjectory[i]

        # Extract positions and orientations
        pos_1, rot_1 = pose_1
        pos_2, rot_2 = pose_2

        # Compute relative translation
        relative_translation = pos_2 - pos_1

        # Compute relative rotation
        rot_matrix_1 = R_scipy.from_euler('xyz', rot_1).as_matrix()
        rot_matrix_2 = R_scipy.from_euler('xyz', rot_2).as_matrix()
        relative_rotation_matrix = np.dot(rot_matrix_2, np.linalg.inv(rot_matrix_1))

        # transform the tool point cloud
        test_tool_pc.rotate(relative_rotation_matrix, center=pos_1)
        test_tool_pc.translate(relative_translation)

        # add the test tool pc to the scene
        if i%10 == 0:
            test_tool_trajectory_pc += test_tool_pc
        
    o3d.io.write_point_cloud(os.path.join(output_path, 'test_tool_traj_pc.ply'), test_tool_trajectory_pc)

    return None








