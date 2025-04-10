import random
import numpy as np
import open3d as o3d
import os
import json
from PIL import Image
import cv2
import math

from utils_IL.perception_utils import convert_to_serializable
from utils_IL.geometry_utils import compute_3d_points, compute_point_cloud, compute_point_cloud_no_add, get_depth_value, align_set_B_to_A_v3, calculate_rotation_matrix_z
from utils_IL.vp_utils import load_prompts, pose_selection
from utils_IL.params import CAM_MATRIX

from openai import OpenAI
client = OpenAI()

def backproject_3d_2d(point_3d, cam_matrix):

    point_2d_homogeneous = np.dot(cam_matrix, point_3d)

    # Normalize to get 2D pixel coordinates
    u = int(point_2d_homogeneous[0] / point_2d_homogeneous[2])
    v = int(point_2d_homogeneous[1] / point_2d_homogeneous[2])

    depth = point_2d_homogeneous[2]

    return (u, v, depth)

def tool_pose_transfer(data_path, test_data_path, test_tool_label, test_target_label, task_label, vp_flag=False):
    """
    Compute function keyframe pose
    """
    output_path = os.path.join(test_data_path, 'tool_pose_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # load prompts
    prompt_path = os.path.join(test_data_path, '..', '..', 'utils_IL', 'prompts')
    prompts = load_prompts(prompt_path)

    ############ Demo 3D functional points ############

    # read function keyframe idx
    keyframe_idx_path = os.path.join(data_path, 'key_point_track_output', 'keyframe_idx.json')
    with open(keyframe_idx_path, 'r') as file:
        keyframe_idx_dict = json.load(file)
    keyframe_idx = keyframe_idx_dict['func']

    # load demo scene transformation info
    demo_cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
    with open(demo_cam_to_target_trans_path, 'r') as file:
        demo_cam_to_target_trans_dict = json.load(file)
    demo_target_dim = demo_cam_to_target_trans_dict['target_dim']

    functional_keypoint_3d_path = os.path.join(data_path, 'tool_pose_track_output', 'tool_track.json')
    with open(functional_keypoint_3d_path, 'r') as json_file:
        functional_keypoint_3d = json.load(json_file)
    demo_tool_func_point_3d = functional_keypoint_3d['function'][keyframe_idx]
    demo_tool_center_point_3d = functional_keypoint_3d['center'][keyframe_idx]
    demo_tool_grasp_point_3d = functional_keypoint_3d['grasp'][keyframe_idx]

    ############ Test 3D functional points ############
    test_init_frame_idx = 0
    test_init_frame_path = os.path.join(test_data_path, 'rgb', f'{test_init_frame_idx:05}.jpg')
    test_init_depth_frame_path = os.path.join(test_data_path, 'depth', f'{test_init_frame_idx:05}.png')   
    test_init_frame_pil = Image.open(test_init_frame_path).convert('RGB')
    test_init_frame = np.array(test_init_frame_pil)
    test_init_depth_frame_pil = Image.open(test_init_depth_frame_path)
    test_init_depth_frame = np.array(test_init_depth_frame_pil)

    # test tool mask
    test_tool_mask_path = os.path.join(test_data_path, 'detection_output', 'tool_mask.npy')
    test_tool_mask = np.load(test_tool_mask_path)
    tool_mask_uint8 = test_tool_mask.astype(np.uint8)
    masked_test_init_frame = cv2.bitwise_and(test_init_frame, test_init_frame, mask=tool_mask_uint8)
    masked_test_init_frame_depth = cv2.bitwise_and(test_init_depth_frame, test_init_depth_frame, mask=tool_mask_uint8)

    # load test scene transformation info
    test_cam_to_target_trans_path = os.path.join(test_data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
    with open(test_cam_to_target_trans_path, 'r') as file:
        test_cam_to_target_trans_dict = json.load(file)
    test_rotation_matrix = np.array(test_cam_to_target_trans_dict['rotation_matrix'])
    test_rotation_center = np.array(test_cam_to_target_trans_dict['rotation_center'])
    test_centroid = np.array(test_cam_to_target_trans_dict['centroid'])
    test_target_dim = test_cam_to_target_trans_dict['target_dim']

    # load test tool function point
    test_tool_func_point_path = os.path.join(test_data_path, 'func_point_transfer_output', 'test_init_frame_func_point_out.json')
    with open(test_tool_func_point_path, 'r') as json_file:
        test_tool_func_point_2d = json.load(json_file)
    x, y = test_tool_func_point_2d
    candidate_points = []
    test_tool_func_point_2d_list = []
    # Generate neighboring points around (x, y)
    for _ in range(400):
        # Generate small random offsets around the original point
        x_offset = x + random.randint(-10, 10)  
        y_offset = y + random.randint(-10, 10)

        # remove points with zero depth value
        depth_value, _ = get_depth_value(masked_test_init_frame_depth, y_offset, x_offset)
        if depth_value == 0:
            continue
        
        # compute distance to the original point
        dist = math.hypot(x_offset - x, y_offset - y)
        candidate_points.append((dist, x_offset, y_offset, depth_value))
    candidate_points.sort(key=lambda item: item[0]) 
    test_tool_func_point_2d_list = [[x_off, y_off, depth] for _, x_off, y_off, depth in candidate_points]
    
    # load test tool grasp point
    test_tool_grasp_point_path = os.path.join(test_data_path, 'grasp_transfer_output', 'test_init_frame_grasp_point_out.json')
    with open(test_tool_grasp_point_path, 'r') as json_file:
        test_tool_grasp_point_2d = json.load(json_file)
    x, y = test_tool_grasp_point_2d
    candidate_points = []
    test_tool_grasp_point_2d_list = []
    for _ in range(100):
        # Generate small random offsets around the original point
        x_offset = x + random.randint(-5, 5)  
        y_offset = y + random.randint(-5, 5) 
        
        # remove points with zero depth value
        depth_value, _ = get_depth_value(masked_test_init_frame_depth, y_offset, x_offset)
        if depth_value == 0:
            continue

        # compute distance to the original point
        dist = math.hypot(x_offset - x, y_offset - y)
        candidate_points.append((dist, x_offset, y_offset, depth_value))
    candidate_points.sort(key=lambda item: item[0]) 
    test_tool_grasp_point_2d_list = [[x_off, y_off, depth] for _, x_off, y_off, depth in candidate_points]
        
    # transform test tool functional keypoints to test target object frame
    points, colors, test_tool_func_point_3d_list = compute_point_cloud_no_add(test_init_frame, test_init_depth_frame, CAM_MATRIX, test_tool_func_point_2d_list)
    test_scene_point_cloud = o3d.geometry.PointCloud()
    test_scene_point_cloud.points = o3d.utility.Vector3dVector(points)
    test_scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    _, _, test_tool_grasp_point_3d_list = compute_3d_points(CAM_MATRIX, test_tool_grasp_point_2d_list)

    # compute plane height for outlier removal
    transformed_test_scene_point_cloud = test_scene_point_cloud.rotate(test_rotation_matrix, center=test_rotation_center)
    transformed_test_scene_point_cloud.translate(-test_centroid)
    plane_model, inliers = transformed_test_scene_point_cloud.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    plane_point_cloud = transformed_test_scene_point_cloud.select_by_index(inliers)
    plane_points = np.asarray(plane_point_cloud.points)
    plane_height_thresh = np.mean(plane_points[:, 2])
    if task_label == 'pour':
        func_height_thresh = plane_height_thresh + 0.06
        grasp_height_thresh = plane_height_thresh + 0.04
    elif task_label == 'scoop':
        func_height_thresh = plane_height_thresh + 0.01
        grasp_height_thresh = plane_height_thresh + 0.01
    elif task_label == 'pound' or task_label == 'sweep':
        func_height_thresh = plane_height_thresh - 0.01
        grasp_height_thresh = plane_height_thresh + 0.03
    elif task_label == 'cut':
        func_height_thresh = plane_height_thresh - 0.01
        grasp_height_thresh = plane_height_thresh + 0.02
    else:
        raise ValueError("Error: Invalid task label.")

    # transform test tool grasp point
    test_tool_grasp_point_3d = None
    for i in range(len(test_tool_grasp_point_3d_list)):
        cur_test_tool_grasp_point_3d = test_tool_grasp_point_3d_list[i]
        cur_test_tool_grasp_point_3d = np.array(cur_test_tool_grasp_point_3d)
        cur_test_tool_grasp_point_3d = np.dot(cur_test_tool_grasp_point_3d - test_rotation_center, test_rotation_matrix.T) + test_rotation_center
        cur_test_tool_grasp_point_3d -= test_centroid
        if cur_test_tool_grasp_point_3d[-1] > grasp_height_thresh:
            test_tool_grasp_point_3d = cur_test_tool_grasp_point_3d
            # also update the 2d grasp point
            print("3D grasp point computed.")
            with open(test_tool_grasp_point_path, "w") as json_file:
                json.dump([int(test_tool_grasp_point_2d_list[i][0]), int(test_tool_grasp_point_2d_list[i][1])], json_file, indent=4)
            break
    if test_tool_grasp_point_3d is None:
        raise ValueError("Error: 3D grasp point not found.")
  
    test_tool_func_point_3d = None
    for i in range(len(test_tool_func_point_3d_list)):
        cur_test_tool_func_point_3d = test_tool_func_point_3d_list[i]
        cur_test_tool_func_point_3d = np.array(cur_test_tool_func_point_3d)
        cur_test_tool_func_point_3d = np.dot(cur_test_tool_func_point_3d - test_rotation_center, test_rotation_matrix.T) + test_rotation_center
        cur_test_tool_func_point_3d -= test_centroid
        if cur_test_tool_func_point_3d[-1] > func_height_thresh:
            test_tool_func_point_3d = cur_test_tool_func_point_3d
            print("3D function point computed.")
            break
    if test_tool_func_point_3d is None:
        raise ValueError("Error: 3D function point not found.")

    # reconstruct test tool and target point cloud
    points, colors, points_3d = compute_point_cloud(masked_test_init_frame, masked_test_init_frame_depth, CAM_MATRIX, [])
    test_tool_point_cloud = o3d.geometry.PointCloud()
    test_tool_point_cloud.points = o3d.utility.Vector3dVector(points)
    test_tool_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    test_target_mask_path = os.path.join(test_data_path, 'detection_output', 'target_mask.npy')
    test_target_mask = np.load(test_target_mask_path)
    target_mask_uint8 = test_target_mask.astype(np.uint8)
    masked_test_init_frame = cv2.bitwise_and(test_init_frame, test_init_frame, mask=target_mask_uint8)
    masked_test_init_frame_depth = cv2.bitwise_and(test_init_depth_frame, test_init_depth_frame, mask=target_mask_uint8)
    points, colors, points_3d = compute_point_cloud(masked_test_init_frame, masked_test_init_frame_depth, CAM_MATRIX, [])
    test_target_point_cloud = o3d.geometry.PointCloud()
    test_target_point_cloud.points = o3d.utility.Vector3dVector(points)
    test_target_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # compute test tool center point
    test_tool_point_cloud = test_tool_point_cloud.rotate(test_rotation_matrix, center=test_rotation_center)
    test_tool_point_cloud.translate(-test_centroid)
    # remove outliers
    cl, ind = test_tool_point_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    filtered_test_tool_point_cloud = test_tool_point_cloud.select_by_index(ind)
    # get aabb and center point
    aabb = filtered_test_tool_point_cloud.get_axis_aligned_bounding_box()
    box_points = np.asarray(aabb.get_box_points())
    test_tool_center_point_3d = np.mean(box_points, axis=0)

    # hack: center point projection
    print("enable center point projection.")
    direction_xy = test_tool_grasp_point_3d[:2] - test_tool_func_point_3d[:2]
    direction_xy_unit = direction_xy / np.linalg.norm(direction_xy)
    center_vector_xy = test_tool_center_point_3d[:2] - test_tool_func_point_3d[:2]
    projection_length = np.dot(center_vector_xy, direction_xy_unit)
    refined_xy = test_tool_func_point_3d[:2] + projection_length * direction_xy_unit
    test_tool_center_point_3d = np.array([refined_xy[0], refined_xy[1], test_tool_center_point_3d[2]])

    # hack: center point correction
    grasp_to_func_vector = test_tool_grasp_point_3d - test_tool_func_point_3d
    t = (test_tool_center_point_3d[0] - test_tool_func_point_3d[0]) / grasp_to_func_vector[0]
    z_on_line = test_tool_func_point_3d[2] + t * grasp_to_func_vector[2]
    if task_label == 'scoop' or task_label == 'pour':
        if test_tool_center_point_3d[2] > z_on_line:
            print("center point is above the function-grasp axis!")
            print("enable center point correction.")
            test_tool_center_point_3d[-1] = z_on_line - 0.01
    elif task_label == 'sweep' or task_label == 'cut':
        if test_tool_center_point_3d[2] < z_on_line:
            print("center point is below the function-grasp axis!")
            print("enable center point correction.")
            test_tool_center_point_3d[-1] = z_on_line + 0.01
    # no correction for pounding

    ########### Test tool functional keypoint visualization ############
    spheres = []
    for point in [test_tool_func_point_3d]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 0, 0.5])  # purple color
        spheres.append(sphere)
    for point in [test_tool_grasp_point_3d]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0, 0, 1])  # blue color
        spheres.append(sphere)
    for point in [test_tool_center_point_3d]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.647, 0.165, 0.165])  # red color
        spheres.append(sphere)

    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    combined_point_cloud = transformed_test_scene_point_cloud + spheres_combined
    o3d.io.write_point_cloud(os.path.join(output_path, f'test_scene_func_center_grasp_points.ply'), combined_point_cloud)

    ############ Functional correspondence establishment ############
    # scale the demo tool functional points
    test_demo_target_scaling_factor = [a / b for a, b in zip(test_target_dim, demo_target_dim)]
    scaled_demo_tool_func_point_3d = [a * b for a, b in zip(demo_tool_func_point_3d, test_demo_target_scaling_factor)]
    scaled_demo_tool_func_diff = [a - b for a, b in zip(scaled_demo_tool_func_point_3d, demo_tool_func_point_3d)]
    scaled_demo_center_point_3d = [a + b for a, b in zip(demo_tool_center_point_3d, scaled_demo_tool_func_diff)]
    scaled_demo_grasp_point_3d = [a + b for a, b in zip(demo_tool_grasp_point_3d, scaled_demo_tool_func_diff)]

    # align the test tool functional points to scaled demo functional points
    A_raw = np.array([scaled_demo_tool_func_point_3d, scaled_demo_center_point_3d, scaled_demo_grasp_point_3d])
    B = np.array([test_tool_func_point_3d, test_tool_center_point_3d, test_tool_grasp_point_3d])

    # coarse initial position alignment
    R_z = calculate_rotation_matrix_z(
        test_tool_func_point_3d,
        functional_keypoint_3d['function'][keyframe_idx])
    A = np.dot(R_z, A_raw.T).T

    # (1) function point alignment, (2) function plane alignment, and (3) function axis alignment
    B_aligned_list,  B_aligned_trans_list = align_set_B_to_A_v3(A, B)

    # (3) function axis alignment
    image_size = test_init_frame_pil.size
    inverse_rotation_matrix = np.transpose(test_rotation_matrix)
    pose_img_list = []
    test_tool_pc_list = []
    for idx in range(len(B_aligned_list)):

        B_aligned_point_trans,  B_aligned_plane_trans, B_aligned_edge_trans, B_aligned_final_trans = B_aligned_trans_list[idx]
        
        # Create a new point cloud by copying the points and colors from the original
        transformed_point_cloud = o3d.geometry.PointCloud()
        transformed_point_cloud_cam = o3d.geometry.PointCloud()

        # move test tool pc to the aligned pose for rendering 
        points = np.asarray(filtered_test_tool_point_cloud.points)
        points_translated = points + B_aligned_point_trans
        points_aligned_plane = np.dot(points_translated - A[0], B_aligned_plane_trans) + A[0]
        points_aligned_edge = np.dot(points_aligned_plane - A[0], B_aligned_edge_trans) + A[0]
        points_aligned_final = np.dot(points_aligned_edge - A[0], B_aligned_final_trans) + A[0]

        transformed_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points_aligned_final))
        transformed_point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(filtered_test_tool_point_cloud.colors))  # Add color data
        test_tool_pc_list.append(transformed_point_cloud)
        transformed_point_cloud_cam.points = o3d.utility.Vector3dVector(np.asarray(points_aligned_final))
        transformed_point_cloud_cam.colors = o3d.utility.Vector3dVector(np.asarray(filtered_test_tool_point_cloud.colors))  # Add color data
        
        # Apply inverse transformation to cam frame
        transformed_point_cloud_cam.translate(test_centroid)
        transformed_point_cloud_cam.rotate(inverse_rotation_matrix, center=test_rotation_center)
        transformed_point_cloud_cam += test_target_point_cloud

        # backprojection for rendering
        points_3d = np.asarray(transformed_point_cloud_cam.points)
        colors = np.asarray(transformed_point_cloud_cam.colors)  # Get the colors (RGB)
        projected_points_2d = []
        for point_3d in points_3d:
            point_2d = backproject_3d_2d(point_3d, CAM_MATRIX)
            projected_points_2d.append(point_2d)
        projected_points_2d = np.asanyarray(projected_points_2d)

        image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 200
        depth_buffer = np.full((image_size[1], image_size[0]), np.inf)
        for pt, color in zip(projected_points_2d, colors):
            x, y, z = int(pt[0]), int(pt[1]), pt[2]
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                if z < depth_buffer[y, x]:
                    depth_buffer[y, x] = z
                    rgb_color = (color * 255).astype(np.uint8)
                    image[y, x] = rgb_color

        # savb rendered images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_img_list.append(Image.fromarray(image_rgb))
        output_image_path = os.path.join(output_path, f'test_projected_image_{idx}.png')
        cv2.imwrite(output_image_path, image_rgb)
    print("projection and image rendering completed.")

    # (3) function axis alignment: visual prompting
    task = {}
    task['object_grasped'] = test_tool_label
    task['object_unattached'] = test_target_label
    task['task_instruction'] = f'use a {test_tool_label} to {task_label} {test_target_label}'

    if vp_flag:
        # visual prompting (not quite stable)
        context = pose_selection(
                    task,
                    pose_img_list,
                    prompts=prompts['select_pose'], 
                    debug=True,
            )
        print(task)
        print(context)
        selected_idx = context['selected_idx']
    else:
        selected_idx = len(pose_img_list) // 2

    o3d.io.write_point_cloud(os.path.join(output_path, 'refined_test_tool_pc.ply'), test_tool_pc_list[selected_idx])

    # save the aligned test tool functional keypoints
    test_tool_func_point_3d_aligned, test_tool_center_point_3d_aligned, test_tool_grasp_point_3d_aligned = B_aligned_list[selected_idx]

    test_tool_track_dict = {'function_init': test_tool_func_point_3d, 
                            'center_init': test_tool_center_point_3d, 
                            'grasp_init': test_tool_grasp_point_3d,
                            'function_func':test_tool_func_point_3d_aligned,
                            'center_func': test_tool_center_point_3d_aligned,
                            'grasp_func': test_tool_grasp_point_3d_aligned
                            }
    with open(os.path.join(output_path, 'test_tool_track.json'), "w") as json_file:
        json.dump(test_tool_track_dict, json_file, default=convert_to_serializable, indent=4)
    print("test tool function keyframe functional keypoints saved.")

    ########### 3D functional keypoints visualization ############
    # visualize the function keyframe demo tool functional keypoints
    spheres = []
    for point in [A[0]]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 0, 0.5])  # function
        spheres.append(sphere)
    for point in [A[1]]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])  # center
        spheres.append(sphere)
    for point in [A[2]]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0.647, 0.165, 0.165])  # grasp
        spheres.append(sphere)
    
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    # combined_point_cloud = scene_point_cloud + spheres_combined
    o3d.io.write_point_cloud(os.path.join(output_path, 'demo_func_center_grasp_points.ply'), spheres_combined)

    # visualize the aligned test tool functional keypoints
    spheres = []
    for point in [test_tool_func_point_3d_aligned]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 0, 0.5]) 
        spheres.append(sphere)
    for point in [test_tool_grasp_point_3d_aligned]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])  
        spheres.append(sphere)
    for point in [test_tool_center_point_3d_aligned]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) 
        sphere.translate(point)
        sphere.paint_uniform_color([0.647, 0.165, 0.165])  
        spheres.append(sphere)
    
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    o3d.io.write_point_cloud(os.path.join(output_path, 'test_func_grasp_center_points_aligned.ply'), spheres_combined)





   
