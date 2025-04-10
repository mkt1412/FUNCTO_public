import os
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import json

from cloud_services.apis.owlv2 import OWLViT
from cloud_services.apis.sam import SAM

from utils_IL.perception_utils import non_maximum_suppression
from utils_IL.geometry_utils import compute_point_cloud
from utils_IL.params import CAM_MATRIX

def demo_test_deteciton(data_path, tool_label, target_label):
    """
    detect tools and targets in demo and test scenes
    """

    output_path = os.path.join(data_path, 'detection_output')
    os.makedirs(output_path, exist_ok=True)

    initial_frame_index = 0
    initial_frame_filenname = f'{initial_frame_index:05d}.jpg'
    initial_frame_file_path = os.path.join(data_path, 'rgb', initial_frame_filenname)
    initial_frame_pil = Image.open(initial_frame_file_path).convert('RGB')
    initial_frame = np.array(initial_frame_pil)

    owl = OWLViT()
    sam_predictor = SAM()

    ############ Demo/test tool detection ############
    detected_objects = owl.detect_objects(
        initial_frame_pil,
        [tool_label],
        bbox_score_top_k=20,
        bbox_conf_threshold=0.1
    )

    if len(detected_objects):
        x = initial_frame_pil.size[0]
        y = initial_frame_pil.size[1]

        # Non-Maximum Suppression
        detected_objects = non_maximum_suppression(detected_objects, iou_threshold=0.2)
        
        # sort the detected object of each class according to the confidence
        filtered_objects = {}
        for detection in detected_objects:
            box_name = detection['box_name']
            if box_name not in filtered_objects or detection['score'] > filtered_objects[box_name]['score']:
                filtered_objects[box_name] = detection
        detected_objects = list(filtered_objects.values())

        boxes_l = [[x * obj['bbox'][0], y * obj['bbox'][1], x * obj['bbox'][2], y * obj['bbox'][3]] for obj in detected_objects]
        pred_phrases = [obj['box_name'] for obj in detected_objects]

        # Draw bounding boxes and labels on the image
        for i, box in enumerate(boxes_l):
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(initial_frame, top_left, bottom_right, (255, 0, 0), 2)  # Blue color box
            cv2.putText(initial_frame, pred_phrases[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Convert the image back to BGR for saving with OpenCV
            initial_frame_bgr = cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
            output_image_path = os.path.join(output_path, 'tool_bbox.png')
            cv2.imwrite(output_image_path, initial_frame_bgr)
            print(f"image with bounding boxes saved at {output_image_path}")
        
        # save bbox info
        tool_bbox_path = os.path.join(output_path, 'tool.json')
        with open(tool_bbox_path, 'w') as json_file:
            json.dump(box, json_file, indent=4)

        ############ Demo tool segmentation ############
        masks = sam_predictor.segment_by_bboxes(image=initial_frame_pil, bboxes=[obj['bbox'] for obj in detected_objects])
        masks = [anno["segmentation"] for anno in masks]

        # Save masks according to the label of each bbox
        for i, mask in enumerate(masks):
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to binary image
            mask_output_path_png = os.path.join(output_path, 'tool_mask.png')
            mask_output_path_npy = os.path.join(output_path, 'tool_mask.npy')
            
            # Save mask as PNG
            mask_image.save(mask_output_path_png)
            print(f"mask for {pred_phrases[i]} saved at {mask_output_path_png}")
            
            # Save mask as NPY
            np.save(mask_output_path_npy, mask)
            print(f"mask for {pred_phrases[i]} saved as NPY at {mask_output_path_npy}")
    else:
        raise RuntimeError("no tool detected!")

    ############ Demo/test target detection ############
    detected_objects = owl.detect_objects(
        initial_frame_pil,
        [target_label],
        bbox_score_top_k=20,
        bbox_conf_threshold=0.1
    )

    if len(detected_objects):
        x = initial_frame_pil.size[0]
        y = initial_frame_pil.size[1]

        # Non-Maximum Suppression
        detected_objects = non_maximum_suppression(detected_objects, iou_threshold=0.2)
        
        # sort the detected object of each class according to the confidence
        filtered_objects = {}
        for detection in detected_objects:
            box_name = detection['box_name']
            if box_name not in filtered_objects or detection['score'] > filtered_objects[box_name]['score']:
                filtered_objects[box_name] = detection
        detected_objects = list(filtered_objects.values())

        boxes_l = [[x * obj['bbox'][0], y * obj['bbox'][1], x * obj['bbox'][2], y * obj['bbox'][3]] for obj in detected_objects]
        pred_phrases = [obj['box_name'] for obj in detected_objects]

        # Draw bounding boxes and labels on the image
        for i, box in enumerate(boxes_l):
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(initial_frame, top_left, bottom_right, (255, 0, 0), 2)  # Blue color box
            cv2.putText(initial_frame, pred_phrases[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Convert the image back to BGR for saving with OpenCV
            initial_frame_bgr = cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
            output_image_path = os.path.join(output_path, 'target_bbox.png')
            cv2.imwrite(output_image_path, initial_frame_bgr)
            print(f"image with bounding boxes saved at {output_image_path}")
        
        # save bbox info
        target_bbox_path = os.path.join(output_path, 'target.json')
        with open(target_bbox_path, 'w') as json_file:
            json.dump(box, json_file, indent=4)

        ############ Demo target segmentation ############
        masks = sam_predictor.segment_by_bboxes(image=initial_frame_pil, bboxes=[obj['bbox'] for obj in detected_objects])
        masks = [anno["segmentation"] for anno in masks]

        # Save masks according to the label of each bbox
        for i, mask in enumerate(masks):
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to binary image
            mask_output_path_png = os.path.join(output_path, 'target_mask.png')
            mask_output_path_npy = os.path.join(output_path, 'target_mask.npy')
            
            # Save mask as PNG
            mask_image.save(mask_output_path_png)
            print(f"mask for {pred_phrases[i]} saved at {mask_output_path_png}")
            
            # Save mask as NPY
            np.save(mask_output_path_npy, mask)
            print(f"mask for {pred_phrases[i]} saved as NPY at {mask_output_path_npy}")
    else:
        raise RuntimeError("no target detected!")
    
    return None

def cam_to_target_trans(data_path):

    """"
    transform from camera to target object frame
    """
    output_path = os.path.join(data_path, 'cam_to_target_trans_output')
    os.makedirs(output_path, exist_ok=True)

    initial_frame_index = 0
    initial_frame_filenname = f'{initial_frame_index:05d}.jpg'
    initial_frame_file_path = os.path.join(data_path, 'rgb', initial_frame_filenname)
    initial_frame_depth_filename = f'{initial_frame_index:05d}.png'
    initial_frame_depth_file_path = os.path.join(data_path, 'depth', initial_frame_depth_filename)

    initial_frame_pil = Image.open(initial_frame_file_path).convert('RGB')
    initial_frame_depth_pil = Image.open(initial_frame_depth_file_path)
    initial_frame = np.array(initial_frame_pil)
    initial_frame_depth = np.array(initial_frame_depth_pil)

    ############ 3D target object detection ############
    # reconstruct the scene point cloud
    # mask = (initial_frame_depth == 0).astype(np.uint8)
    initial_frame_depth = initial_frame_depth.astype(np.float32)
    points, colors, _ = compute_point_cloud(initial_frame, initial_frame_depth, CAM_MATRIX, [])
    scene_point_cloud = o3d.geometry.PointCloud()
    scene_point_cloud.points = o3d.utility.Vector3dVector(points)
    scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)

     # plane fitting
    plane_model, inliers = scene_point_cloud.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"plane equation: {a}x + {b}y + {c}z + {d} = 0")
    normal_vector = np.array([a, b, c])
    normal_vector /= np.linalg.norm(normal_vector)  # the normal vector goes down
    z_axis = np.array([0, 0, -1])  # Z-axis vector
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(normal_vector, z_axis))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    # rotate the scene point cloud for visualization
    rotated_scene_point_cloud = scene_point_cloud.rotate(rotation_matrix, center=[0, 0, 0])

    # get target object point cloud
    target_mask_path = os.path.join(data_path, 'detection_output', 'target_mask.npy')
    target_mask = np.load(target_mask_path)
    target_mask_uint8 = target_mask.astype(np.uint8)
    masked_initial_frame = cv2.bitwise_and(initial_frame, initial_frame, mask=target_mask_uint8)
    masked_initial_frame_depth = cv2.bitwise_and(initial_frame_depth, initial_frame_depth, mask=target_mask_uint8)
    points, colors, points_3d = compute_point_cloud(masked_initial_frame, masked_initial_frame_depth, CAM_MATRIX, [])
    target_point_cloud = o3d.geometry.PointCloud()
    target_point_cloud.points = o3d.utility.Vector3dVector(points)
    target_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # remove target object outliers and rotate
    cl, ind = target_point_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.75)
    filtered_target_point_cloud = target_point_cloud.select_by_index(ind)
    rotated_target_point_cloud = filtered_target_point_cloud.rotate(rotation_matrix, center=[0, 0, 0])

    # get axis-aligned bounding box and compute the center point
    aabb = rotated_target_point_cloud.get_axis_aligned_bounding_box()
    box_points = np.asarray(aabb.get_box_points())
    box_dim = aabb.get_max_bound()- aabb.get_min_bound()
    center_point = np.mean(box_points, axis=0)
    translated_box_points = box_points - center_point

    # move the origin to the target object
    rotated_scene_point_cloud_pts = np.asarray(rotated_scene_point_cloud.points)
    rotated_scene_point_cloud_pts -= center_point
    rotated_scene_point_cloud.points = o3d.utility.Vector3dVector(rotated_scene_point_cloud_pts)
    o3d.io.write_point_cloud(os.path.join(output_path, 'transformed_scene_point_cloud.ply'), rotated_scene_point_cloud)
    rotated_target_point_cloud_pts = np.asarray(rotated_target_point_cloud.points)
    rotated_target_point_cloud_pts -= center_point
    rotated_target_point_cloud.points = o3d.utility.Vector3dVector(rotated_target_point_cloud_pts)
    print("target object frame transformation done.")

    # save rotation matrix, rotattion center, centroid, box dim, and box points
    cam_to_target_trans = {}
    cam_to_target_trans['rotation_matrix'] = rotation_matrix.tolist()
    cam_to_target_trans['rotation_center'] = [0, 0, 0]
    cam_to_target_trans['centroid'] = center_point.tolist()
    cam_to_target_trans['target_dim'] = box_dim.tolist()
    cam_to_target_trans['box_points'] = translated_box_points.tolist()

    with open(os.path.join(output_path, 'cam_to_target_trans.json' ), 'w') as json_file:
        json.dump(cam_to_target_trans, json_file, indent=4)

    ########### 3D target object detection visualization ############
    spheres = []
    for point in translated_box_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
        sphere.translate(point)
        sphere.paint_uniform_color([0, 1, 0])  # Green color
        spheres.append(sphere)
    # Combine the transformed target object point cloud and the spheres
    spheres_combined = o3d.geometry.PointCloud()
    for sphere in spheres:
        sphere_pc = sphere.sample_points_poisson_disk(100)
        spheres_combined += sphere_pc
    combined_point_cloud = rotated_target_point_cloud + spheres_combined
    o3d.io.write_point_cloud(os.path.join(output_path, 'target_point_cloud_with_aabb_corners.ply'), combined_point_cloud)

    return None

 

