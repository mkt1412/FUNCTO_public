import os 
import cv2
import numpy as np
import json
from PIL import Image
import open3d as o3d

from cloud_services.apis.owlv2 import OWLViT
from cloud_services.apis.sam import SAM

from utils_IL.perception_utils import non_maximum_suppression, convert_to_serializable
from utils_IL.geometry_utils import compute_point_cloud, compute_3d_points
from utils_IL.vp_utils import propose_candidate_keypoints, annotate_visual_prompts, request_motion, generate_vertices_and_update, load_prompts
from utils_IL.params import CAM_MATRIX

from openai import OpenAI
client = OpenAI()


def func_point_detect(data_path, tool_label='mug', obj_label='bowl', task_label='pour', num_candidate_keypoints=10):
    output_path = os.path.join(data_path, 'func_point_det_output')
    prompt_path = os.path.join(data_path, '..', '..', 'utils_IL', 'prompts')
    os.makedirs(output_path, exist_ok=True)

    # load pre-function keyframe idx
    keyframe_idx_path = os.path.join(data_path, 'key_point_track_output', 'keyframe_idx.json')
    with open(keyframe_idx_path, 'r') as file:
        keyframe_idx_dict = json.load(file)
    pre_func_frame_index = keyframe_idx_dict['pre-func']

    pre_func_frame_filename = f'{pre_func_frame_index:05d}.jpg'
    pre_func_frame_file_path = os.path.join(data_path, 'rgb', pre_func_frame_filename)
    pre_func_frame_depth_filename = f'{pre_func_frame_index:05d}.png'
    pre_func_frame_depth_file_path = os.path.join(data_path, 'depth', pre_func_frame_depth_filename)

    obs_image = Image.open(pre_func_frame_file_path).convert('RGB')
    obs_image_np = np.array(obs_image)
    obs_image_depth = Image.open(pre_func_frame_depth_file_path)
    obs_img_depth_np = np.array(obs_image_depth)

    task_instruction = f'use a {tool_label} to {task_label} {obj_label}'
    # print('Task:', task_instruction)

    task = {}
    task['object_grasped'] = tool_label
    task['object_unattached'] = obj_label
    task['task_instruction'] = task_instruction
    all_object_names = [tool_label, obj_label]

    # load prompts
    prompts = load_prompts(prompt_path)
    
    owl = OWLViT()
    sam_predictor = SAM()

    ############ Detection ############
    detected_objects = owl.detect_objects(
        obs_image,
        all_object_names,
        bbox_score_top_k=20,
        bbox_conf_threshold=0.1
    )

    if len(detected_objects):
        x = obs_image.size[0]
        y = obs_image.size[1]

        # Non-Maximum Suppression
        detected_objects = non_maximum_suppression(detected_objects, iou_threshold=0.2)
        
        filtered_objects = {}
        for detection in detected_objects:
            box_name = detection['box_name']
            if box_name not in filtered_objects or detection['score'] > filtered_objects[box_name]['score']:
                filtered_objects[box_name] = detection
        detected_objects = list(filtered_objects.values())

        boxes_l = [[x * obj['bbox'][0], y * obj['bbox'][1], x * obj['bbox'][2], y * obj['bbox'][3]] for obj in detected_objects]
        pred_phrases = [obj['box_name'] for obj in detected_objects]
    else:
        raise RuntimeError("no tool detected.")

    ############ Get interaction region ############
    converted_boxes = boxes_l
    x_min_union, y_min_union, x_max_union, y_max_union = converted_boxes[0]
    for box in converted_boxes[1:]:
        x_min, y_min, x_max, y_max = box
        x_min_union = min(x_min_union, x_min)
        y_min_union = min(y_min_union, y_min)
        x_max_union = max(x_max_union, x_max)
        y_max_union = max(y_max_union, y_max)

    # expand the union bbox
    x_min_union *= 0.9
    y_min_union *= 0.9
    x_max_union *= 1.05
    y_max_union *= 1.05
    union_bbox = (x_min_union, y_min_union, x_max_union, y_max_union)

    # crop the union bbox
    cropped_image = obs_image.crop(union_bbox)
    cropped_image_depth = obs_image_depth.crop(union_bbox)
    # cropped_image.save(os.path.join(output_path, 'interaction_region_crop.png'))
    cropped_image_np = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    
    # compute coordinate of the union bbox
    cropped_width, cropped_height = cropped_image.size
    offset_x = x_min_union
    offset_y = y_min_union

    adjusted_boxes = []
    adjusted_boxes_01 = []
    for i, box in enumerate(converted_boxes):
        x_min, y_min, x_max, y_max = box
        
        # Adjust coordinates to the cropped image
        adj_x_min = x_min - offset_x
        adj_y_min = y_min - offset_y
        adj_x_max = x_max - offset_x
        adj_y_max = y_max - offset_y

        cv2.rectangle(cropped_image_np, (int(adj_x_min), int(adj_y_min)), (int(adj_x_max), int(adj_y_max)), (0, 255, 0), 2)

        adjusted_boxes.append([adj_x_min, adj_y_min, adj_x_max, adj_y_max])
        adjusted_boxes_01.append([adj_x_min/cropped_width, adj_y_min/cropped_height, adj_x_max/cropped_width, adj_y_max/cropped_height])
    
    cv2.imwrite(os.path.join(output_path, 'interaction_region_bbox.png'), cropped_image_np)

    ############ Segmentation ############
    masks = sam_predictor.segment_by_bboxes(image=cropped_image, bboxes=adjusted_boxes_01)
    masks = [anno["segmentation"] for anno in masks]

    # # Save masks according to the label of each bbox
    # for i, mask in enumerate(masks):
    #     mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to binary image
    #     mask_output_path_png = os.path.join(output_path, f'{pred_phrases[i]}_mask.png')
    #     mask_output_path_npy = os.path.join(output_path, f'{pred_phrases[i]}_mask.npy')
        
    #     # Save mask in png and npy
    #     # mask_image.save(mask_output_path_png)
    #     # np.save(mask_output_path_npy, mask)

    # record masks according to the label of each bbox
    segmasks = {}
    for i, test_mask in enumerate(masks):
        # only sample from nonzero-depth point
        cropped_depth = np.array(cropped_image_depth)
        depth_mask = (cropped_depth != 0).astype(np.uint8)
        test_mask_uint8 = test_mask.astype(np.uint8)
        filtered_test_mask_uint8 = cv2.bitwise_and(test_mask_uint8, depth_mask)

        # erode the mask to avoid edge point
        kernel = np.ones((6, 6), np.uint8) 
        eroded_mask = cv2.erode(filtered_test_mask_uint8.astype(np.uint8), kernel, iterations=1)
        # eroded_mask_image = Image.fromarray(eroded_mask * 255)
        # eroded_mask_image_path = os.path.join(output_path, f'{pred_phrases[i]}_eroded_mask.png')
        # eroded_mask_image.save(eroded_mask_image_path)

        final_test_mask = eroded_mask.astype(bool)
        segmasks[pred_phrases[i]] = {'mask': final_test_mask}

     ############ Visual prompting ############  
    segmasks = generate_vertices_and_update(segmasks)

    # Annotate visual marks
    candidate_keypoints = propose_candidate_keypoints(
        task,
        segmasks, 
        num_samples=num_candidate_keypoints)
    
    annotated_image = annotate_visual_prompts(
                cropped_image,
                candidate_keypoints)
    annotated_image.save(os.path.join(output_path, 'visual_prompting_candidates.png'))


    # visual prompting
    context, _, vp_img = request_motion(
                task,
                cropped_image,
                annotated_image,
                candidate_keypoints, 
                prompts=prompts, 
                debug=True
        )
    vp_img.save(os.path.join(output_path, 'visual_prompting_result.png'))

    print(task)
    print(context)

    ############ Visual prompting post-processing ############ 
    context_crop = {}
    context_full = {}
    context_3d = {}

    for i, box in enumerate(adjusted_boxes):
        label = pred_phrases[i]

        if label != tool_label:
            continue

        x_crop, y_crop = context['keypoints_2d']['function']
        adj_x_min, adj_y_min, adj_x_max, adj_y_max = box
        # crop_obj = cropped_image.crop((adj_x_min, adj_y_min, adj_x_max, adj_y_max))
        # crop_obj.save(os.path.join(output_path, 'tool.jpg'))

        offset_x = adj_x_min
        offset_y = adj_y_min
        crop_obj_x, crop_obj_y = x_crop - offset_x, y_crop - offset_y

        # save function point on the full pre-function keframe
        full_obj_x, full_obj_y = x_crop + x_min_union, y_crop + y_min_union

        # get 3d function point
        func_point_depth = obs_img_depth_np[int(full_obj_y), int(full_obj_x)]

        if func_point_depth == 0:
            raise RuntimeError(f"{label} has zero-depth function point")
        func_point_3d = [int(full_obj_x), int(full_obj_y), func_point_depth]

        # compute 3D function point in the camera frame
        _, _, points_3d = compute_3d_points(CAM_MATRIX, [func_point_3d])
        points, colors, points_3d = compute_point_cloud(obs_image_np, obs_img_depth_np, CAM_MATRIX, [func_point_3d])
        scene_point_cloud = o3d.geometry.PointCloud()
        scene_point_cloud.points = o3d.utility.Vector3dVector(points)
        scene_point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # transform to target object frame
        cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
        with open(cam_to_target_trans_path, 'r') as file:
            cam_to_target_trans_dict = json.load(file)
        rotation_matrix = np.array(cam_to_target_trans_dict['rotation_matrix'])
        rotation_center = np.array(cam_to_target_trans_dict['rotation_center'])
        centroid = np.array(cam_to_target_trans_dict['centroid'])

        rotated_scene_point_cloud = scene_point_cloud.rotate(rotation_matrix, center=rotation_center)
        rotated_scene_point_cloud.translate(-centroid)
        points_3d = np.array(points_3d[0])
        rotated_points_3d = np.dot(points_3d - rotation_center, rotation_matrix.T) + rotation_center
        rotated_points_3d -= centroid
        
        # visualize predicted function point in the target object frame
        spheres = []
        for point in [rotated_points_3d]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Adjust the radius as needed
            sphere.translate(point)
            sphere.paint_uniform_color([0.5, 0, 0.5])  # purple for function point
            spheres.append(sphere)
        # Combine the transformed target object point cloud and the spheres
        spheres_combined = o3d.geometry.PointCloud()
        for sphere in spheres:
            sphere_pc = sphere.sample_points_poisson_disk(100)
            spheres_combined += sphere_pc
        combined_point_cloud = rotated_scene_point_cloud + spheres_combined

        o3d.io.write_point_cloud(os.path.join(output_path, f'3d_scene_w_function_point.ply'), combined_point_cloud)

        context_3d['function'] = rotated_points_3d
        context_crop['function'] = [int(crop_obj_x), int(crop_obj_y)]
        context_full['function'] = [int(full_obj_x), int(full_obj_y)]

    # save function point info
    with open(os.path.join(output_path, 'func_point_crop_out.json'), "w") as json_file:
        json.dump(context_crop, json_file, default=convert_to_serializable, indent=4)
    
    with open(os.path.join(output_path, 'func_point_full_out.json'), "w") as json_file:
        json.dump(context_full, json_file, default=convert_to_serializable, indent=4)

    with open(os.path.join(output_path, 'func_point_3d_out.json'), "w") as json_file:
        json.dump(context_3d, json_file, default=convert_to_serializable, indent=4)

    return None
