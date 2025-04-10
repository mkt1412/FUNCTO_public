import numpy as np
import cv2
import json
import os
from PIL import Image

from cloud_services.apis.owlv2 import OWLViT
from cloud_services.apis.sam import SAM

from utils_IL.perception_utils import fit_gaussian, convert_to_serializable, non_maximum_suppression_class
from utils_IL.geometry_utils import get_depth_value, compute_3d_points
from utils_IL.params import CAM_MATRIX

def dilate_masks(mask1, mask2, kernel_size=(5, 5), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_mask1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=iterations)
    dilated_mask2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=iterations)
    return dilated_mask1, dilated_mask2

def intersect_masks(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    return intersection

def draw_points_on_image(image, points, color=(0, 255, 0), radius=5, thickness=-1):
    # Ensure points are integers
    points = points.astype(int)
    cv2.circle(image, (points[1], points[0]), radius, color, thickness)
    return image

def detect_grasp_point(img, hand_mask, tool_mask):
    dilated_mask1, dilated_mask2 = dilate_masks(hand_mask, tool_mask)
    intersection = intersect_masks(dilated_mask1, dilated_mask2)
    intersection_points = np.argwhere(intersection > 0)
    intersection_points[:, 0] += 1
    intersection_points[:, 1] += 1

    if intersection_points.shape[0] > 0:
        print("intersection points detected.")
        mean, cov = fit_gaussian(intersection_points)
        img_with_points = draw_points_on_image(img, mean)
    else:
        raise RuntimeError("no intersection points detected!")

    return img_with_points, intersection_points

def grasp_point_detection(data_path, tool_label='mug'):
    output_path = os.path.join(data_path, 'grasp_det_output')
    os.makedirs(output_path, exist_ok=True)

    owl = OWLViT()
    sam_predictor = SAM()
    text_queries = ['hand', tool_label]

    # read grasp frame idx
    keyframe_idx_path = os.path.join(data_path, 'key_point_track_output', 'keyframe_idx.json')
    with open(keyframe_idx_path, 'r') as file:
        keyframe_idx_dict = json.load(file)
    grasp_frame_idx = keyframe_idx_dict['grasp']

    # load grasping and initial keyframes
    grasping_frame_path = os.path.join(data_path, 'rgb', f'{grasp_frame_idx:05}.jpg')
    grasping_frame = cv2.imread(grasping_frame_path)
    if grasping_frame is None:
        raise FileNotFoundError(f"Image not found at path: {grasping_frame_path}")
    grasping_frame = cv2.cvtColor(grasping_frame, cv2.COLOR_BGR2RGB)
    grasping_frame_pil = Image.fromarray(grasping_frame)

    init_frame_idx = 0
    init_depth_frame_path = os.path.join(data_path, 'depth', f'{init_frame_idx:05}.png')
    init_frame_depth_pil = Image.open(init_depth_frame_path)
    init_frame_depth = np.array(init_frame_depth_pil)

    ############ Detection ############
    detected_objects = owl.detect_objects(
        grasping_frame_pil,
        text_queries,
        bbox_score_top_k=20,
        bbox_conf_threshold=0.12
    )   

    if len(detected_objects):
        x = grasping_frame_pil.size[0]
        y = grasping_frame_pil.size[1]

        detected_objects = non_maximum_suppression_class(detected_objects, iou_threshold=0.2)

        filtered_objects = {}
        for detection in detected_objects:
            box_name = detection['box_name']
            if box_name not in filtered_objects or detection['score'] > filtered_objects[box_name]['score']:
                filtered_objects[box_name] = detection
        detected_objects = list(filtered_objects.values())

        boxes_l = [[x * obj['bbox'][0], y * obj['bbox'][1], x * obj['bbox'][2], y * obj['bbox'][3]] for obj in detected_objects]
        pred_phrases = [obj['box_name'] for obj in detected_objects]

        # Draw bounding boxes and labels on the image
        for i, box in enumerate(boxes_l[:2]):
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(grasping_frame, top_left, bottom_right, (255, 0, 0), 2)  # Blue color box
            cv2.putText(grasping_frame, pred_phrases[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        grasping_frame_bgr = cv2.cvtColor(grasping_frame, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(output_path, 'grasping_frame_with_boxes.jpg')
        cv2.imwrite(output_image_path, grasping_frame_bgr)
        print(f"image with bounding boxes saved at {output_image_path}")

        ############ Segmentation ############
        masks = sam_predictor.segment_by_bboxes(image=grasping_frame_pil, bboxes=[obj['bbox'] for obj in detected_objects])
        masks = [anno["segmentation"] for anno in masks]

        # Save masks according to the label of each bbox
        for i, mask in enumerate(masks):
            if pred_phrases[i] == 'hand':
                hand_mask = mask
            else:
                tool_mask = mask
            
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to binary image
            mask_output_path_png = os.path.join(output_path, f'{pred_phrases[i]}_mask.png')
            
            # Save mask as PNG
            mask_image.save(mask_output_path_png)
            print(f"mask for {pred_phrases[i]} saved at {mask_output_path_png}")
    else:
        raise RuntimeError("No object detected.")
    
    ############ 2D grasp point detection ############
    img_vis, grasp_points = detect_grasp_point(grasping_frame, hand_mask, tool_mask)

    # Save the visualization image
    img_vis_output_path = os.path.join(output_path, 'grasp_point_vis.jpg')
    cv2.imwrite(img_vis_output_path,cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    print(f"visualization image saved at {img_vis_output_path}")

    # Save the grasp points in a JSON file
    grasp_points_list = [point.tolist() if isinstance(point, np.ndarray) else point for point in grasp_points]
    grasp_points_output_path = os.path.join(output_path, 'grasp_points.json')
    with open(grasp_points_output_path, 'w') as json_file:
        json.dump(grasp_points_list, json_file, indent=4)

    ############ 3D grasp point detection ############
    grasp_point_2d, _ = fit_gaussian(grasp_points)
    y, x = int(grasp_point_2d[0]), int(grasp_point_2d[1]) 
    depth_value, _ = get_depth_value(init_frame_depth, y, x) 
    grasp_point_3d = [x, y, depth_value]
    _, _, grasp_point_3d = compute_3d_points(CAM_MATRIX, [grasp_point_3d])

    # tranform to target object frame
    cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
    with open(cam_to_target_trans_path, 'r') as file:
        cam_to_target_trans_dict = json.load(file)
    rotation_matrix = np.array(cam_to_target_trans_dict['rotation_matrix'])
    rotation_center = np.array(cam_to_target_trans_dict['rotation_center'])
    centroid = np.array(cam_to_target_trans_dict['centroid'])

    grasp_points_3d = np.array(grasp_point_3d[0])
    rotated_grasp_points_3d = np.dot(grasp_points_3d - rotation_center, rotation_matrix.T) + rotation_center
    rotated_grasp_points_3d -= centroid

    json_file_path = os.path.join(output_path, 'grasp_point_3d.json')
    with open(json_file_path, "w") as json_file:
        json.dump(rotated_grasp_points_3d, json_file, default=convert_to_serializable, indent=4)

    return None
    










    