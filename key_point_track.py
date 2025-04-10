import os
import torch
import numpy as np
from PIL import Image
import cv2
import json

from utils_IL.perception_utils import non_maximum_suppression, read_frames_from_path, compute_iou
from utils_IL.geometry_utils import compute_point_cloud, get_depth_value, compute_3d_points
from utils_IL.params import CAM_MATRIX

from cloud_services.apis.owlv2 import OWLViT

owl = OWLViT()

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def key_point_track(data_path, grid_size=80):
    """
    track visible points in the demo video
    """
        
    output_path = os.path.join(data_path, 'key_point_track_output')
    os.makedirs(output_path, exist_ok=True)

    video_path_rgb = os.path.join(data_path, 'rgb')
    video_path_depth = os.path.join(data_path, 'depth')
    video, depth_stream = read_frames_from_path(video_path_rgb, video_path_depth)  # (33, 720, 1280, 3)
    print(f'demo video shape: {video.shape}')
    
    ############ Visible keypoint tracking ############
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    mask = np.load(os.path.join(data_path, 'detection_output', 'tool_mask.npy')).astype(np.uint8)
    # mask = masks[0].astype(np.uint8)

    # tool erosion to remove noise
    kernel = np.ones((3, 3), np.uint8) 
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    eroded_mask_image = Image.fromarray(eroded_mask * 255)
    eroded_mask_image_path = os.path.join(output_path, 'eroded_mask.png')
    eroded_mask_image.save(eroded_mask_image_path)
    segm_mask = torch.from_numpy(eroded_mask)[None, None]

    # load cotracker model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    # track masked object
    pred_tracks, pred_visibility = model(
        video, 
        grid_size=grid_size, 
        backward_tracking=True,
        segm_mask=segm_mask
        )
    print("demo tool computed.") 

    # track consistently visible points
    batch_size, num_frames, num_points, _ = pred_tracks.shape
    visible_points_indices = []

    for point_idx in range(num_points):
        # Check if the point is visible in all frames
        if torch.all(pred_visibility[:, :, point_idx]):
            visible_points_indices.append(point_idx)
    print(f"{len(visible_points_indices)} visible keypoints detected.")
       
    # Extract the tracks of consistently visible points
    consistent_tracks = pred_tracks[:, :, visible_points_indices, :]  # (1, 83, 39, 2)

    depth_images = []
    for frame_idx in range(len(depth_stream)):
        frame_depth = depth_stream[frame_idx]
        frame_depth = frame_depth.astype(np.float32)
        depth_images.append(frame_depth)

    # extract 3d keypoints
    keypoint_depth_list = []
    for i in range(len(visible_points_indices)):
        keypoint_depth = []
        pt_traj = consistent_tracks[0, :, i, :].cpu().numpy()
        # print(f'processing keypoint {i}')
        for frame_idx in range(pt_traj.shape[0]):
            frame_depth = depth_images[frame_idx]
            point = pt_traj[frame_idx]
            x, y = int(point[0]), int(point[1])  
            depth_value = frame_depth[int(y), int(x)]
            keypoint_depth.append(depth_value)
        keypoint_depth_list.append(keypoint_depth)
    
    # filter out keypoint trajectories with zero depth
    traj_pts = []
    for i in range(len(visible_points_indices)):
        pt_traj = consistent_tracks[0, :, i, :].cpu().numpy()
        pt_depth = keypoint_depth_list[i]
        if 0 in pt_depth:
            continue
        # print(f"skip the keypoint {visible_points_indices[i]}")
        for j, point in enumerate(pt_traj):
            depth_value = pt_depth[j]
            point_3d = np.array([point[0], point[1], depth_value])
            traj_pts.append(point_3d)
    
    filtered_num_keypoints = int(len(traj_pts) / pt_traj.shape[0])
    print(f"{filtered_num_keypoints} filtered visible keypoints detected.")

    _, _, points_3d = compute_3d_points(CAM_MATRIX, traj_pts)
    # transform to target object frame
    cam_to_target_trans_path = os.path.join(data_path, 'cam_to_target_trans_output', 'cam_to_target_trans.json')
    with open(cam_to_target_trans_path, 'r') as file:
        cam_to_target_trans_dict = json.load(file)
    rotation_matrix = np.array(cam_to_target_trans_dict['rotation_matrix'])
    rotation_center = np.array(cam_to_target_trans_dict['rotation_center'])
    centroid = np.array(cam_to_target_trans_dict['centroid'])
    points_3d = np.array(points_3d)
    rotated_points_3d = np.dot(points_3d - rotation_center, rotation_matrix.T) + rotation_center
    rotated_points_3d -= centroid

    ############ Save tracked keypoints ############
    points_3d_arr = np.array(rotated_points_3d)
    points_3d_arr = points_3d_arr.reshape((filtered_num_keypoints, pt_traj.shape[0], 3))

    # loop over each frame and record point
    keypoints_data = {}  # frame_index: keypoint locations
    for frame_idx in range(points_3d_arr.shape[1]):
        frame_points = points_3d_arr[:, frame_idx, :].tolist()
        keypoints_data[frame_idx] = frame_points

    # Save tracked key points to a JSON file
    json_file_path = os.path.join(output_path, 'tracked_visible_points.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(keypoints_data, json_file, indent=4)

    print(f"tracked keypoints saved to {json_file_path}")
    
    return None

def pre_func_compute(data_path, tool_label, target_label, pre_func_thresh, bbox_expansion):
    video_path_rgb = os.path.join(data_path, 'rgb')
    video_path_depth = os.path.join(data_path, 'depth')
    video, _ = read_frames_from_path(video_path_rgb, video_path_depth)

    output_path = os.path.join(data_path, 'key_point_track_output')
    os.makedirs(output_path, exist_ok=True)

    # read grasp_idx
    grasp_idx_path = os.path.join(data_path, 'grasp_det_output', 'grasp_frame_list.json')
    with open(grasp_idx_path, 'r') as file:
        grasp_idx = json.load(file)[0]

    text_queries = [tool_label, target_label]

    # loop from the grasp to the last frame
    for idx in range(grasp_idx, video.shape[0]):

        print(f"processing frame idx {idx}")

        # get the current frame
        cur_frame = video[idx, :]
        cur_frame_pil = Image.fromarray(cur_frame)

        ############ Detection ############
        detected_objects = owl.detect_objects(
            cur_frame_pil,
            text_queries,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.12
        )   

        if len(detected_objects):
            x = cur_frame_pil.size[0]
            y = cur_frame_pil.size[1]

            detected_objects = non_maximum_suppression(detected_objects, iou_threshold=0.2)

            filtered_objects = {}
            for detection in detected_objects:
                box_name = detection['box_name']
                if box_name not in filtered_objects or detection['score'] > filtered_objects[box_name]['score']:
                    filtered_objects[box_name] = detection
            detected_objects = list(filtered_objects.values())

            print(detected_objects)

            boxes_l = [[x * obj['bbox'][0], y * obj['bbox'][1], x * obj['bbox'][2], y * obj['bbox'][3]] for obj in detected_objects]
        else:
            raise RuntimeError("no object detected")
        
        # skip if detection fails
        if len(boxes_l) < 2:
            continue
        if bbox_expansion is not None:
            for i, box in enumerate(boxes_l):
                x_min, y_min, x_max, y_max = box
                box = [x_min*(1-bbox_expansion), y_min*((1-bbox_expansion)), x_max*(1+bbox_expansion), y_max*(1+bbox_expansion)]
                boxes_l[i] = box
        
        # compute IOU between tool and target
        iou = compute_iou(boxes_l[0], boxes_l[1])
        print(f"IOU between tool and target: {iou}")
        if iou > pre_func_thresh:
            print(f"pre-function frame found at idx {idx}")
            break

    return grasp_idx, idx


def keyframe_localization(data_path, tool_label, target_label, pre_function_thresh, bbox_expansion):
             
    kf_idx = {}
    output_path = os.path.join(data_path, 'key_point_track_output')
    os.makedirs(output_path, exist_ok=True)

    key_point_track(data_path, grid_size=500)
    grasp_idx, pre_func_idx = pre_func_compute(data_path, tool_label, target_label, pre_function_thresh, bbox_expansion)

    kf_idx['grasp'] = grasp_idx
    kf_idx['pre-func'] = pre_func_idx

    keyframe_result_filename = 'keyframe_idx.json'
    with open(os.path.join(output_path, keyframe_result_filename), 'w') as file:
        json.dump(kf_idx, file)

    print("pre-function and grasping keyframe idx saved!")

    return None
   


   
 