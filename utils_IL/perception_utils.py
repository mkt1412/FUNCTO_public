import numpy as np
import re
from PIL import Image
import os
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

def non_maximum_suppression(detected_objects, iou_threshold=0.2):
    if len(detected_objects) == 0:
        return []

    scores = [obj['score'] for obj in detected_objects]
    boxes = [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in detected_objects]
    
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Compute the area of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort the bounding boxes by their scores in descending order
    order = areas.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(detected_objects[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # print(iou)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def non_maximum_suppression_class(detected_objects, iou_threshold=0.2):
    if len(detected_objects) == 0:
        return []

    # Group detected objects by their class (box_name)
    objects_by_class = {}
    for obj in detected_objects:
        box_name = obj['box_name']
        if box_name not in objects_by_class:
            objects_by_class[box_name] = []
        objects_by_class[box_name].append(obj)

    # Apply NMS separately for each class
    final_detections = []
    for box_name, objects in objects_by_class.items():
        scores = [obj['score'] for obj in objects]
        boxes = [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in objects]

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Compute the area of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort the bounding boxes by their scores in descending order
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(objects[i])

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        final_detections.extend(keep)

    return final_detections

def fit_gaussian(points):
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    return mean, cov

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def read_frames_from_path(rgb_folder, depth_folder):
    # Use regular expression to find the numeric part of the filename
    def extract_timestamp(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    # Get list of RGB and depth file paths
    rgb_path_list = [os.path.join(rgb_folder, fname) for fname in os.listdir(rgb_folder) if fname.endswith('.jpg')]
    depth_path_list = [os.path.join(depth_folder, fname) for fname in os.listdir(depth_folder) if fname.endswith('.png')]

    # Sort the lists based on the extracted numeric part
    rgb_path_list_sorted = sorted(rgb_path_list, key=lambda x: extract_timestamp(os.path.basename(x)))
    depth_path_list_sorted = sorted(depth_path_list, key=lambda x: extract_timestamp(os.path.basename(x)))

    # Initialize an empty list to store the images
    rgb_list = []
    depth_list = []

    # Loop through each image path in the list and load the RGB images
    for rgb_path in rgb_path_list_sorted:
        with Image.open(rgb_path) as img:
            img = img.convert('RGB')
            img_array = np.array(img)
            rgb_list.append(img_array)
    
    for depth_path in depth_path_list_sorted:
        with Image.open(depth_path) as img:
            depth_array = np.array(img)
            depth_list.append(depth_array)

    stacked_rgbs = np.stack(rgb_list, axis=0)  
    stacked_depths = np.stack(depth_list, axis=0) 
    
    return stacked_rgbs, stacked_depths

def read_frames_from_path_npy(rgb_folder, depth_folder):
    # Use regular expression to find the numeric part of the filename
    def extract_timestamp(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    # Get list of RGB and depth file paths
    rgb_path_list = [os.path.join(rgb_folder, fname) for fname in os.listdir(rgb_folder) if fname.endswith('.jpg')]
    depth_path_list = [os.path.join(depth_folder, fname) for fname in os.listdir(depth_folder) if fname.endswith('.npy')]

    # Sort the lists based on the extracted numeric part
    rgb_path_list_sorted = sorted(rgb_path_list, key=lambda x: extract_timestamp(os.path.basename(x)))
    depth_path_list_sorted = sorted(depth_path_list, key=lambda x: extract_timestamp(os.path.basename(x)))

    # Initialize an empty list to store the images
    rgb_list = []
    depth_list = []

    # Loop through each image path in the list and load the RGB images
    for rgb_path in rgb_path_list_sorted:
        with Image.open(rgb_path) as img:
            img = img.convert('RGB')
            img_array = np.array(img)
            rgb_list.append(img_array)
    
    # Load depth images from npy files
    for depth_path in depth_path_list_sorted:
        depth_array = np.load(depth_path)
        depth_list.append(depth_array)

    stacked_rgbs = np.stack(rgb_list, axis=0)  
    stacked_depths = np.stack(depth_list, axis=0) 
    
    return stacked_rgbs, stacked_depths

def map_coordinates_to_original(x_resized, y_resized, scale_factor, padding):
    x_original = (x_resized - padding[0]) / scale_factor
    y_original = (y_resized - padding[1]) / scale_factor
    return x_original, y_original

def map_coordinates_to_resized(x_original, y_original, scale_factor, padding):
    x_resized = x_original * scale_factor + padding[0]
    y_resized = y_original * scale_factor + padding[1]
    return x_resized, y_resized

def resize_(img, target_res=224, resize=True, to_pil=True, edge=False):
    # Convert NumPy array to PIL Image if necessary
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            # img = (img * 255).astype(np.uint8)
            img = img.astype(np.uint8)
        img = Image.fromarray(img)
    
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    scale_factor = 1.0
    padding = [0, 0]

    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                scale_factor = target_res / original_width
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[1] = (target_res - height) // 2
            canvas[padding[1]: padding[1] + height] = img
        else:
            if resize:
                scale_factor = target_res / original_height
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[0] = (target_res - width) // 2
            canvas[:, padding[0]: padding[0] + width] = img
    else:
        if original_height <= original_width:
            if resize:
                scale_factor = target_res / original_width
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[1] = (target_res - height) // 2
            bottom_pad = target_res - height - padding[1]
            img = np.pad(img, pad_width=[(padding[1], bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                scale_factor = target_res / original_height
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[0] = (target_res - width) // 2
            right_pad = target_res - width - padding[0]
            img = np.pad(img, pad_width=[(0, 0), (padding[0], right_pad), (0, 0)], mode='edge')
        canvas = img
    
    if to_pil:
        canvas = Image.fromarray(canvas)
    
    # Return additional information needed for mapping
    return canvas, scale_factor, padding

def resize_w_color_(img, target_res=224, resize=True, to_pil=True, edge=False, padding_color=(255, 255, 255)):
    # Convert NumPy array to PIL Image if necessary
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = Image.fromarray(img)
    
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    scale_factor = 1.0
    padding = [0, 0]

    if not edge:
        canvas = np.full([target_res, target_res, 3], padding_color, dtype=np.uint8)
        if original_channels == 1:
            canvas = np.full([target_res, target_res], padding_color[0], dtype=np.uint8)  # Grayscale padding
        if original_height <= original_width:
            if resize:
                scale_factor = target_res / original_width
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[1] = (target_res - height) // 2
            canvas[padding[1]: padding[1] + height] = img
        else:
            if resize:
                scale_factor = target_res / original_height
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[0] = (target_res - width) // 2
            canvas[:, padding[0]: padding[0] + width] = img
    else:
        if original_height <= original_width:
            if resize:
                scale_factor = target_res / original_width
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[1] = (target_res - height) // 2
            bottom_pad = target_res - height - padding[1]
            img = np.pad(img, pad_width=[(padding[1], bottom_pad), (0, 0), (0, 0)], mode='constant', constant_values=[padding_color])
        else:
            if resize:
                scale_factor = target_res / original_height
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            padding[0] = (target_res - width) // 2
            right_pad = target_res - width - padding[0]
            img = np.pad(img, pad_width=[(0, 0), (padding[0], right_pad), (0, 0)], mode='constant', constant_values=[padding_color])
        canvas = img
    
    if to_pil:
        canvas = Image.fromarray(canvas)
    
    # Return additional information needed for mapping
    return canvas, scale_factor, padding

def compute_iou(box1, box2):
    # Extract the coordinates of the bounding boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Calculate the area of the intersection rectangle
    inter_width = max(inter_x_max - inter_x_min, 0)
    inter_height = max(inter_y_max - inter_y_min, 0)
    inter_area = inter_width * inter_height
    
    # Calculate the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate the union area
    union_area = box1_area + box2_area - inter_area
    
    # Compute the Intersection over Union (IoU)
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def backproject_3d_2d(point_3d, rotation_matrix, rotation_center, centroid, cam_matrix):
     # Inverse the translation by adding the centroid
    translated_back_point = point_3d + centroid

    # Compute the inverse of the rotation matrix
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)

    # Reverse the rotation about the rotation center
    # Shift the point to the center, apply the inverse rotation, and shift back
    original_point_3d = np.dot(rotation_matrix_inv, (translated_back_point - rotation_center)) + rotation_center

    point_2d_homogeneous = np.dot(cam_matrix, original_point_3d)

    # Normalize to get 2D pixel coordinates
    u = int(point_2d_homogeneous[0] / point_2d_homogeneous[2])
    v = int(point_2d_homogeneous[1] / point_2d_homogeneous[2])

    return (u, v)

def batch_cosine_similarity(feature_matrix1, feature_matrix2, batch_size=64):

    feature_matrix1 = feature_matrix1.view(768, -1).transpose(0, 1)  # [480*480, 768]
    feature_matrix2 = feature_matrix2.view(768, -1).transpose(0, 1)  # [480*480, 768]

    feature_matrix1_norm = F.normalize(feature_matrix1, dim=1)
    feature_matrix2_norm = F.normalize(feature_matrix2, dim=1)

    num_samples = feature_matrix1_norm.shape[0]
    cosine_similarity_matrix = torch.zeros((num_samples, num_samples), device=feature_matrix1_norm.device)

    for i in range(0, num_samples, batch_size):
        end_i = min(i + batch_size, num_samples)
        batch1 = feature_matrix1_norm[i:end_i]
        
        for j in range(0, num_samples, batch_size):
            end_j = min(j + batch_size, num_samples)
            batch2 = feature_matrix2_norm[j:end_j]
            
            cosine_similarity_matrix[i:end_i, j:end_j] = torch.mm(batch1, batch2.transpose(0, 1))

    return cosine_similarity_matrix

def map_block_indices_to_original_coordinates(x_indices, y_indices, num_blocks, img_dimension):

    block_size = img_dimension // num_blocks  
    original_x = (x_indices * block_size) + (block_size // 2)
    original_y = (y_indices * block_size) + (block_size // 2)

    return original_x, original_y

def find_closest_points(x, y, best_img1_xy, N):
    
    candidates = np.array([best_img1_xy[0], best_img1_xy[1]]).T
    distances = np.sqrt((candidates[:, 0] - x) ** 2 + (candidates[:, 1] - y) ** 2)
    # Get the indices of the N smallest distances
    closest_indices = np.argsort(distances)[:N]
    # Get the N closest distances
    closest_distances = distances[closest_indices]
    
    return closest_indices, closest_distances

def global_pose_alignment(ft, ft_dino_1, ft_dino_2, mask1, mask2, img_size=480):
    """
    wenlong code
    """
    src_ft = ft[0].unsqueeze(0) #1*768*60*60
    trg_ft = ft[1].unsqueeze(0)
    # src_ft_up = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)#1*768*480*480
    # trg_ft_up = nn.Upsample(size=(img_size, img_size), mode='bilinear')(ft[1:]) # 1, C, H, W

    avg_tensor1 = ft_dino_1.mean(dim=1)
    avg_tensor2 = ft_dino_2.mean(dim=1)
    saliency_map1 = avg_tensor1.view(-1)
    saliency_map2 = avg_tensor2.view(-1)
    fg_mask1 = mask1
    fg_mask2 = mask2

    cosine_sim_matrix = batch_cosine_similarity(src_ft, trg_ft, batch_size=1024)#3600*3600
    cosine_sim_matrix = cosine_sim_matrix.unsqueeze(0).unsqueeze(0)
    
    num_patches1 = (60,60)
    num_patches2 = (60,60)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device='cuda')
    sim_1, nn_1 = torch.max(cosine_sim_matrix, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(cosine_sim_matrix, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0] #4015
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0] #3080
    bbs_mask = nn_2[nn_1] == image_idxs #4015

    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device='cuda')
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)#4015

    descriptors1 = src_ft.permute(0, 2, 3, 1)  # 1 x 60 x 60 x 768
    descriptors1 = descriptors1.reshape(1, 1, 3600, 768)
    descriptors2 = src_ft.permute(0, 2, 3, 1)  # 1 x 60 x 60 x 768
    descriptors2 = descriptors2.reshape(1, 1, 3600, 768)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(100, len(all_keys_together))  # if not enough pairs, show all found pairs.
    # n_clusters = min(20, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)  #10

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i
    
    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
    bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device='cuda')[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]

    # coordinates in descriptor map coordinate
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()

    # coordinates in 480 image coordinate
    original_img1_x, original_img1_y = map_block_indices_to_original_coordinates(img1_x_to_show, img1_y_to_show, 60, 480)
    original_img2_x, original_img2_y = map_block_indices_to_original_coordinates(img2_x_to_show, img2_y_to_show, 60, 480)


    # compute IMS (instance matching similarity) using two descriptors. the higher, the better
    ims = 0
    for i in range(len(img1_indices_to_show)):
        matching_dist = cosine_sim_matrix[0][0][img1_indices_to_show[i], img2_indices_to_show[i]]
        ims += matching_dist

    return ims, [original_img1_x, original_img1_y], [original_img2_x, original_img2_y]


def global_pose_alignment_viz(img1, img2, ft, ft_dino_1, ft_dino_2, mask1, mask2, img_size=480):
    """
    wenlong code
    """
    src_ft = ft[0].unsqueeze(0) #1*768*60*60
    trg_ft = ft[1].unsqueeze(0)
    # src_ft_up = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)#1*768*480*480
    # trg_ft_up = nn.Upsample(size=(img_size, img_size), mode='bilinear')(ft[1:]) # 1, C, H, W

    avg_tensor1 = ft_dino_1.mean(dim=1)
    avg_tensor2 = ft_dino_2.mean(dim=1)
    saliency_map1 = avg_tensor1.view(-1)
    saliency_map2 = avg_tensor2.view(-1)
    fg_mask1 = mask1
    fg_mask2 = mask2

    cosine_sim_matrix = batch_cosine_similarity(src_ft, trg_ft, batch_size=1024)#3600*3600
    cosine_sim_matrix = cosine_sim_matrix.unsqueeze(0).unsqueeze(0)
    
    num_patches1 = (60,60)
    num_patches2 = (60,60)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device='cuda')
    sim_1, nn_1 = torch.max(cosine_sim_matrix, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(cosine_sim_matrix, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0] #4015
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0] #3080
    bbs_mask = nn_2[nn_1] == image_idxs #4015

    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device='cuda')
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)#4015

    descriptors1 = src_ft.permute(0, 2, 3, 1)  # 1 x 60 x 60 x 768
    descriptors1 = descriptors1.reshape(1, 1, 3600, 768)
    descriptors2 = src_ft.permute(0, 2, 3, 1)  # 1 x 60 x 60 x 768
    descriptors2 = descriptors2.reshape(1, 1, 3600, 768)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(100, len(all_keys_together))  # if not enough pairs, show all found pairs.
    # n_clusters = min(20, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)  #10

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i
    
    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
    bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device='cuda')[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]

    # coordinates in descriptor map coordinate
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()

    # coordinates in 480 image coordinate
    original_img1_x, original_img1_y = map_block_indices_to_original_coordinates(img1_x_to_show, img1_y_to_show, 60, 480)
    original_img2_x, original_img2_y = map_block_indices_to_original_coordinates(img2_x_to_show, img2_y_to_show, 60, 480)

    # 将PIL图像转换为NumPy数组格式
    img1_cv = np.array(img1)
    img2_cv = np.array(img2)

    # 确保图像为BGR格式，OpenCV默认使用BGR格式
    img1_cv = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2BGR)

    # 创建一个大的画布来并排显示两个图像
    canvas = np.ones((480, 960, 3), dtype=np.uint8) * 255
    canvas[:, :480, :] = img1_cv
    canvas[:, 480:, :] = img2_cv

    # 绘制点和连接线
    for x1, y1, x2, y2 in zip(original_img1_x, original_img1_y, original_img2_x, original_img2_y):
        # 转换坐标为整数类型
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 绘制图像1的点
        cv2.circle(canvas, (x1, y1), 5, (0, 0, 255), -1)  # 红色点
        # 绘制图像2的点
        cv2.circle(canvas, (x2 + 480, y2), 5, (0, 0, 255), -1)  # 红色点
        # 连接对应的点
        cv2.line(canvas, (x1, y1), (x2 + 480, y2), (255, 0, 0), 1)  # 蓝色线

    # compute IMS (instance matching similarity) using two descriptors. the higher, the better
    ims = 0
    for i in range(len(img1_indices_to_show)):
        matching_dist = cosine_sim_matrix[0][0][img1_indices_to_show[i], img2_indices_to_show[i]]
        ims += matching_dist

    return canvas, ims, [original_img1_x, original_img1_y], [original_img2_x, original_img2_y]

def rotate_point_with_expansion(x_old, y_old, image_width, image_height, angle_degrees, expanded_image_width, expanded_image_height):
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)

    # Compute the original center of the image
    x_center_old = image_width / 2
    y_center_old = image_height / 2

    # Compute the new center of the expanded image
    x_center_new = expanded_image_width / 2
    y_center_new = expanded_image_height / 2

    # Translate point to origin (relative to the old center of the image)
    x_translated = x_old - x_center_old
    y_translated = y_old - y_center_old

    # Apply the rotation matrix for counterclockwise rotation
    x_rotated = x_translated * math.cos(angle_radians) - y_translated * math.sin(angle_radians)
    y_rotated = x_translated * math.sin(angle_radians) + y_translated * math.cos(angle_radians)

    # Translate point to the new center of the expanded image
    x_new = x_rotated + x_center_new
    y_new = y_rotated + y_center_new

    return x_new, y_new


