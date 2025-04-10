import os
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
import numpy as np
import cv2
import json
import io
import base64
import requests

from utils_IL.perception_utils import fit_gaussian, map_coordinates_to_resized, map_coordinates_to_original, resize_, convert_to_serializable
from utils_IL.vp_utils import propose_candidate_keypoints, annotate_visual_prompts, load_prompts, generate_vertices_and_update, keypoint_transfer


def get_features(image):
    # Convert the image to a PNG byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Send the image as base64 to the server
    response = requests.post("http://crane1.d2.comp.nus.edu.sg:4002/process_image", json={"image": img_encoded})

    return response.json()

def grasp_point_transfer(data_path, test_data_path, test_tool_label='mug', test_target_label='bowl', task_label='pour', num_candidate_keypoints=8, sd_dino_flag=False):
    grasp_det_path = os.path.join(data_path, 'grasp_det_output')
    output_path = os.path.join(test_data_path, 'grasp_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # load prompts
    prompt_path = os.path.join(test_data_path, '..', '..', 'utils_IL', 'prompts')
    prompts = load_prompts(prompt_path)

    # load test initial keyframe
    test_init_frame_idx = 0
    test_init_frame_path = os.path.join(test_data_path, 'rgb', f'{test_init_frame_idx:05}.jpg')
    test_init_frame = cv2.imread(test_init_frame_path)
    demo_init_frame_pil_crop = Image.open(os.path.join(test_data_path, 'func_point_transfer_output', 'demo_tool_crop.png')).convert('RGB')
    demo_init_frame_crop = np.array(demo_init_frame_pil_crop)
    test_init_frame_pil_crop = Image.open(os.path.join(test_data_path, 'func_point_transfer_output', 'test_tool_crop.png')).convert('RGB')
    test_init_frame_mask_crop = np.load(os.path.join(test_data_path, 'func_point_transfer_output', 'test_tool_mask_crop.npy'))

    ############ Demo tool processing ############ 

    # load demo tool bbox
    demo_tool_box_path = os.path.join(data_path, 'detection_output', 'tool.json')
    with open(demo_tool_box_path, 'r') as file:
        demo_tool_box = json.load(file)
    x_min, y_min, x_max, y_max = demo_tool_box
    x_min *= 0.95
    y_min *= 0.95
    x_max *= 1.03
    y_max *= 1.03
    demo_tool_box = [x_min, y_min, x_max, y_max]

    # load demo grasp point
    grasp_point_path = os.path.join(grasp_det_path, 'grasp_points.json')
    with open(grasp_point_path, 'r') as json_file:
        grasp_points = json.load(json_file)
    # compute 2d demo grasp point
    mean, _ = fit_gaussian(grasp_points)
    x, y = int(mean[0]), int(mean[1])
    demo_grasp_point = [y - demo_tool_box[0],  x - demo_tool_box[1]]

    # save cropped demo grasp point
    with open(os.path.join(output_path, 'demo_init_frame_crop_grasp_point_out.json'), "w") as json_file:
        json.dump([int(demo_grasp_point[0]), int(demo_grasp_point[1])], json_file, default=convert_to_serializable, indent=4)
    scatter_size = 3
    cv2.circle(demo_init_frame_crop, (int(demo_grasp_point[0]), int(demo_grasp_point[1])), scatter_size, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_path, 'demo_init_frame_grasp_point_vis.jpg'), demo_init_frame_crop)

    ############ Test tool processing ############ 

    # load test tool bbox
    test_tool_box_path = os.path.join(test_data_path, 'detection_output', 'tool.json')
    with open(test_tool_box_path, 'r') as file:
        test_tool_box = json.load(file)
    x_min, y_min, x_max, y_max = test_tool_box
    x_min *= 0.95
    y_min *= 0.95
    x_max *= 1.03
    y_max *= 1.03
    test_tool_box = [x_min, y_min, x_max, y_max]
            
    ############ Coarse-grained region proposal ############ 
    segmasks = {}
    segmasks[test_tool_label] = {'mask': test_init_frame_mask_crop}

    task_instruction = f'use a {test_tool_label} to {task_label} {test_target_label}'

    # ignore target objects
    task = {}
    task['object_grasped'] = test_tool_label
    task['object_unattached'] = ''
    task['task_instruction'] = task_instruction

    segmasks = generate_vertices_and_update(segmasks)

    # Annotate visual marks.
    candidate_keypoints = propose_candidate_keypoints(
        task,
        segmasks, 
        num_samples=num_candidate_keypoints)

    annotated_image = annotate_visual_prompts(
                test_init_frame_pil_crop,
                candidate_keypoints)
    annotated_image.save(os.path.join(output_path, 'visual_prompting_candidates.png'))
    
    print("zero-shot grasp point transfer")
    context, context_cor, vp_img = keypoint_transfer(
        task,
        test_init_frame_pil_crop,
        annotated_image,
        candidate_keypoints,
        prompts=prompts['select_motion_grasp'], 
        debug=True,
        grasp=True)
    
    print(task)
    print(context)

    vp_grasp_point = context_cor['selected_keypoint']
    vp_img.save(os.path.join(output_path, 'visual_prompting_result.png'))

    print("coarse-grained region proposal done.")

    x_offset, y_offset, _, _ = test_tool_box
    if sd_dino_flag:
        ############## Fine-grained point transfer ##############
        # load demo and test images
        img_size = 480
        img1, sf1, pd1 = resize_(demo_init_frame_pil_crop.convert('RGB'), target_res=img_size, resize=True, to_pil=True)
        img2, sf2, pd2 = resize_(test_init_frame_pil_crop.convert('RGB'), target_res=img_size, resize=True, to_pil=True)

        demo_source_x, demo_source_y = map_coordinates_to_resized(demo_grasp_point[0], demo_grasp_point[1], sf1, pd1)
        demo_source_x, demo_source_y = np.int64(demo_source_x), np.int64(demo_source_y)
        test_source_x, test_source_y = map_coordinates_to_resized(vp_grasp_point[0], vp_grasp_point[1], sf2, pd2)
        test_source_x, test_source_y = np.int64(test_source_x), np.int64(test_source_y)

        # create vp mask
        x = np.arange(0, img_size, 1)
        y = np.arange(0, img_size, 1)
        x, y = np.meshgrid(x, y)
        scale = 0.15
        rect_width = (img_size -  pd2[1]*2) * scale # Width of the rectangle
        rect_height = (img_size -  pd2[0]*2) * scale  # Height of the rectangle
        rect_center_x = test_source_x  # X-coordinate of rectangle center
        rect_center_y = test_source_y  # Y-coordinate of rectangle center
        # Calculate rectangle boundaries
        x_min = rect_center_x - rect_width / 2
        x_max = rect_center_x + rect_width / 2
        y_min = rect_center_y - rect_height / 2
        y_max = rect_center_y + rect_height / 2
        # Create the rectangle mask
        vp_mask_normalized = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max), 1, 0)

        mask2_raw = Image.fromarray(test_init_frame_mask_crop)
        mask2, sf_m1, pd_m1 = resize_(mask2_raw, target_res=img_size, resize=True, to_pil=False)

        # grasp point transfer with SD+DINO
        response_1 = get_features(img1)
        features_bytes_1 = base64.b64decode(response_1['features'])
        feat1 = np.load(io.BytesIO(features_bytes_1))
        response_2 = get_features(img2)
        features_bytes_2 = base64.b64decode(response_2['features'])
        feat2 = np.load(io.BytesIO(features_bytes_2))
        feat1_cuda = torch.tensor(feat1).to('cuda')
        feat2_cuda = torch.tensor(feat2).to('cuda')
        ft = torch.cat([feat1_cuda, feat2_cuda], dim=0)

        # compute cosine similarity map
        num_channel = ft.size(1)
        cos = nn.CosineSimilarity(dim=1)
        src_ft = ft[0].unsqueeze(0)  # []
        src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
        src_vec = src_ft[0, :, demo_source_y, demo_source_x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
        trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(ft[1:]) # 1, C, H, W
        cos_map = cos(src_vec, trg_ft).cpu().numpy()    # 1, H, W
        # search correspondence within the test tool mask
        cos_map = np.multiply(mask2, cos_map)
        cos_map = np.multiply(vp_mask_normalized, cos_map)  # no vp

        print("fine-grained point transfer done.")

        # compute test grasp point
        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
                
        # map to the uncropped image
        original_x, original_y = map_coordinates_to_original(int(max_yx[1]), int(max_yx[0]), sf2, pd2)
        test_frame_x, test_frame_y = int(original_x + x_offset), int(original_y + y_offset) # VP + SD+DINO
    else:
        test_frame_x, test_frame_y = int(vp_grasp_point[0] + x_offset), int(vp_grasp_point[1] + y_offset)  # no SD+DINO, VP only
        
    cv2.circle(test_init_frame, (test_frame_x, test_frame_y), 3, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_path, 'test_init_frame_grasp_point_vis.jpg'), cv2.cvtColor(test_init_frame, cv2.COLOR_BGR2RGB))

    with open(os.path.join(output_path, 'test_init_frame_grasp_point_out.json'), 'w') as json_file:
        json.dump([test_frame_x, test_frame_y], json_file, indent=4)

    print("grasp point transfer done.")



 
    


