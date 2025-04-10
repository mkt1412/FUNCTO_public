
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

from utils_IL.perception_utils import map_coordinates_to_original, map_coordinates_to_resized, resize_
from utils_IL.perception_utils import convert_to_serializable
from utils_IL.vp_utils import propose_candidate_keypoints, annotate_visual_prompts, annotate_candidate_keypoints, load_prompts, generate_vertices_and_update, keypoint_transfer

def get_features(image):
    # Convert the image to a PNG byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Send the image as base64 to the server
    response = requests.post("http://crane1.d2.comp.nus.edu.sg:4002/process_image", json={"image": img_encoded})

    return response.json()
        
def func_point_transfer(data_path, test_data_path, test_tool_label='mug', test_target_label='bowl', task_label='pour', num_candidate_keypoints=8, sd_dino_flag=False):
    """
    Function point transfer
    """

    output_path = os.path.join(test_data_path, 'func_point_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # load prompts
    prompt_path = os.path.join(test_data_path, '..', '..', 'utils_IL', 'prompts')
    prompts = load_prompts(prompt_path)

    # load demo function point in the initial keyframe
    demo_func_point_init_path = os.path.join(data_path, 'func_point_track_output', 'func_point_2d_init.json')
    with open(demo_func_point_init_path, 'r') as file:
        demo_func_point_init = json.load(file) 
    demo_func_point_init_x, demo_func_point_init_y = demo_func_point_init

    # load demo initial keyframe
    demo_init_frame_index = 0
    demo_init_frame_filenname = f'{demo_init_frame_index:05d}.jpg'
    demo_init_frame_file_path = os.path.join(data_path, 'rgb', demo_init_frame_filenname)
    demo_init_frame_pil = Image.open(demo_init_frame_file_path).convert('RGB')
    demo_init_frame_mask_path = os.path.join(data_path, 'detection_output', 'tool_mask.npy')
    demo_init_frame_mask = np.load(demo_init_frame_mask_path)

    # load test initial keyframe
    test_init_frame_idx = 0
    test_init_frame_filename = f'{test_init_frame_idx:05d}.jpg'
    test_init_frame_file_path = os.path.join(test_data_path, 'rgb', test_init_frame_filename)
    test_init_frame_pil =  Image.open(test_init_frame_file_path).convert('RGB')
    test_init_frame = np.array(test_init_frame_pil)
    test_init_frame_mask_path = os.path.join(test_data_path, 'detection_output', 'tool_mask.npy')
    test_init_frame_mask = np.load(test_init_frame_mask_path)

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

    # crop demo tool and plot function point
    demo_init_frame_pil_crop = demo_init_frame_pil.crop(demo_tool_box)
    demo_init_frame_pil_crop.save(os.path.join(output_path, 'demo_tool_crop.png'))
    demo_func_point_2d_init_x_crop, demo_func_point_2d_init_y_crop = int(demo_func_point_init_x - demo_tool_box[0]), int(demo_func_point_init_y - demo_tool_box[1])
    demo_func_point_2d_init_crop = [demo_func_point_2d_init_x_crop, demo_func_point_2d_init_y_crop]
    demo_annotate_keypoints = {'grasped': [demo_func_point_2d_init_crop]}
    demo_init_frame_pil_crop_annotated = annotate_candidate_keypoints(demo_init_frame_pil_crop, demo_annotate_keypoints, add_caption=False)
    # hack: flip for better global alignment
    demo_init_frame_pil_crop_annotated = demo_init_frame_pil_crop_annotated.transpose(Image.FLIP_LEFT_RIGHT)
    demo_init_frame_pil_crop_annotated.save(os.path.join(output_path, 'demo_tool_func_point_init_vis.jpg'))

    # crop demo tool mask
    demo_init_frame_mask_pil = Image.fromarray(demo_init_frame_mask)
    demo_init_frame_mask_pil_crop = demo_init_frame_mask_pil.crop(demo_tool_box)
    demo_init_frame_mask_crop = np.array(demo_init_frame_mask_pil_crop).astype(np.uint8)
    np.save(os.path.join(output_path, 'demo_tool_mask_crop.npy'), demo_init_frame_mask_crop)
    demo_init_frame_mask_pil_crop.save(os.path.join(output_path, 'demo_tool_mask_crop.png'))

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

    # crop test tool and mask
    test_init_frame_pil_crop = test_init_frame_pil.crop(test_tool_box)
    test_init_frame_pil_crop.save(os.path.join(output_path, 'test_tool_crop.png'))
    test_init_frame_mask = np.load(os.path.join(test_data_path, 'detection_output', 'tool_mask.npy'))
    test_init_frame_mask_pil = Image.fromarray(test_init_frame_mask)
    test_init_frame_mask_pil_crop = test_init_frame_mask_pil.crop(test_tool_box)
    test_init_frame_mask_crop = np.array(test_init_frame_mask_pil_crop).astype(np.uint8)
    np.save(os.path.join(output_path, 'test_tool_mask_crop.npy'), test_init_frame_mask_crop)
    test_init_frame_mask_pil_crop.save(os.path.join(output_path, 'test_tool_mask_crop.png'))

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
    
    print("in-context function point transfer")
    context, context_cor, vp_img = keypoint_transfer(
        task,
        test_init_frame_pil_crop,
        [demo_init_frame_pil_crop_annotated, annotated_image],
        candidate_keypoints,
        prompts=prompts['select_motion_func_demo'], 
        debug=True)
    
    print(task)
    print(context)

    vp_func_point = context_cor['selected_keypoint']
    vp_img.save(os.path.join(output_path, 'visual_prompting_result.png'))

    print("coarse-grained region proposal done.")

    x_offset, y_offset, _, _ = test_tool_box
    if sd_dino_flag:
        ############## Fine-grained point transfer ##############
        # load demo and test images
        img_size = 480
        img1, sf1, pd1 = resize_(demo_init_frame_pil_crop.convert('RGB'), target_res=img_size, resize=True, to_pil=True)
        img2, sf2, pd2 = resize_(test_init_frame_pil_crop.convert('RGB'), target_res=img_size, resize=True, to_pil=True)

        # load test tool mask
        mask2_raw = test_init_frame_mask_pil_crop
        mask2, sf_m1, pd_m1 = resize_(mask2_raw, target_res=img_size, resize=True, to_pil=False)

        demo_source_x, demo_source_y = map_coordinates_to_resized(demo_func_point_2d_init_crop[0], demo_func_point_2d_init_crop[1], sf1, pd1)
        demo_source_x, demo_source_y = np.int64(demo_source_x), np.int64(demo_source_y)
        test_source_x, test_source_y = map_coordinates_to_resized(vp_func_point[0], vp_func_point[1], sf2, pd2)
        test_source_x, test_source_y = np.int64(test_source_x), np.int64(test_source_y)

        # create vp region mask
        x = np.arange(0, img_size, 1)
        y = np.arange(0, img_size, 1)
        x, y = np.meshgrid(x, y)
        scale = 0.15
        rect_width = (img_size -  pd2[0]*2) * scale # Width of the rectangle
        rect_height = (img_size -  pd2[1]*2) * scale  # Height of the rectangle
        rect_center_x = test_source_x  # X-coordinate of rectangle center
        rect_center_y = test_source_y  # Y-coordinate of rectangle center
        # Calculate region boundaries
        x_min = rect_center_x - rect_width / 2
        x_max = rect_center_x + rect_width / 2
        y_min = rect_center_y - rect_height / 2
        y_max = rect_center_y + rect_height / 2
        # Create the region mask
        vp_mask_normalized = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max), 1, 0)

        # function point transfer with SD+DINO
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
        # search correspondence within the test tool region mask
        cos_map = np.multiply(mask2, cos_map)
        cos_map = np.multiply(vp_mask_normalized, cos_map)  # no vp

        print("fine-grained point transfer done.")

        # compute test function point
        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
                    
        # test function point visualization
        original_x, original_y = map_coordinates_to_original(int(max_yx[1]), int(max_yx[0]), sf2, pd2)
        test_init_frame_x, test_init_frame_y = int(original_x + x_offset), int(original_y + y_offset) # VP + SD+DINO
    else:
        test_init_frame_x, test_init_frame_y = int(vp_func_point[0] + x_offset), int(vp_func_point[1] + y_offset)  # no SD+DINO, VP only

    cv2.circle(test_init_frame, (test_init_frame_x, test_init_frame_y), 3, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_path, 'test_init_frame_func_point_vis.jpg'), test_init_frame)

    with open(os.path.join(output_path, 'test_init_frame_func_point_out.json'), "w") as json_file:
        json.dump([test_init_frame_x, test_init_frame_y], json_file, default=convert_to_serializable, indent=4)

    print("function point transfer done.")