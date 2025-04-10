import numpy as np
import os
import io
import requests
import base64
import json
import cv2
from io import BytesIO
from PIL import Image
import traceback
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

DEFAULT_LLM_MODEL_NAME = 'gpt-4'
DEFAULT_VLM_MODEL_NAME = 'gpt-4o'

api_key = os.environ['OPENAI_API_KEY']

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

def load_prompts(prompt_dir):
    """
    Load prompts from files.
    """
    prompts = dict()
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path[-4:] == '.txt':
            with open(path, 'r') as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts

def generate_vertices_and_update(segmasks):
    for obj_name, obj_data in segmasks.items():
        mask = obj_data['mask'].astype(np.uint8)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Collect vertices
        obj_vertices = []
        for contour in contours:
            for point in contour:
                obj_vertices.append(point[0])
        
        # Update the dictionary with vertices
        segmasks[obj_name]['vertices'] = np.array(obj_vertices)
    
    return segmasks

def point_2_object_dis(vertices, mask):
    """
    compute the distance distribution of candidate function points
    """
    vertices_dis_list = []
    mask_points = np.argwhere(mask)
    for ver in vertices:
        distances = distance.cdist([ver], mask_points, metric='euclidean')
        ver_dis = np.sum(distances)
        vertices_dis_list.append(ver_dis)
    
    vertices_dis_array = np.array(vertices_dis_list)
    mean_value = np.mean(vertices_dis_array)
    std_value = np.std(vertices_dis_array)
    normalized_vertices_dis_array = (vertices_dis_array - mean_value) / std_value

    # apply softmax to get a distribution
    exp_values = np.exp(normalized_vertices_dis_array)
    softmax_values = exp_values / np.sum(exp_values)
    
    return softmax_values

def fps(points, n_samples):
    """
    Farthest Point Sampling (FPS) with inclusion of the leftmost and rightmost points.
    
    Args:
        points: [N, 2] array containing the whole point cloud.
        n_samples: Number of points to sample, including the leftmost and rightmost points.
    
    Returns:
        Sampled points: [n_samples, 2] array of the sampled point cloud.
    """
    points = np.array(points)

    # Find indices of the leftmost and rightmost points
    leftmost_idx = np.argmin(points[:, 0])
    rightmost_idx = np.argmax(points[:, 0])

    # Initialize sampled indices with the leftmost and rightmost points
    sample_inds = [rightmost_idx, leftmost_idx]
    
    # Reduce the number of points to sample since we've already included two
    remaining_samples = n_samples - 2

    # Represent the points by their indices in points
    points_left = np.delete(np.arange(len(points)), sample_inds)

    # Initialize distances to inf
    dists = np.ones(len(points_left)) * float("inf")

    # Iteratively select the remaining points
    for _ in range(remaining_samples):
        # Calculate distances from the last added sampled point
        last_added = sample_inds[-1]
        dist_to_last_added = ((points[last_added] - points[points_left]) ** 2).sum(axis=1)

        # Update distances to reflect the nearest sampled point
        dists = np.minimum(dists, dist_to_last_added)

        # Select the point with the maximum distance
        farthest_idx = np.argmax(dists)
        sample_inds.append(points_left[farthest_idx])

        # Remove the selected point from points_left and corresponding distance
        points_left = np.delete(points_left, farthest_idx)
        dists = np.delete(dists, farthest_idx)

    # Return the sampled points
    return points[sample_inds]

def get_keypoints_from_segmentation(segmasks, subtask, num_samples=5, include_center=True, dis_thresh=0.1):
    """
    Args:
        mask: a dict of segmentation masks, each mask is a dict with keys: 'mask', 'bbox', 'score'
        objects: a list of objects, each string is a query
    """
    object_vertices = {}
    for object_name in segmasks.keys():
        vertices = segmasks[object_name]["vertices"]
        mask = segmasks[object_name]["mask"]

        center_point = vertices.mean(0)
        # if mask[int(center_point[1])][
        #     int(center_point[0])
        # ] and include_center:  # ignore if geometric mean is not in mask
        if include_center:
            vertices = np.concatenate([center_point[None, ...], vertices], axis=0)

        if vertices.shape[0] > num_samples:
            kps = fps(vertices, num_samples)
        else:
            kps = vertices

        kps = np.concatenate([kps[[0]], kps[1:][kps[1:, 1].argsort()]], axis=0)
        object_vertices[object_name] = kps

    tool_kps = object_vertices[subtask['object_grasped']]
    if subtask['object_unattached'] != '':
        target_mask = segmasks[subtask['object_unattached']]['mask']
        kps_dis_dist = point_2_object_dis(tool_kps, target_mask)

        # remove points far away
        mask = kps_dis_dist <= dis_thresh
        filtered_tool_kps = tool_kps[mask]
    else:
        filtered_tool_kps = tool_kps
    object_vertices[subtask['object_grasped']] = filtered_tool_kps

    return object_vertices

def propose_candidate_keypoints(subtask, segmasks, num_samples):
    """
    Propose candidate keypoints for object_grasped and object_unattached.
    """
    input_object_names = []
    if subtask['object_grasped'] != '':
        input_object_names.append(subtask['object_grasped'])

    if subtask['object_unattached'] != '':
        input_object_names.append(subtask['object_unattached'])

    object_vertices = get_keypoints_from_segmentation(
        segmasks,
        subtask,
        num_samples=num_samples,
        include_center=False)

    candidate_keypoints = {
        'grasped': None,
        'unattached': None,
    }

    if subtask['object_grasped'] != '':
        candidate_keypoints['grasped'] = object_vertices[
            subtask['object_grasped']
        ]

    return candidate_keypoints

def plot_keypoints(
        ax, image_size, keypoints, color, prefix='', annotate_index=True,
        add_caption=True):
    if keypoints is None:
        return
        
    (h, w) = image_size
    for i, keypoint in enumerate(keypoints):
        if keypoint is None:
            continue

        ax.plot(
            keypoint[0], keypoint[1],
            color=color, alpha=0.4,
            marker='o', markersize=15,
            markeredgewidth=2, markeredgecolor='black',
        )

        if add_caption:
            text = ''
            if annotate_index:
                text = text + str(i + 1)
            text = prefix + text

            xytext = (
                min(max(30, keypoint[0]), h - 30),
                min(max(30, keypoint[1]), w - 30),
            )

            if prefix == 'P':
                # ax.annotate(text, keypoint, xytext, size=15, color='white')
                annotation = ax.annotate(text, keypoint, xytext, size=15)
                annotation.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()])
            else:
                ax.annotate(text, keypoint, xytext, size=15, color='white')
            # ax.annotate(text, keypoint, xytext, size=15)

def annotate_candidate_keypoints(
        image,
        candidate_keypoints,
        add_caption=True
):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]

    if 'grasped' in candidate_keypoints.keys():
        plot_keypoints(
            ax, image_size, candidate_keypoints['grasped'], 'r', prefix='P', add_caption=add_caption)

    if 'unattached' in candidate_keypoints.keys():
        plot_keypoints(
            ax, image_size, candidate_keypoints['unattached'], 'b', prefix='Q', add_caption=add_caption)

    buf = io.BytesIO()
    # fig.savefig(buf, transparent=True, bbox_inches='tight',
    #             pad_inches=0, format='jpg', dpi=int(image_size[0] / fig.get_size_inches()[0]))
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg', dpi=int(image_size[0]))
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)

def annotate_visual_prompts(
        obs_image,
        candidate_keypoints,
        log_dir=None,
):
    """
    nnotate the visual prompts on the image.
    """

    annotated_image = annotate_candidate_keypoints(
        obs_image,
        candidate_keypoints,
    )

    if log_dir is not None:
        annotated_image.save(os.path.join(log_dir, 'keypoints.png'))
    else:
        plt.imshow(annotated_image)
        plt.show()

    return annotated_image

def encode_image_from_file(image_path):
    # Function to encode the image
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def remove_trailing_comments(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Process each line to remove comments
    processed_lines = []
    for line in lines:
        comment_index = line.find('//')
        if comment_index != -1:
            # Remove the comment
            line = line[:comment_index]
        processed_lines.append(line.strip())

    # Join the processed lines back into a single string
    return """{}""".format('\n'.join(processed_lines))

def parse_json_string(res, verbose=False):
    if '```' in res:
        try:
            res_clean = res

            if '```json' in res:
                res_clean = res_clean.split('```')[1].split('json')[1]
            elif '```JSON' in res:
                res_clean = res_clean.split('```')[1].split('JSON')[1]
            elif '```' in res:
                res_clean = res_clean.split('```')[1]
            else:
                print('Invalid response: ')
                print(res)

        except Exception:
            print(traceback.format_exc())
            print('Invalid response: ')
            print(res)
            return None
    else:
        res_clean = res

    try:
        res_filtered = remove_trailing_comments(res_clean)
        object_info = json.loads(res_filtered)

        # if verbose:
        #     print_object_info(object_info)

        return object_info

    except Exception:
        print(traceback.format_exc())
        print('The original response: ')
        print(res)
        print('Invalid cleaned response: ')
        print(res_clean)
        return None

def prepare_inputs(messages,
                   images,
                   meta_prompt,
                   model_name,
                   local_image):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)
    
    if not isinstance(images, list):
        images = [images]
    
    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800,
        # 'temperature': 0,
        # 'top_p': 0.1
    }

    return payload

def request_gpt(message,
                images,
                meta_prompt='',
                model_name=None,
                local_image=False):

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs(message,
                             images,
                             meta_prompt=meta_prompt,
                             model_name=model_name,
                             local_image=local_image)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception:
        print('\nInvalid response: ')
        print(response)
        print('\nInvalid response: ')
        print(response.json())
        exit()

    return res

def prepare_inputs_incontext(
        messages,
        images,
        meta_prompt,
        model_name,
        local_image,
        example_images,
        example_responses,
):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for example_image, example_response in zip(
            example_images, example_responses):
        if local_image:
            base64_image = encode_image_from_file(example_image)
        else:
            base64_image = encode_image_from_pil(example_image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

        content = {
            'type': 'text',
            'text': example_response,
        }
        user_content.append(content)
    
    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload

def request_gpt_incontext(
        message,
        images,
        meta_prompt='',
        example_images=None,
        example_responses=None,
        model_name=None,
        local_image=False):

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs_incontext(
        message,
        images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
        example_images=example_images,
        example_responses=example_responses)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception:
        print('\nInvalid response: ')
        print(response)
        print('\nInvalid response: ')
        print(response.json())
        exit()

    return res

def annotate_motion(image, context, log_dir=None, add_caption=True):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]

    if 'keypoints_2d' in context.keys():
        function_keypoint = context['keypoints_2d']['function']
    else:
        function_keypoint = context['selected_keypoint']

    plot_keypoints(
        ax, image_size, [function_keypoint], 'y', 'function', False,
        add_caption=add_caption)

    plt.show()
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)

    return Image.open(buf)

def request_motion(  # NOQA
        subtask,
        obs_image_reshaped,
        annotated_image,
        candidate_keypoints,
        prompts,
        debug=False,
        example_images=None,
        example_responses=None,
        loaded_context=None,
        use_center=False,
        log_dir=None,
        suffix='',
        add_caption=True,
        grasp=False
):
    """
    Generate the visual marks that specify the motion.
    """
    # annotation_size = obs_image_reshaped.size[:2]
    # print('annotation_size (request_motion)', annotation_size)

    # import pdb;pdb.set_trace()

    if grasp:
        selection = 'select_motion_grasp'
    else:
        selection = 'select_motion_func'

    text_requests = [f"Task: {subtask}"]

    if loaded_context is not None:
        context = loaded_context
    else:
        context = None
        while context is None:
            if (example_images is not None and
                    example_responses is not None):
                print('example_respnses:')
                print(example_responses)
                assert len(example_images) == len(example_responses)
                res = request_gpt_incontext(
                    text_requests,
                    [annotated_image],
                    prompts[selection],
                    example_images=example_images,
                    example_responses=example_responses)
            else:
                res = request_gpt(
                    text_requests,
                    [annotated_image],
                    prompts[selection])

            if debug:
                print('--------------------------------')
                print('| Selected keypoint.')
                print('--------------------------------')
                print(res)
            context = parse_json_string(res)

    context_json = context
   
    detected_keypoint = None
    # detect grasp point
    if grasp:
        # grasp_keypoint = None
        val = context['grasp_keypoint']
    # detect function point
    else:
        # function_keypoint = None
        val = context['function_keypoint']
    if val != '':
        idx = int(val[1:]) - 1
        detected_keypoint = candidate_keypoints['grasped'][idx]

    context = dict(
        keypoints_2d=dict(
            function=detected_keypoint,
        )
    )

    if debug:
        # if True:
        log_img = annotate_motion(obs_image_reshaped, context,
                                  add_caption=add_caption)
        if log_dir is not None:
            log_img.save(os.path.join(log_dir, f'motion{suffix}.png'))
            out_file = open(os.path.join(log_dir, f'context{suffix}.json'), 'w')
            json.dump(context_json, out_file)

        return context, context_json, log_img
    else:
        return context, context_json, None

def keypoint_transfer(  # NOQA
        subtask,
        obs_image_reshaped,
        annotated_image,
        candidate_keypoints,
        prompts,
        waypoint_grid_size=None,
        debug=False,
        example_images=None,
        example_responses=None,
        loaded_context=None,
        use_center=False,
        log_dir=None,
        suffix='',
        add_caption=False,
        grasp=False
):
    """Generate the visual marks that specify the motion.
    """
    text_requests = [f"Task: {subtask}"]

    if loaded_context is not None:
        context = loaded_context
    else:
        context = None
        while context is None:
            if (example_images is not None and
                    example_responses is not None):
                print('example_respnses:')
                print(example_responses)
                assert len(example_images) == len(example_responses)
                res = request_gpt_incontext(
                    text_requests,
                    [annotated_image],
                    prompts,
                    example_images=example_images,
                    example_responses=example_responses)
            else:
                # res = request_gpt(
                #     text_requests,
                #     [annotated_image],
                #     prompts[selection])
                res = request_gpt(
                    text_requests,
                    annotated_image,
                    prompts)
            if debug:
                print('--------------------------------')
                print('| Selected keypoint.')
                print('--------------------------------')
                print(res)
            context = parse_json_string(res)

    if grasp:
        context_cor = {'selected_keypoint': candidate_keypoints['grasped'][int(context['grasp_keypoint'][1:])-1]}
    else:
        context_cor = {'selected_keypoint': candidate_keypoints['grasped'][int(context['function_keypoint'][1:])-1]}
    
    if debug:
        # if True:
        log_img = annotate_motion(obs_image_reshaped, context_cor,
                                  add_caption=add_caption)

    return context, context_cor, log_img

def pose_selection(  # NOQA
        subtask,
        obs_image,
        prompts,
        debug=False,
):
    """Select the pose that supports the task
    """

    text_requests = [f"Task: {subtask}"]

    context = None
    while context is None:
        res = request_gpt(
            text_requests,
            obs_image,
            prompts)

        if debug:
            print('--------------------------------')
            print('| Selected pose.')
            print('--------------------------------')
            print(res)
        context = parse_json_string(res)
    
    return context
    

    
